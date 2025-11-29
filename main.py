from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("operartis_forecaster")

app = FastAPI(title="Operartis Sales Forecaster")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HEALTH CHECK ---
@app.get("/")
def health_check():
    return {"status": "online", "system": "Operartis AI Engine Ready"}

def metrics(residuals):
    residuals = residuals.dropna()
    if len(residuals) == 0:
        return float('inf'), float('inf'), float('inf')
    mse = np.mean((residuals)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    return mse, rmse, mae

@app.post("/predict")
async def predict_sales(
    file: UploadFile = File(...),
    model_type: str = Form("auto") 
):
    try:
        logger.info(f"--- NEW JOB: {file.filename} [{model_type}] ---")
        contents = await file.read()
        
        # --- DATA LOADING ---
        try:
            df = pd.read_excel(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Could not read Excel file.")

        df.columns = df.columns.astype(str).str.strip()
        
        # UPDATED: Changed 'Sales_quantity' to 'Actual'
        required_columns = ['Period', 'Actual']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
             raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

        try:
            df['Period'] = pd.to_datetime(df['Period'])
            df = df.set_index('Period').sort_index()
        except:
             raise HTTPException(status_code=400, detail="The 'Period' column must contain valid dates.")
        
        df = df.fillna(0)
        if df.index.freq is None:
            df.index.freq = pd.infer_freq(df.index)
        if df.index.freq is None:
             df = df.asfreq('MS').fillna(0)

        # UPDATED: Mapping to 'Actual' column
        training_series = df['Actual']
        forecast_steps = 6
        models_run = {} 

        # --- MODEL DEFINITIONS ---
        def run_ses():
            model = ExponentialSmoothing(training_series, trend=None, seasonal=None)
            fit = model.fit(optimized=True)
            mse, _, _ = metrics(training_series - fit.fittedvalues)
            return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}

        def run_des():
            model = ExponentialSmoothing(training_series, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            mse, _, _ = metrics(training_series - fit.fittedvalues)
            return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}

        def run_tes():
            # FIXED: Raise ValueError instead of returning None so the exception is caught and reported
            if len(training_series) < 24: 
                raise ValueError("Insufficient data for TES (needs at least 24 periods/months)")
            
            model = ExponentialSmoothing(training_series, seasonal_periods=12, trend='add', seasonal='add', freq='MS')
            fit = model.fit(optimized=True)
            mse, _, _ = metrics(training_series - fit.fittedvalues)
            return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}

        def run_prophet():
            df_prophet = pd.DataFrame({'ds': training_series.index, 'y': training_series.values})
            m = Prophet()
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=forecast_steps, freq='MS')
            forecast = m.predict(future)
            fitted_vals = forecast.set_index('ds').loc[training_series.index]['yhat']
            mse, _, _ = metrics(training_series - fitted_vals)
            return {'mse': mse, 'fit': fitted_vals, 'forecast': pd.Series(forecast['yhat'].iloc[len(training_series):].values, index=future['ds'].iloc[len(training_series):])}

        # --- EXECUTION ENGINE ---
        available_models = {
            "ses": run_ses,
            "des": run_des,
            "tes": run_tes,
            "prophet": run_prophet
        }

        if model_type != "auto" and model_type in available_models:
            try:
                res = available_models[model_type]()
                if res: models_run[model_type.upper()] = res
            except Exception as e:
                logger.error(f"Model {model_type} failed: {e}")
                # FIXED: Immediately raise error for manual selection so Frontend sees the specific reason
                raise HTTPException(status_code=400, detail=f"Model {model_type} failed: {str(e)}")
        else:
            for name, func in available_models.items():
                try:
                    res = func()
                    if res: models_run[name.upper()] = res
                except Exception as e:
                    logger.warning(f"Auto-run model {name} failed: {e}")

        if not models_run:
             raise HTTPException(status_code=500, detail="All forecasting models failed.")

        best_model_name = min(models_run, key=lambda k: models_run[k]['mse'])
        best_result = models_run[best_model_name]
        
        # --- RESPONSE ---
        history_data = []
        best_fit_values = best_result['fit']
        for date_idx, actual_val in training_series.items():
            fitted_val = None
            if date_idx in best_fit_values.index:
                val = best_fit_values.loc[date_idx]
                if not np.isnan(val): fitted_val = int(round(val))
            history_data.append({
                "name": date_idx.strftime('%b %Y'), 
                "actual": int(round(actual_val)), 
                "fitted": fitted_val
            })

        # STRICT SEPARATION: No bridge code here. 
        # History ends at T. Forecast starts at T+1.

        forecast_data = []
        best_forecast_values = best_result['forecast']
        for date_idx, val in best_forecast_values.items():
            clean_val = 0 if np.isnan(val) else int(round(val))
            forecast_data.append({
                "name": date_idx.strftime('%b %Y') + " (fc)", 
                "forecast": clean_val
            })

        return {
            "history": history_data,
            "forecast": forecast_data,
            "model_name": best_model_name,
            "mode": model_type,
            "message": "Optimization Complete"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Global Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))