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

# --- COMMON DATA PREP ---
def load_and_prep_data(contents):
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read Excel file.")

    df.columns = df.columns.astype(str).str.strip()
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
            
    return df['Actual']

# --- VALIDATION ENDPOINT ---
@app.post("/validate")
async def validate_models(
    file: UploadFile = File(...),
    model_type: str = Form("auto"),
    train_horizon: int = Form(...),
    test_horizon: int = Form(...)
):
    try:
        logger.info(f"--- VALIDATION JOB: {file.filename} Train: {train_horizon} Test: {test_horizon} ---")
        contents = await file.read()
        full_series = load_and_prep_data(contents)
        
        total_required = train_horizon + test_horizon
        if len(full_series) < total_required:
            raise HTTPException(status_code=400, detail=f"Data length ({len(full_series)}) is smaller than required Train+Test ({total_required})")

        # Split data: Take the last (train + test) points
        # Train on the first part of that slice, Test on the last part
        validation_slice = full_series.iloc[-total_required:]
        train_series = validation_slice.iloc[:train_horizon]
        test_series = validation_slice.iloc[train_horizon:]

        models_run = {}
        
        # Model Runners for Validation (similar to predict but returns MSE on test set)
        def val_ses():
            model = ExponentialSmoothing(train_series, trend=None, seasonal=None)
            fit = model.fit(optimized=True)
            forecast = fit.forecast(test_horizon)
            mse, _, _ = metrics(test_series - forecast)
            return {'mse': mse, 'name': 'ses'}

        def val_des():
            model = ExponentialSmoothing(train_series, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            forecast = fit.forecast(test_horizon)
            mse, _, _ = metrics(test_series - forecast)
            return {'mse': mse, 'name': 'des'}

        def val_tes():
            if len(train_series) < 24: return None # TES requirement
            model = ExponentialSmoothing(train_series, seasonal_periods=12, trend='add', seasonal='add', freq='MS')
            fit = model.fit(optimized=True)
            forecast = fit.forecast(test_horizon)
            mse, _, _ = metrics(test_series - forecast)
            return {'mse': mse, 'name': 'tes'}

        def val_prophet():
            df_prophet = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
            m = Prophet()
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=test_horizon, freq='MS')
            forecast_df = m.predict(future)
            # Extract only the forecast part corresponding to test_series
            forecast_vals = forecast_df.iloc[-test_horizon:]['yhat'].values
            # Align with test series for subtraction
            forecast_series = pd.Series(forecast_vals, index=test_series.index)
            mse, _, _ = metrics(test_series - forecast_series)
            return {'mse': mse, 'name': 'prophet'}

        available_models = {
            "ses": val_ses,
            "des": val_des,
            "tes": val_tes,
            "prophet": val_prophet
        }

        # Run Logic
        if model_type != "auto" and model_type in available_models:
            try:
                res = available_models[model_type]()
                if res: models_run[model_type] = res
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Validation for {model_type} failed: {e}")
        else:
            for name, func in available_models.items():
                try:
                    res = func()
                    if res: models_run[name] = res
                except Exception:
                    pass # Skip failing models in auto mode
        
        if not models_run:
            raise HTTPException(status_code=500, detail="All validation models failed.")

        best_model_name = min(models_run, key=lambda k: models_run[k]['mse'])
        best_mse = models_run[best_model_name]['mse']

        return {
            "winner": best_model_name,
            "best_mse": best_mse if best_mse != float('inf') else 0,
            "details": models_run,
            "message": f"Best model is {best_model_name.upper()}"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Validation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- PREDICTION ENDPOINT (Existing logic preserved) ---
@app.post("/predict")
async def predict_sales(
    file: UploadFile = File(...),
    model_type: str = Form("auto"),
    forecast_steps: int = Form(6)
):
    try:
        logger.info(f"--- NEW JOB: {file.filename} [{model_type}] Steps: {forecast_steps} ---")
        contents = await file.read()
        training_series = load_and_prep_data(contents)
        
        models_run = {} 

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