from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
# from pmdarima import auto_arima  <-- KEPT COMMENTED OUT
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_forecaster")

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HEALTH CHECK ---
@app.get("/")
def health_check():
    return {"status": "online", "message": "Sales Forecaster (Fixed Syntax)"}

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
        logger.info(f"--- NEW REQUEST: {file.filename} [{model_type}] ---")
        contents = await file.read()
        
        # --- SAFE LOAD ---
        try:
            df = pd.read_excel(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"File Load Error: {e}")
            raise HTTPException(status_code=400, detail="Could not read Excel file.")

        df.columns = df.columns.astype(str).str.strip()
        required_columns = ['Period', 'Sales_quantity']
        if not all(col in df.columns for col in required_columns):
             raise HTTPException(status_code=400, detail=f"Missing columns. Required: {required_columns}")

        try:
            df['Period'] = pd.to_datetime(df['Period'])
            df = df.set_index('Period')
            df = df.sort_index()
        except:
             raise HTTPException(status_code=400, detail="Period column must be dates.")
        
        df = df.fillna(0)
        if df.index.freq is None:
            try:
                df.index.freq = pd.infer_freq(df.index)
            except: pass
        if df.index.freq is None:
             df = df.asfreq('MS')
             df = df.fillna(0)

        training_series = df['Sales_quantity']
        forecast_steps = 6
        models_run = {} 

        # --- MODEL FUNCTIONS ---
        def run_ses():
            try:
                logger.info("Running SES...")
                model = ExponentialSmoothing(training_series, trend=None, seasonal=None)
                fit = model.fit(optimized=True)
                mse, _, _ = metrics(training_series - fit.fittedvalues)
                return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}
            except Exception as e: 
                logger.warning(f"SES Error: {e}")
                return None

        def run_des():
            try:
                logger.info("Running DES...")
                model = ExponentialSmoothing(training_series, trend="add", seasonal=None)
                fit = model.fit(optimized=True)
                mse, _, _ = metrics(training_series - fit.fittedvalues)
                return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}
            except Exception as e: 
                logger.warning(f"DES Error: {e}")
                return None

        def run_tes():
            try:
                logger.info("Running TES...")
                model = ExponentialSmoothing(training_series, seasonal_periods=12, trend='add', seasonal='add', freq='MS')
                fit = model.fit(optimized=True)
                mse, _, _ = metrics(training_series - fit.fittedvalues)
                return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}
            except Exception as e: 
                logger.warning(f"TES Error: {e}")
                return None

        def run_prophet():
            try:
                logger.info("Running Prophet...")
                df_prophet = pd.DataFrame({'ds': training_series.index, 'y': training_series.values})
                m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=forecast_steps, freq='MS')
                forecast = m.predict(future)
                
                fitted_vals = forecast.set_index('ds').loc[training_series.index]['yhat']
                mse, _, _ = metrics(training_series - fitted_vals)
                return {'mse': mse, 'fit': fitted_vals, 'forecast': pd.Series(forecast['yhat'].iloc[len(training_series):].values, index=future['ds'].iloc[len(training_series):])}
            except Exception as e: 
                logger.warning(f"Prophet Error: {e}")
                return None

        # --- SELECTION LOGIC ---
        
        # If specific model selected, run ONLY that one
        if model_type == "ses":
            res = run_ses()
            if res: models_run["Single Exponential Smoothing (SES)"] = res
            
        elif model_type == "des":
            res = run_des()
            if res: models_run["Double Exponential Smoothing (DES)"] = res
            
        elif model_type == "tes":
            res = run_tes()
            if res: models_run["Triple Exponential Smoothing (TES)"] = res
            
        elif model_type == "prophet":
            res = run_prophet()
            if res: models_run["Prophet"] = res

        # NOTE: Triple quotes """ """ break the if/else chain. 
        # Use # to comment out code inside logic blocks.
        # elif model_type == "arima":
        #    res = run_arima()
        #    if res: models_run["ARIMA"] = res
            
        else: # "auto" or unknown -> Run ALL and compare
            res_ses = run_ses()
            if res_ses: models_run["SES"] = res_ses
            
            res_des = run_des()
            if res_des: models_run["DES"] = res_des
            
            res_tes = run_tes()
            if res_tes: models_run["TES"] = res_tes
            
            res_pp = run_prophet()
            if res_pp: models_run["Prophet"] = res_pp
            
            # res_arima = run_arima()
            # if res_arima: models_run["ARIMA"] = res_arima

        # Filter out failed models
        models_run = {k: v for k, v in models_run.items() if v is not None}

        if not models_run:
             raise HTTPException(status_code=500, detail="All models failed. Check data format.")

        best_model_name = min(models_run, key=lambda k: models_run[k]['mse'])
        best_result = models_run[best_model_name]
        
        logger.info(f"Winner: {best_model_name}")

        # --- RESPONSE ---
        history_data = []
        best_fit_values = best_result['fit']
        for date_idx, actual_val in training_series.items():
            fitted_val = None
            if date_idx in best_fit_values.index:
                val = best_fit_values.loc[date_idx]
                if not np.isnan(val) and not np.isinf(val):
                    fitted_val = int(round(val))
            history_data.append({"name": date_idx.strftime('%b %Y'), "actual": int(round(actual_val)), "fitted": fitted_val})

        if len(history_data) > 0:
            last_point = history_data[-1]
            if last_point["fitted"] is not None:
                last_point["forecast"] = last_point["fitted"]

        forecast_data = []
        best_forecast_values = best_result['forecast']
        for date_idx, val in best_forecast_values.items():
            clean_val = 0
            if not np.isnan(val) and not np.isinf(val):
                clean_val = int(round(val))
            forecast_data.append({"name": date_idx.strftime('%b %Y') + " (fc)", "forecast": clean_val})

        return {
            "history": history_data,
            "forecast": forecast_data,
            "model_name": best_model_name,
            "mode": model_type,
            "message": "Success"
        }

    except Exception as e:
        logger.error(f"Global Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))