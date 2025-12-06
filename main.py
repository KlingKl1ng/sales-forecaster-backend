from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Security, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import logging
import xlsxwriter 
import os
import math 
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings
import time

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
    expose_headers=["Content-Disposition"]
)

# ---------------------------------------------------------
# NEW: SECURITY CONFIGURATION
# ---------------------------------------------------------
API_KEY_NAME = "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# ADMIN CONFIG: Add your clients here
AUTHORIZED_USERS = {
    "Admin (You)": "sk_admin_master_key_888",
    "Friend 1": "sk_friend_001_abc", 
    "Friend 2": "sk_friend_002_xyz",
    "Client A": "sk_client_a_prod_777"
}

# Rate Limiting
RATE_LIMIT_WINDOW = 60 
RATE_LIMIT_MAX_REQUESTS = 20
request_history = {} 

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    if api_key_header in AUTHORIZED_USERS.values():
        return api_key_header
    raise HTTPException(status_code=403, detail="Access Denied: Invalid or inactive API Key.")

async def check_rate_limit(api_key: str = Security(get_api_key)):
    current_time = time.time()
    if api_key not in request_history:
        request_history[api_key] = []
    request_history[api_key] = [t for t in request_history[api_key] if current_time - t < RATE_LIMIT_WINDOW]
    if len(request_history[api_key]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    request_history[api_key].append(current_time)
    return api_key

# NEW: Pydantic Models for API
class DataPoint(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    value: float = Field(..., description="The observed value")

class ForecastRequest(BaseModel):
    data: List[DataPoint]
    model_type: str = Field("ses")
    forecast_steps: int = Field(6, ge=1, le=60)

class ForecastResponse(BaseModel):
    model_name: str
    forecast_mse: Optional[float]
    history: List[Dict[str, Any]]
    forecast: List[Dict[str, Any]]
    message: str

# Helper to fix "Out of range float values" error in JSON
def clean_float(val):
    if val is None: return None
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val): return None
        return f_val
    except: return None

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

# --- MODEL NAMING MAPPING ---
MODEL_DISPLAY_NAMES = {
    "ses": "SES (Level)",
    "des": "DES (Trend)",
    "tes": "TES (Seasonal)",
    "prophet": "Prophet (Complex)"
}

# --- HELPER FOR EXPORT (Original) ---
def _run_model_logic(series, model_code, steps):
    try:
        if model_code == 'ses':
            model = ExponentialSmoothing(series, trend=None, seasonal=None)
            fit = model.fit(optimized=True)
            return fit.forecast(steps)
        elif model_code == 'des':
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            return fit.forecast(steps)
        elif model_code == 'tes':
            if len(series) < 24: return None
            model = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add', freq='MS')
            fit = model.fit(optimized=True)
            return fit.forecast(steps)
        elif model_code == 'prophet':
            df_p = pd.DataFrame({'ds': series.index, 'y': series.values})
            m = Prophet()
            m.fit(df_p)
            future = m.make_future_dataframe(periods=steps, freq='MS')
            fc_df = m.predict(future)
            return pd.Series(fc_df['yhat'].iloc[-steps:].values, index=future['ds'].iloc[-steps:])
    except:
        return None
    return None

# --- VALIDATION ENDPOINT (Original Logic) ---
@app.post("/validate")
async def validate_models(
    file: UploadFile = File(...),
    model_type: str = Form("auto"),
    validation_method: str = Form("simple"), 
    train_horizon: int = Form(...), 
    test_horizon: int = Form(...)   
):
    try:
        logger.info(f"--- VALIDATION JOB: {file.filename} Method: {validation_method} MinTrain: {train_horizon} Test: {test_horizon} ---")
        
        if test_horizon < 1:
            raise HTTPException(status_code=400, detail="Test horizon must be at least 1 period.")

        contents = await file.read()
        full_series = load_and_prep_data(contents)
        total_len = len(full_series)
        
        if total_len < train_horizon + test_horizon:
             raise HTTPException(status_code=400, detail=f"Insufficient data. Length ({total_len}) must be >= Train ({train_horizon}) + Test ({test_horizon})")

        def fit_and_score(model_code, train_data, test_data, horizon):
            try:
                mse = float('inf')
                fc_series = None
                
                if model_code == 'ses':
                    model = ExponentialSmoothing(train_data, trend=None, seasonal=None)
                    fit = model.fit(optimized=True)
                    fc = fit.forecast(horizon)
                    mse, _, _ = metrics(test_data - fc)
                    fc_series = fc
                elif model_code == 'des':
                    model = ExponentialSmoothing(train_data, trend="add", seasonal=None)
                    fit = model.fit(optimized=True)
                    fc = fit.forecast(horizon)
                    mse, _, _ = metrics(test_data - fc)
                    fc_series = fc
                elif model_code == 'tes':
                    if len(train_data) >= 24:
                        model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add', freq='MS')
                        fit = model.fit(optimized=True)
                        fc = fit.forecast(horizon)
                        mse, _, _ = metrics(test_data - fc)
                        fc_series = fc
                    else: return None, None 
                elif model_code == 'prophet':
                    df_p = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
                    m = Prophet()
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=horizon, freq='MS')
                    fc_df = m.predict(future)
                    fc_vals = fc_df.iloc[-horizon:]['yhat'].values
                    fc_series = pd.Series(fc_vals, index=test_data.index)
                    mse, _, _ = metrics(test_data - fc_series)
                return mse, fc_series
            except Exception as e: return None, None

        models_to_test = ["ses", "des", "tes", "prophet"] if model_type == "auto" else [model_type]
        models_mse_list = {m: [] for m in models_to_test} 
        models_stitched_forecasts = {m: pd.Series(dtype='float64') for m in models_to_test}

        if validation_method == "simple":
            validation_slice = full_series.iloc[:train_horizon + test_horizon]
            train_s = validation_slice.iloc[:train_horizon]
            test_s = validation_slice.iloc[train_horizon:]
            for m in models_to_test:
                mse, fc = fit_and_score(m, train_s, test_s, test_horizon)
                if mse is not None: 
                    models_mse_list[m].append(mse)
                    models_stitched_forecasts[m] = fc

        elif validation_method == "walk_forward":
            start_index = train_horizon
            end_index = total_len - test_horizon
            step = test_horizon 
            logger.info(f"Running WFV. Start: {start_index}, End: {end_index}, Step: {step}")
            for i in range(start_index, end_index + 1, step):
                train_s = full_series.iloc[:i]
                test_s = full_series.iloc[i : i + test_horizon]
                for m in models_to_test:
                    mse, fc = fit_and_score(m, train_s, test_s, test_horizon)
                    if mse is not None: 
                        models_mse_list[m].append(mse)
                        if test_horizon == 1:
                             if models_stitched_forecasts[m].empty:
                                 models_stitched_forecasts[m] = fc
                             else:
                                 models_stitched_forecasts[m] = pd.concat([models_stitched_forecasts[m], fc])
                        else:
                             if models_stitched_forecasts[m].empty:
                                 models_stitched_forecasts[m] = fc

        final_scores = {}
        for m, errors in models_mse_list.items():
            if errors: final_scores[m] = np.mean(errors) 
        
        if not final_scores:
            raise HTTPException(status_code=500, detail="All validation models failed or data insufficient.")

        best_model_code = min(final_scores, key=final_scores.get)
        best_mse = final_scores[best_model_code]
        best_model_display_name = MODEL_DISPLAY_NAMES.get(best_model_code, best_model_code)

        winner_forecast_series = models_stitched_forecasts[best_model_code]
        if winner_forecast_series.empty: raise HTTPException(status_code=500, detail="Winner produced no forecast data.")
             
        forecast_start_date = winner_forecast_series.index[0]
        viz_history_series = full_series[full_series.index < forecast_start_date]
        
        viz_fitted_values = None
        try:
            if best_model_code == 'ses':
                m = ExponentialSmoothing(viz_history_series, trend=None, seasonal=None).fit(optimized=True)
                viz_fitted_values = m.fittedvalues
            elif best_model_code == 'des':
                m = ExponentialSmoothing(viz_history_series, trend="add", seasonal=None).fit(optimized=True)
                viz_fitted_values = m.fittedvalues
            elif best_model_code == 'tes':
                if len(viz_history_series) >= 24:
                     m = ExponentialSmoothing(viz_history_series, seasonal_periods=12, trend='add', seasonal='add', freq='MS').fit(optimized=True)
                     viz_fitted_values = m.fittedvalues
            elif best_model_code == 'prophet':
                df_p = pd.DataFrame({'ds': viz_history_series.index, 'y': viz_history_series.values})
                m = Prophet()
                m.fit(df_p)
                fc = m.predict(df_p)
                viz_fitted_values = pd.Series(fc['yhat'].values, index=viz_history_series.index)
        except Exception as e:
            logger.warning(f"Failed to generate fitted history: {e}")

        viz_validation_actuals = full_series[full_series.index.isin(winner_forecast_series.index)]

        history_data = []
        for date_idx, actual_val in viz_history_series.items():
            fitted_val = None
            if viz_fitted_values is not None and date_idx in viz_fitted_values.index:
                val = viz_fitted_values.loc[date_idx]
                if not np.isnan(val): fitted_val = int(round(val))
            history_data.append({
                "name": date_idx.strftime('%b %Y'), 
                "actual_train": int(round(actual_val)), 
                "fitted": fitted_val 
            })

        forecast_data = []
        for date_idx, val in winner_forecast_series.items():
            clean_val = 0 if np.isnan(val) else int(round(val))
            actual_val = None
            if date_idx in viz_validation_actuals.index:
                actual_val = int(round(viz_validation_actuals.loc[date_idx]))
            forecast_data.append({
                "name": date_idx.strftime('%b %Y') + " (val)", 
                "forecast": clean_val,
                "actual_test": actual_val
            })

        return {
            "winner_code": best_model_code, 
            "winner": best_model_display_name, 
            "best_mse": best_mse if best_mse != float('inf') else 0,
            "details": final_scores,
            "message": f"Winner: {best_model_display_name} (Avg MSE: {best_mse:.2f})",
            "model_name": best_model_display_name,
            "history": history_data,
            "forecast": forecast_data
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Validation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- PREDICTION ENDPOINT (Original Logic) ---
@app.post("/predict")
async def predict_sales(
    file: UploadFile = File(...),
    model_type: str = Form(...), 
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
            if len(training_series) < 24: raise ValueError("Insufficient data for TES")
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

        available_models = {"ses": run_ses, "des": run_des, "tes": run_tes, "prophet": run_prophet}

        if model_type in available_models:
            try:
                res = available_models[model_type]()
                if res: models_run[model_type] = res
            except Exception as e: raise HTTPException(status_code=400, detail=f"Model {model_type} failed: {str(e)}")
        else: raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

        if not models_run: raise HTTPException(status_code=500, detail="Forecasting failed.")

        best_model_code = list(models_run.keys())[0] 
        best_result = models_run[best_model_code]
        best_model_display_name = MODEL_DISPLAY_NAMES.get(best_model_code, best_model_code)
        
        history_data = []
        best_fit_values = best_result['fit']
        for date_idx, actual_val in training_series.items():
            fitted_val = None
            if date_idx in best_fit_values.index:
                val = best_fit_values.loc[date_idx]
                if not np.isnan(val): fitted_val = int(round(val))
            history_data.append({
                "name": date_idx.strftime('%b %Y'), 
                "actual_train": int(round(actual_val)),
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
            "model_name": best_model_display_name,
            "forecast_mse": best_result['mse'],
            "mode": model_type,
            "message": "Optimization Complete"
        }
    except HTTPException as he: raise he
    except Exception as e:
        logger.error(f"Global Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: API ENDPOINT (Using copied logic to preserve original structure) ---
@app.post("/api/v1/forecast", response_model=ForecastResponse)
async def api_forecast(
    request: ForecastRequest, 
    api_key: str = Security(check_rate_limit)
):
    try:
        # 1. Convert JSON to Pandas Series
        data_dicts = [item.dict() for item in request.data]
        df = pd.DataFrame(data_dicts)
        
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df['value'] = pd.to_numeric(df['value'])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid data format.")
            
        if df.empty:
            raise HTTPException(status_code=400, detail="Data payload is empty.")
            
        # Ensure Frequency
        if df.index.freq is None:
            df.index.freq = pd.infer_freq(df.index)
        if df.index.freq is None:
            df = df.asfreq('MS').fillna(0)
        
        training_series = df['value']
        model_type = request.model_type
        forecast_steps = request.forecast_steps
        
        # 2. RUN MODEL LOGIC (Copied from /predict to ensure isolation)
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
            if len(training_series) < 24: raise ValueError("Insufficient data for TES")
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

        available_models = {"ses": run_ses, "des": run_des, "tes": run_tes, "prophet": run_prophet}

        if model_type in available_models:
            try:
                res = available_models[model_type]()
                if res: models_run[model_type] = res
            except Exception as e: raise HTTPException(status_code=400, detail=f"Model {model_type} failed: {str(e)}")
        else: raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

        if not models_run: raise HTTPException(status_code=500, detail="Forecasting failed.")

        best_model_code = list(models_run.keys())[0] 
        best_result = models_run[best_model_code]
        best_model_display_name = MODEL_DISPLAY_NAMES.get(best_model_code, best_model_code)
        
        # 3. BUILD RESPONSE (With clean_float for safety)
        history_data = []
        best_fit_values = best_result['fit']
        for date_idx, actual_val in training_series.items():
            fitted_val = None
            if date_idx in best_fit_values.index:
                val = best_fit_values.loc[date_idx]
                if not np.isnan(val): fitted_val = clean_float(val)
            history_data.append({
                "name": date_idx.strftime('%b %Y'), 
                "actual_train": clean_float(actual_val),
                "fitted": fitted_val
            })

        forecast_data = []
        best_forecast_values = best_result['forecast']
        for date_idx, val in best_forecast_values.items():
            clean_val = clean_float(val) if not np.isnan(val) else 0
            forecast_data.append({
                "name": date_idx.strftime('%b %Y') + " (fc)", 
                "forecast": clean_val
            })

        return {
            "history": history_data,
            "forecast": forecast_data,
            "model_name": best_model_display_name,
            "forecast_mse": clean_float(best_result['mse']),
            "message": "Optimization Complete"
        }

    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- EXPORT ENDPOINT (Original) ---
@app.post("/export")
async def export_report(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    validation_method: str = Form("simple"),
    train_horizon: int = Form(...),
    test_horizon: int = Form(...),
    forecast_steps: int = Form(...)
):
    try:
        logger.info(f"Generating Report for {model_type}")
        contents = await file.read()
        full_series = load_and_prep_data(contents)
        clean_filename = os.path.splitext(file.filename)[0]
        
        # --- 1. RUN VALIDATION (Conditional Logic) ---
        val_forecast = None
        if len(full_series) <= test_horizon:
             raise HTTPException(status_code=400, detail="Not enough data for validation export")
        
        if validation_method == "simple":
             # Logic: First Slice (Train 1..N -> Forecast N+1)
             validation_scope_df = full_series.iloc[:train_horizon + test_horizon]
             train_subset = validation_scope_df.iloc[:train_horizon]
             val_forecast = _run_model_logic(train_subset, model_type, test_horizon)
        
        elif validation_method == "walk_forward":
             # Logic matches the chart: 
             # If test_horizon == 1: Stitch all folds
             # If test_horizon > 1: Only the first fold
             
             if test_horizon > 1:
                 # Behave like simple split (First Fold)
                 validation_scope_df = full_series.iloc[:train_horizon + test_horizon]
                 train_subset = validation_scope_df.iloc[:train_horizon]
                 val_forecast = _run_model_logic(train_subset, model_type, test_horizon)
             else:
                 # Stitch everything
                 val_forecast_series = pd.Series(dtype='float64')
                 start_index = train_horizon
                 end_index = len(full_series) - test_horizon
                 step = test_horizon 
                 
                 for i in range(start_index, end_index + 1, step):
                    train_s = full_series.iloc[:i]
                    fc = _run_model_logic(train_s, model_type, test_horizon)
                    val_forecast_series = pd.concat([val_forecast_series, fc])
                 val_forecast = val_forecast_series

        if val_forecast is None: raise HTTPException(status_code=500, detail="Validation model failed")

        # --- 2. RUN FUTURE FORECAST ---
        future_forecast = _run_model_logic(full_series, model_type, forecast_steps)
        if future_forecast is None: raise HTTPException(status_code=500, detail="Forecast model failed")

        # --- 3. BUILD MASTER DATAFRAME ---
        all_dates = sorted(list(set(list(full_series.index) + list(future_forecast.index))))
        master_df = pd.DataFrame(index=all_dates)
        master_df.index.name = "Period"
        master_df['Actual'] = full_series
        master_df['Validation (Backtest)'] = val_forecast
        master_df['Forecast (Future)'] = future_forecast
        
        master_df['Accuracy Delta %'] = master_df.apply(
            lambda row: (row['Validation (Backtest)'] - row['Actual']) / row['Actual'] 
            if (pd.notna(row['Actual']) and row['Actual'] != 0 and pd.notna(row['Validation (Backtest)'])) else None, 
            axis=1
        )
        
        # --- 4. EXCEL GENERATION ---
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        worksheet = workbook.add_worksheet("Operartis Report")
        
        # Formats
        fmt_header_main = workbook.add_format({'bold': True, 'font_size': 20, 'font_color': '#b45309', 'align': 'left', 'valign': 'vcenter'})
        fmt_header_sub = workbook.add_format({'italic': True, 'font_color': '#b45309', 'align': 'left', 'valign': 'top'})
        fmt_meta_label = workbook.add_format({'bold': True, 'font_size': 9, 'font_color': '#475569', 'bg_color': '#f1f5f9', 'border': 1, 'align': 'left'})
        fmt_meta_value = workbook.add_format({'font_size': 9, 'font_color': '#0f172a', 'bg_color': '#ffffff', 'border': 1, 'align': 'left'})
        fmt_table_header = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#1e293b', 'border': 1, 'align': 'center'})
        fmt_date = workbook.add_format({'num_format': 'mmm yyyy', 'border': 1, 'align': 'left'})
        fmt_num = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'right'})
        fmt_pct = workbook.add_format({'num_format': '0.0%', 'border': 1, 'align': 'center'})
        fmt_val_col = workbook.add_format({'num_format': '#,##0', 'font_color': '#64748b', 'italic': True, 'border': 1, 'align': 'right'})
        fmt_fc_col = workbook.add_format({'num_format': '#,##0', 'bold': True, 'font_color': '#b45309', 'bg_color': '#fffbeb', 'border': 1, 'align': 'right'}) 
        fmt_footer = workbook.add_format({'font_color': '#94a3b8', 'italic': True, 'font_size': 8, 'align': 'left'})

        # Layout matching screenshot
        worksheet.set_column('A:A', 2)   
        worksheet.set_column('B:B', 22)  
        worksheet.set_column('C:C', 20)  
        worksheet.set_column('D:D', 25)  
        worksheet.set_column('E:E', 20)  
        worksheet.set_column('F:F', 25)  

        logo_path = "icononly_transparent_nobuffer.png" 
        if os.path.exists(logo_path):
            # Anchor at B2, scale 0.05 (5% - adjusted based on previous feedback)
            worksheet.insert_image('B2', logo_path, {'x_scale': 0.05, 'y_scale': 0.05, 'x_offset': 50})
            
        worksheet.set_row(1, 45) 
        worksheet.set_row(2, 25) 

        worksheet.merge_range('C2:F2', "OPERARTIS FORECAST REPORT", fmt_header_main)
        worksheet.merge_range('C3:F3', "Optimizing Today, Growing Tomorrow.", fmt_header_sub)
        
        # Metadata Block
        worksheet.write('B5', "Report By:", fmt_meta_label)
        worksheet.write('C5', "Operartis Analytics", fmt_meta_value)
        
        worksheet.write('B6', "Report Date:", fmt_meta_label)
        worksheet.write('C6', datetime.now().strftime("%Y-%m-%d %H:%M"), fmt_meta_value)
        
        worksheet.write('B7', "Report For:", fmt_meta_label)
        worksheet.write('C7', clean_filename, fmt_meta_value)

        worksheet.write('B8', "Model Used:", fmt_meta_label)
        worksheet.write('C8', MODEL_DISPLAY_NAMES.get(model_type, model_type), fmt_meta_value)
        
        # LEGEND (Added to E5-E7)
        worksheet.write('E5', "Legend:", fmt_meta_label)
        worksheet.write('E6', "Green: Over-forecast (Positive)", fmt_meta_value)
        worksheet.write('E7', "Red: Under-forecast (Negative)", fmt_meta_value)

        # Dynamic Column Header
        headers = ["Period", "Actual History", f"Validation (Test {test_horizon} horizon)", "Accuracy Delta", "Future Forecast"]
        for col, h in enumerate(headers):
            worksheet.write(9, col+1, h, fmt_table_header) 
            
        row = 10
        for date_idx, data_row in master_df.iterrows():
            worksheet.write_datetime(row, 1, date_idx, fmt_date)
            if not pd.isna(data_row['Actual']): worksheet.write_number(row, 2, data_row['Actual'], fmt_num)
            else: worksheet.write(row, 2, "-", fmt_num)
            if not pd.isna(data_row['Validation (Backtest)']): worksheet.write_number(row, 3, data_row['Validation (Backtest)'], fmt_val_col)
            else: worksheet.write(row, 3, "", fmt_val_col)
            if not pd.isna(data_row['Accuracy Delta %']): worksheet.write_number(row, 4, data_row['Accuracy Delta %'], fmt_pct)
            else: worksheet.write(row, 4, "", fmt_pct)
            if not pd.isna(data_row['Forecast (Future)']): worksheet.write_number(row, 5, data_row['Forecast (Future)'], fmt_fc_col)
            else: worksheet.write(row, 5, "", fmt_fc_col)
            row += 1
            
        worksheet.conditional_format(10, 4, row-1, 4, {'type': '3_color_scale', 'min_color': "#f87171", 'mid_color': "#ffffff", 'max_color': "#4ade80"})
        
        # COPYRIGHT FOOTER
        footer_row = row + 2
        worksheet.write(footer_row, 1, f"© {datetime.now().year} Operartis Analytics. All rights reserved.", fmt_footer)

        #Worksheet Protection Layer
        worksheet.protect('Operartis020561', {'select_locked_cells': True, 'select_unlocked_cells': True})
        workbook.close()
        output.seek(0)
        
        # Updated filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Operartis_Forecast_{clean_filename}_{timestamp_str}.xlsx"
        
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))