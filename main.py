from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import logging
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
    validation_method: str = Form("simple"), # 'simple' or 'walk_forward'
    train_horizon: int = Form(...), # Acts as Min Train Size in Walk-Forward
    test_horizon: int = Form(...)   # Fixed Forecast Horizon
):
    try:
        logger.info(f"--- VALIDATION JOB: {file.filename} Method: {validation_method} MinTrain: {train_horizon} Test: {test_horizon} ---")
        
        if test_horizon < 1:
            raise HTTPException(status_code=400, detail="Test horizon must be at least 1 period.")

        contents = await file.read()
        full_series = load_and_prep_data(contents)
        total_len = len(full_series)
        
        # --- DEFINING MODEL LOGIC (Reusable) ---
        def fit_and_score(model_name, train_data, test_data, horizon):
            try:
                mse = float('inf')
                
                if model_name == 'ses':
                    model = ExponentialSmoothing(train_data, trend=None, seasonal=None)
                    fit = model.fit(optimized=True)
                    fc = fit.forecast(horizon)
                    mse, _, _ = metrics(test_data - fc)
                    
                elif model_name == 'des':
                    model = ExponentialSmoothing(train_data, trend="add", seasonal=None)
                    fit = model.fit(optimized=True)
                    fc = fit.forecast(horizon)
                    mse, _, _ = metrics(test_data - fc)
                    
                elif model_name == 'tes':
                    if len(train_data) >= 24:
                        model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add', freq='MS')
                        fit = model.fit(optimized=True)
                        fc = fit.forecast(horizon)
                        mse, _, _ = metrics(test_data - fc)
                    else:
                        return None # Not eligible
                        
                elif model_name == 'prophet':
                    df_p = pd.DataFrame({'ds': train_data.index, 'y': train_data.values})
                    m = Prophet()
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=horizon, freq='MS')
                    fc_df = m.predict(future)
                    fc_vals = fc_df.iloc[-horizon:]['yhat'].values
                    fc_series = pd.Series(fc_vals, index=test_data.index)
                    mse, _, _ = metrics(test_data - fc_series)
                
                return mse
            except Exception as e:
                # logger.warning(f"{model_name} failed on split: {e}")
                return None

        # --- EXECUTION ---
        models_to_test = ["ses", "des", "tes", "prophet"] if model_type == "auto" else [model_type.lower()]
        models_results = {m: [] for m in models_to_test} # Store list of MSEs

        # 1. SIMPLE SPLIT (Original Logic)
        if validation_method == "simple":
            total_required = train_horizon + test_horizon
            if total_len < total_required:
                raise HTTPException(status_code=400, detail=f"Data length ({total_len}) too short for requested split.")
            
            # Fixed Split: Last (Train + Test) points
            validation_slice = full_series.iloc[-total_required:]
            train_s = validation_slice.iloc[:train_horizon]
            test_s = validation_slice.iloc[train_horizon:]
            
            for m in models_to_test:
                mse = fit_and_score(m, train_s, test_s, test_horizon)
                if mse is not None: models_results[m].append(mse)

        # 2. WALK-FORWARD VALIDATION (Auto / All Possible Splits)
        elif validation_method == "walk_forward":
            # Start from min_train, expand until we can't fit a test horizon
            # e.g. Data=50, MinTrain=24, Test=6.
            # Split 1: Train[0:24], Test[24:30]
            # Split 2: Train[0:25], Test[25:31]
            # ...
            # Split N: Train[0:44], Test[44:50]
            
            start_index = train_horizon
            end_index = total_len - test_horizon
            
            if start_index > end_index:
                 raise HTTPException(status_code=400, detail="Constraints impossible: Min Train + Test Horizon > Total Data")

            # Optimization: If dataset is huge, Step > 1 to prevent timeout
            step = 1
            possible_folds = end_index - start_index + 1
            if possible_folds > 10 and "prophet" in models_to_test:
                step = max(1, possible_folds // 5) # Reduce folds for Prophet speed
            
            logger.info(f"Running WFV. Folds: {possible_folds}, Step: {step}")

            for i in range(start_index, end_index + 1, step):
                train_s = full_series.iloc[:i]
                test_s = full_series.iloc[i : i + test_horizon]
                
                for m in models_to_test:
                    mse = fit_and_score(m, train_s, test_s, test_horizon)
                    if mse is not None: models_results[m].append(mse)

        # --- AGGREGATE RESULTS ---
        final_scores = {}
        for m, errors in models_results.items():
            if errors:
                final_scores[m] = np.mean(errors) # Average MSE
        
        if not final_scores:
            raise HTTPException(status_code=500, detail="All validation models failed or data insufficient.")

        best_model_name = min(final_scores, key=final_scores.get)
        best_mse = final_scores[best_model_name]

        return {
            "winner": best_model_name,
            "best_mse": best_mse if best_mse != float('inf') else 0,
            "details": final_scores,
            "message": f"Winner: {best_model_name.upper()} (Avg MSE: {best_mse:.2f})"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Validation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- PREDICTION ENDPOINT (Unchanged) ---
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