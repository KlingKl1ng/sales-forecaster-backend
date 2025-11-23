from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- METRICS FUNCTION ---
def metrics(residuals):
    residuals = residuals.dropna()
    if len(residuals) == 0:
        return float('inf'), float('inf'), float('inf')
    mse = np.mean((residuals)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    return mse, rmse, mae

@app.post("/predict")
async def predict_sales(file: UploadFile = File(...)):
    try:
        # 1. READ CONTENT
        contents = await file.read()
        
        # --- SAFE LOAD START (The Fix) ---
        # We read raw first to avoid crashing on missing index columns
        try:
            df = pd.read_excel(io.BytesIO(contents))
        except:
             raise HTTPException(status_code=400, detail="The uploaded file does not satisfy the requirements above")

        # Clean headers (remove accidental spaces)
        df.columns = df.columns.astype(str).str.strip()

        # Check Required Columns BEFORE processing
        required_columns = ['Period', 'Sales_quantity']
        if not all(col in df.columns for col in required_columns):
             # This ensures we send YOUR custom message, not a Server Error
             raise HTTPException(status_code=400, detail="The uploaded file does not satisfy the requirements above")

        # Now safe to parse dates
        try:
            df['Period'] = pd.to_datetime(df['Period'])
            df = df.set_index('Period')
            df = df.sort_index()
        except:
             raise HTTPException(status_code=400, detail="The uploaded file does not satisfy the requirements above")
        # --- SAFE LOAD END ---
        
        # 2. CLEANING
        df = df.fillna(0)
        
        # Force Frequency
        if df.index.freq is None:
            try:
                df.index.freq = pd.infer_freq(df.index)
            except:
                pass
        
        if df.index.freq is None:
             df = df.asfreq('MS')
             df = df.fillna(0)

        # --- FORECASTING LOGIC ---
        time_series = df['Sales_quantity']
        training_series = time_series 
        forecast_steps = 6

        # A. SES
        model_ses = ExponentialSmoothing(training_series, trend=None, seasonal=None)
        fit_ses = model_ses.fit(optimized=True)
        residuals_ses = training_series - fit_ses.fittedvalues
        mse_ses, _, _ = metrics(residuals_ses)

        # B. DES
        model_des = ExponentialSmoothing(training_series, trend="add", seasonal=None)
        fit_des = model_des.fit(optimized=True)
        residuals_des = training_series - fit_des.fittedvalues
        mse_des, _, _ = metrics(residuals_des)

        # C. TES
        mse_tes = float('inf')
        fit_tes = None
        try:
            model_tes = ExponentialSmoothing(
                training_series,
                seasonal_periods=12,
                trend='add',
                seasonal='add',
                freq='MS'
            )
            fit_tes = model_tes.fit(optimized=True)
            residuals_tes = training_series - fit_tes.fittedvalues
            mse_tes, _, _ = metrics(residuals_tes)
        except Exception as e:
            print(f"TES Failed: {e}")

        # --- MODEL SELECTION ---
        best_model_name = ""
        best_fit_values = None
        best_forecast_values = None

        min_mse = min(mse_ses, mse_des, mse_tes)

        if min_mse == mse_ses:
            best_model_name = "Single Exponential Smoothing (SES)"
            best_fit_values = fit_ses.fittedvalues
            best_forecast_values = fit_ses.forecast(forecast_steps)
        elif min_mse == mse_des:
            best_model_name = "Double Exponential Smoothing (DES)"
            best_fit_values = fit_des.fittedvalues
            best_forecast_values = fit_des.forecast(forecast_steps)
        else:
            best_model_name = "Triple Exponential Smoothing (TES)"
            best_fit_values = fit_tes.fittedvalues
            best_forecast_values = fit_tes.forecast(forecast_steps)

        print(f"Selected Model: {best_model_name}")

        # --- PREPARE RESPONSE ---
        history_data = []
        for date_idx, actual_val in training_series.items():
            fitted_val = None
            if date_idx in best_fit_values.index:
                val = best_fit_values.loc[date_idx]
                if not np.isnan(val) and not np.isinf(val):
                    fitted_val = int(round(val))
            history_data.append({"name": date_idx.strftime('%b %Y'), "actual": int(round(actual_val)), "fitted": fitted_val})

        # Bridge Gap
        if len(history_data) > 0:
            last_point = history_data[-1]
            if last_point["fitted"] is not None:
                last_point["forecast"] = last_point["fitted"]

        forecast_data = []
        for date_idx, val in best_forecast_values.items():
            clean_val = 0
            if not np.isnan(val) and not np.isinf(val):
                clean_val = int(round(val))
            forecast_data.append({"name": date_idx.strftime('%b %Y') + " (fc)", "forecast": clean_val})

        return {
            "history": history_data,
            "forecast": forecast_data,
            "model_name": best_model_name,
            "message": "Success"
        }

    except HTTPException as he:
        # Pass through our custom errors (like the 400 we raised above)
        raise he
    except Exception as e:
        # Catch unexpected server crashes
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")