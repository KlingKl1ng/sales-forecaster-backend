from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from pmdarima import auto_arima

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
# Added model_type as a Form parameter (defaults to 'auto')
async def predict_sales(
    file: UploadFile = File(...), 
    model_type: str = Form("auto") 
):
    try:
        # 1. READ CONTENT
        contents = await file.read()
        
        # --- SAFE LOAD START ---
        try:
            df = pd.read_excel(io.BytesIO(contents))
        except:
             raise HTTPException(status_code=400, detail="The uploaded file does not satisfy the requirements above")

        df.columns = df.columns.astype(str).str.strip()
        required_columns = ['Period', 'Sales_quantity']
        if not all(col in df.columns for col in required_columns):
             raise HTTPException(status_code=400, detail="The uploaded file does not satisfy the requirements above")

        try:
            df['Period'] = pd.to_datetime(df['Period'])
            df = df.set_index('Period')
            df = df.sort_index()
        except:
             raise HTTPException(status_code=400, detail="The uploaded file does not satisfy the requirements above")
        # --- SAFE LOAD END ---
        
        # 2. CLEANING
        df = df.fillna(0)
        if df.index.freq is None:
            try:
                df.index.freq = pd.infer_freq(df.index)
            except:
                pass
        if df.index.freq is None:
             df = df.asfreq('MS')
             df = df.fillna(0)

        # --- FORECASTING SETUP ---
        training_series = df['Sales_quantity']
        forecast_steps = 6
        
        # Dictionary to store results of all models run
        models_run = {} 
        # Structure: { 'MODEL_NAME': {'mse': float, 'fit': series, 'forecast': series} }

        # --- MODEL EXECUTION FUNCTIONS ---
        
        def run_ses():
            try:
                model = ExponentialSmoothing(training_series, trend=None, seasonal=None)
                fit = model.fit(optimized=True)
                mse, _, _ = metrics(training_series - fit.fittedvalues)
                return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}
            except: return None

        def run_des():
            try:
                model = ExponentialSmoothing(training_series, trend="add", seasonal=None)
                fit = model.fit(optimized=True)
                mse, _, _ = metrics(training_series - fit.fittedvalues)
                return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}
            except: return None

        def run_tes():
            try:
                model = ExponentialSmoothing(training_series, seasonal_periods=12, trend='add', seasonal='add', freq='MS')
                fit = model.fit(optimized=True)
                mse, _, _ = metrics(training_series - fit.fittedvalues)
                return {'mse': mse, 'fit': fit.fittedvalues, 'forecast': fit.forecast(forecast_steps)}
            except: return None

        def run_prophet():
            try:
                df_prophet = pd.DataFrame({'ds': training_series.index, 'y': training_series.values})
                m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=forecast_steps, freq='MS')
                forecast = m.predict(future)
                
                # Extract fitted (historical) and forecast (future)
                # Fit: match indices
                fitted_vals = forecast.set_index('ds').loc[training_series.index]['yhat']
                mse, _, _ = metrics(training_series - fitted_vals)
                
                return {
                    'mse': mse, 
                    'fit': fitted_vals,
                    'forecast': pd.Series(forecast['yhat'].iloc[len(training_series):].values, index=future['ds'].iloc[len(training_series):])
                }
            except Exception as e: 
                print(f"Prophet error: {e}")
                return None

        def run_arima():
            try:
                model = auto_arima(training_series, seasonal=True, m=12, suppress_warnings=True, stepwise=True)
                fit = model.fit(training_series)
                fitted_vals = pd.Series(model.predict_in_sample(), index=training_series.index)
                mse, _, _ = metrics(training_series - fitted_vals)
                
                forecast_vals = model.predict(n_periods=forecast_steps)
                last_date = training_series.index[-1]
                future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='MS')[1:]
                
                return {'mse': mse, 'fit': fitted_vals, 'forecast': pd.Series(forecast_vals, index=future_dates)}
            except Exception as e:
                print(f"ARIMA error: {e}")
                return None

        # --- SELECTION LOGIC ---
        
        # If specific model selected, run ONLY that one
        if model_type == "ses":
            res = run_ses()
            if res: models_run["SES"] = res
            
        elif model_type == "des":
            res = run_des()
            if res: models_run["DES"] = res
            
        elif model_type == "tes":
            res = run_tes()
            if res: models_run["TES"] = res
            
        elif model_type == "prophet":
            res = run_prophet()
            if res: models_run["PRO"] = res
            
        elif model_type == "arima":
            res = run_arima()
            if res: models_run["ARI"] = res
            
        else: # "auto" or unknown -> Run ALL and compare
            res_ses = run_ses()
            if res_ses: models_run["SES"] = res_ses
            
            res_des = run_des()
            if res_des: models_run["DES"] = res_des
            
            res_tes = run_tes()
            if res_tes: models_run["TES"] = res_tes
            
            res_pp = run_prophet()
            if res_pp: models_run["PRO"] = res_pp
            
            res_arima = run_arima()
            if res_arima: models_run["ARI"] = res_arima

        # --- PICK WINNER ---
        if not models_run:
             raise HTTPException(status_code=500, detail="All models failed to run. Check data quality.")

        # Find model with min MSE
        best_model_name = min(models_run, key=lambda k: models_run[k]['mse'])
        best_result = models_run[best_model_name]

        # --- PREPARE RESPONSE ---
        history_data = []
        best_fit_values = best_result['fit']
        
        for date_idx, actual_val in training_series.items():
            fitted_val = None
            if date_idx in best_fit_values.index:
                val = best_fit_values.loc[date_idx]
                if not np.isnan(val) and not np.isinf(val):
                    fitted_val = int(round(val))
            history_data.append({"name": date_idx.strftime('%b %Y'), "actual": int(round(actual_val)), "fitted": fitted_val})

        forecast_data = []
        best_forecast_values = best_result['forecast']
        
        for date_idx, val in best_forecast_values.items():
            clean_val = 0
            if not np.isnan(val) and not np.isinf(val):
                clean_val = int(round(val))
            forecast_data.append({"name": date_idx.strftime('%b %Y') + " (fc)", "forecast": clean_val})

        # Formatting the display name
        display_name = best_model_name
        # If specific model was chosen, we don't need to say "Winner", just the name is fine.
        # But "Winner: SES" still makes sense if we consider it the "selected" model.
        # To match requirements: "without showing the winner" for manual choice implies
        # we might want to change the frontend display, or just change the text here.
        # Let's handle the text in Frontend based on the mode.

        return {
            "history": history_data,
            "forecast": forecast_data,
            "model_name": display_name,
            "mode": model_type, # Send back the mode so frontend knows what to display
            "message": "Success"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")