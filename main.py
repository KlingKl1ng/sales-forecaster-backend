from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import logging
import xlsxwriter 
import os
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

# --- MODEL NAMING MAPPING ---
MODEL_DISPLAY_NAMES = {
    "ses": "SES (Level)",
    "des": "DES (Trend)",
    "tes": "TES (Seasonal)",
    "prophet": "Prophet (Complex)"
}

# --- NEW: HELPER FOR EXPORT ---
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

# --- VALIDATION ENDPOINT ---
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

        # --- DEFINING MODEL LOGIC (Local Scope for Validation) ---
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
            # UPDATED LOGIC: Simple Split = First valid run (Train 1..N -> Forecast N+1)
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
                        if models_stitched_forecasts[m].empty: models_stitched_forecasts[m] = fc
                        else: models_stitched_forecasts[m] = pd.concat([models_stitched_forecasts[m], fc])

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


# --- PREDICTION ENDPOINT ---
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

# --- NEW: EXPORT ENDPOINT ---
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
        logger.info(f"Generating Report for {model_type} using {validation_method}")
        contents = await file.read()
        full_series = load_and_prep_data(contents)
        
        # --- 1. RUN VALIDATION (Conditional Logic) ---
        val_forecast = None
        
        if validation_method == "simple":
            # Logic: First Slice (Train 1..N -> Forecast N+1)
            if len(full_series) < train_horizon + test_horizon:
                 raise HTTPException(status_code=400, detail="Not enough data for simple validation")
            
            # Just take the slice required for the first run
            validation_scope_df = full_series.iloc[:train_horizon + test_horizon]
            train_subset = validation_scope_df.iloc[:train_horizon]
            val_forecast = _run_model_logic(train_subset, model_type, test_horizon)

        elif validation_method == "walk_forward":
            # Logic: Loop through all windows
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
            worksheet.insert_image('B2', logo_path, {'x_scale': 0.05, 'y_scale': 0.05})
            
        worksheet.set_row(1, 45) 
        worksheet.set_row(2, 25) 

        worksheet.merge_range('C2:F2', "OPERARTIS INTELLIGENCE REPORT", fmt_header_main)
        worksheet.merge_range('C3:F3', "Optimizing Today, Growing Tomorrow.", fmt_header_sub)
        
        worksheet.write('B5', "Report By:", fmt_meta_label)
        worksheet.write('C5', "Operartis Analytics", fmt_meta_value)
        
        worksheet.write('B6', "Report Date:", fmt_meta_label)
        worksheet.write('C6', datetime.now().strftime("%Y-%m-%d %H:%M"), fmt_meta_value)
        
        worksheet.write('B7', "Model Used:", fmt_meta_label)
        worksheet.write('C7', MODEL_DISPLAY_NAMES.get(model_type, model_type), fmt_meta_value)
        
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
        worksheet.protect('Operartis', {'select_locked_cells': True, 'select_unlocked_cells': True})
        workbook.close()
        output.seek(0)
        
        filename = f"Operartis_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))