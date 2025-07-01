import os
import math
import requests
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime, timedelta
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from config import MODELS_DIR as CONFIG_MODELS_DIR, WINDOW_SIZE as CONFIG_WINDOW_SIZE, HORIZON as CONFIG_HORIZON

load_dotenv()





API_TOKEN = os.getenv("HCDP_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("HCDP_API_TOKEN environment variable not set")
API_URL = "https://api.hcdp.ikewai.org/raster/timeseries"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


farms = [
    {"name": "Kahuku Farm", "lat": 21.6832, "lng": -157.9604},
    {"name": "Nozawa Farms", "lat": 21.688, "lng": -157.9648},
    {"name": "Kuilima Farms", "lat": 21.6958, "lng": -158.0053},
    {"name": "Cabaero Farms", "lat": 20.8425, "lng": -156.3471},
    {"name": "Kupa'a Farms", "lat": 20.7658, "lng": -156.3513},
    {"name": "MA'O Organic Farms (original site)", "lat": 21.4645, "lng": -158.1132},
    {"name": "MA'O Organic Farms (new site)", "lat": 21.41505, "lng": -158.13707},
    {"name": "2K Farm LLC", "lat": 21.445354, "lng": -158.181649},
    {"name": "Wong Hon Hin Inc", "lat": 21.466595, "lng": -158.164714},
    {"name": "Hawaii Taro Farm, LLC", "lat": 20.839723, "lng": -156.510438},
    {"name": "Hawaii Seed Pro LLC Farm", "lat": 20.796725, "lng": -156.359714},
    {"name": "Cabaero Farm", "lat": 20.791703, "lng": -156.358194}, 
    {"name": "Kupaa Farms2", "lat": 20.765515, "lng": -156.35185},
    {"name": "Hirako Farm", "lat": 20.018748, "lng": -155.692546}, 
    {"name": "Hirako Farm1", "lat": 20.002619, "lng": -155.694092},
    {"name": "Anoano Farms", "lat": 20.020913, "lng": -155.693966},
]
farm_coords = {farm["name"]: {"lat": farm["lat"], "lng": farm["lng"]} for farm in farms}

# Directory containing per-farm models and scaler files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", CONFIG_MODELS_DIR)

# Caches for loaded models and scalers to avoid re-loading on each request
models_cache = {}
scalers_cache = {}


WINDOW_SIZE = CONFIG_WINDOW_SIZE
HORIZON = CONFIG_HORIZON
FETCH_TOTAL_DAYS = 50 




FEATURE_COLS_FOR_SCALER = ['Rainfall (mm)', 'Tmax (°C)', 'Tmin (°C)', 'ET (mm/day)', 'Ra (mm/day)', 'day', 'month']




@tf.keras.utils.register_keras_serializable()
def r2_keras(y_true, y_pred):
    """ Custom R² metric for Keras. """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    ss_res = K.sum(K.square(y_true_f - y_pred_f))
    ss_tot = K.sum(K.square(y_true_f - K.mean(y_true_f)))
    return 1 - ss_res / (ss_tot + K.epsilon())

dependencies = {'r2_keras': r2_keras}

def sanitize_farm_name(farm_name: str) -> str:
    """Convert farm name to a filesystem friendly format used for models."""
    return (farm_name.replace(" ", "_")
                     .replace("'", "")
                     .replace(",", "")
                     .replace("(", "")
                     .replace(")", ""))


def load_models_for_farm(farm_name):
    """Load ET/Rain models and scalers for a specific farm, using caches when available."""
    if farm_name in models_cache and farm_name in scalers_cache:
        return (models_cache[farm_name].get('et'),
                models_cache[farm_name].get('rain'),
                scalers_cache[farm_name].get('et'),
                scalers_cache[farm_name].get('rain'))

    base = sanitize_farm_name(farm_name)
    model_et_path = os.path.join(MODELS_DIR, f"{base}_model_lstm.keras")
    model_rain_path = os.path.join(MODELS_DIR, f"{base}_model_rain_lstm.keras")
    scaler_et_path = os.path.join(MODELS_DIR, f"{base}_scaler_ET.pkl")
    scaler_rain_path = os.path.join(MODELS_DIR, f"{base}_scaler_Rain.pkl")

    try:
        model_et = load_model(model_et_path, custom_objects=dependencies, compile=False)
        model_et.compile(optimizer='adam', loss='mse')
    except Exception as e:
        print(f"Error loading ET model for {farm_name}: {e}")
        model_et = None

    try:
        model_rain = load_model(model_rain_path, custom_objects=dependencies, compile=False)
        model_rain.compile(optimizer='adam', loss='mse')
    except Exception as e:
        print(f"Error loading Rainfall model for {farm_name}: {e}")
        model_rain = None

    try:
        scaler_et = joblib.load(scaler_et_path)
    except Exception as e:
        print(f"Error loading ET scaler for {farm_name}: {e}")
        scaler_et = None

    try:
        scaler_rain = joblib.load(scaler_rain_path)
    except Exception as e:
        print(f"Error loading Rain scaler for {farm_name}: {e}")
        scaler_rain = None

    models_cache[farm_name] = {'et': model_et, 'rain': model_rain}
    scalers_cache[farm_name] = {'et': scaler_et, 'rain': scaler_rain}

    return model_et, model_rain, scaler_et, scaler_rain




def extraterrestrial_radiation_mm(doy, latitude_degs):
    """ Compute daily extraterrestrial radiation (Ra) in mm/day. """
    lat = math.radians(latitude_degs)
    Gsc = 0.082  
    dr = 1 + 0.033 * math.cos((2 * math.pi / 365) * doy)
    delta = 0.409 * math.sin((2 * math.pi / 365) * doy - 1.39)
    try:
        cos_omega_s = -math.tan(lat) * math.tan(delta)
        
        cos_omega_s = max(-1.0, min(1.0, cos_omega_s))
        omega_s = math.acos(cos_omega_s)
    except ValueError: 
        omega_s = math.pi if math.tan(lat) * math.tan(delta) > 1.0 else 0.0 

    Ra_MJ = (24 * 60 / math.pi) * Gsc * dr * (
        omega_s * math.sin(lat) * math.sin(delta) +
        math.cos(lat) * math.cos(delta) * math.sin(omega_s)
    )
    return max(0.0, 0.408 * Ra_MJ) 

def hargreaves_samani_et0_mm(tmax, tmin, tmean, ra):
    """ Hargreaves-Samani ET0 in mm/day. """
    tdiff = tmax - tmin
    
    tdiff = max(0.0, tdiff) 
    return 0.0023 * (tmean + 17.8) * sqrt(tdiff) * ra

def fetch_recent_daily_data(lat, lng, datatype, start_date, end_date, aggregation=None):
    """ Fetches daily data from HCDP API for a specific date range. """
    params = {
        "start": start_date, "end": end_date, "lat": lat, "lng": lng,
        "extent": "statewide", "datatype": datatype, "period": "day"
    }
    if datatype == "rainfall": params["production"] = "new"
    if datatype == "temperature" and aggregation: params["aggregation"] = aggregation

    try:
        response = requests.get(API_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status() 
        data = response.json()
        if not data: 
            print(f"API returned empty data for {datatype}, agg={aggregation}, range={start_date}-{end_date}")
            return pd.DataFrame(columns=["Date", "Value"])
        df = pd.DataFrame(list(data.items()), columns=["Date", "Value"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"API Request failed ({datatype}, agg={aggregation}): {e}")
    except Exception as e: 
        print(f"Error processing data ({datatype}, agg={aggregation}): {e}")
    return pd.DataFrame(columns=["Date", "Value"]) 

def prepare_input_features(df_raw_window, latitude):
    """
    Prepares the DataFrame with all necessary features for scaling and prediction.
    Assumes df_raw_window contains 'Rainfall (mm)', 'Tmax (°C)', 'Tmin (°C)', 'Date'.
    """
    if df_raw_window.empty or not all(col in df_raw_window.columns for col in ['Rainfall (mm)', 'Tmax (°C)', 'Tmin (°C)', 'Date']):
        print("Error: df_raw_window is empty or missing required columns for feature preparation.")
        return pd.DataFrame(), None

    df = df_raw_window.copy() 
    
    df['Date'] = pd.to_datetime(df['Date'])

    
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["doy"] = df["Date"].dt.dayofyear 
    df["Tmean"] = (df["Tmax (°C)"] + df["Tmin (°C)"]) / 2.0
    df["Ra (mm/day)"] = df.apply(lambda row: extraterrestrial_radiation_mm(row["doy"], latitude), axis=1)
    df["ET (mm/day)"] = df.apply(lambda row: hargreaves_samani_et0_mm(row["Tmax (°C)"], row["Tmin (°C)"], row["Tmean"], row["Ra (mm/day)"]), axis=1)

    
    try:
        df_featured = df[FEATURE_COLS_FOR_SCALER].copy()
    except KeyError as e:
        print(f"Error selecting feature columns: {e}. Available columns: {df.columns.tolist()}")
        return pd.DataFrame(), None

    
    if df_featured.isnull().values.any():
        print("Warning: NaNs found after feature engineering. Attempting forward fill (ffill).")
        print(df_featured[df_featured.isnull().any(axis=1)])
        df_featured.ffill(inplace=True) 
        df_featured.bfill(inplace=True) 
        if df_featured.isnull().values.any():
             
             print("Error: NaNs persist even after ffill and bfill. Cannot proceed with prediction.")
             return pd.DataFrame(), None

    return df_featured, df['Date'] 

def inverse_transform_predictions(y_scaled, target_scaler, feature_column_order, target_col_name):
    """
    Inversely transforms predictions using the provided scaler.
    y_scaled shape: (1, HORIZON) for single sequence prediction
    """
    if target_scaler is None:
        print(f"Error: Scaler for '{target_col_name}' not loaded.")
        return None
    if y_scaled.ndim == 1: y_scaled = y_scaled.reshape(1, -1) 

    n_samples, horizon = y_scaled.shape 
    n_features_expected_by_scaler = len(feature_column_order)

    try:
        target_idx_in_scaler = feature_column_order.index(target_col_name)
    except ValueError:
        print(f"Critical Error: Target column '{target_col_name}' not found in scaler's expected columns: {feature_column_order}")
        return None

    
    
    dummy_array_for_horizon = np.zeros((horizon, n_features_expected_by_scaler))
    
    
    dummy_array_for_horizon[:, target_idx_in_scaler] = y_scaled[0, :] 

    try:
        inversed_dummy_horizon = target_scaler.inverse_transform(dummy_array_for_horizon)
    except Exception as e:
        print(f"Error during inverse transform for '{target_col_name}': {e}")
        return None

    
    inversed_target_values = inversed_dummy_horizon[:, target_idx_in_scaler]

    return inversed_target_values.reshape(1, horizon) 




def fetch_and_predict_hawaii(farm_name):

    model_et, model_rain, scaler_et, scaler_rain = load_models_for_farm(farm_name)

    if not all([model_et, model_rain, scaler_et, scaler_rain]):
        print(f"Error: One or more models/scalers could not be loaded for {farm_name}.")
        return None

    if farm_name not in farm_coords:
        print(f"Error: Farm '{farm_name}' not found in coordinate list.")
        return None

    lat = farm_coords[farm_name]["lat"]
    lng = farm_coords[farm_name]["lng"]

    
    end_date_dt = datetime.now()
    start_date_dt = end_date_dt - timedelta(days=FETCH_TOTAL_DAYS - 1) 
    end_date_str = end_date_dt.strftime('%Y-%m-%d')
    start_date_str = start_date_dt.strftime('%Y-%m-%d')

    print(f"Fetching data ({FETCH_TOTAL_DAYS} days) for {farm_name} ({lat}, {lng}) from {start_date_str} to {end_date_str}")
    df_rain_raw = fetch_recent_daily_data(lat, lng, "rainfall", start_date_str, end_date_str)
    df_tmax_raw = fetch_recent_daily_data(lat, lng, "temperature", start_date_str, end_date_str, aggregation="max")
    df_tmin_raw = fetch_recent_daily_data(lat, lng, "temperature", start_date_str, end_date_str, aggregation="min")

    
    df_rain_raw = df_rain_raw.rename(columns={"Value": "Rainfall (mm)"})
    df_tmax_raw = df_tmax_raw.rename(columns={"Value": "Tmax (°C)"})
    df_tmin_raw = df_tmin_raw.rename(columns={"Value": "Tmin (°C)"})

    if df_rain_raw.empty or df_tmax_raw.empty or df_tmin_raw.empty:
        print(f"Error: Failed to fetch one or more essential data types (Rain, Tmax, Tmin) in the range {start_date_str} to {end_date_str}.")
        return None

    
    try:
        
        
        
        df_merged = pd.merge(df_rain_raw, df_tmax_raw, on="Date", how="inner")
        df_merged = pd.merge(df_merged, df_tmin_raw, on="Date", how="inner")
    except Exception as e:
         print(f"Error merging API data: {e}")
         return None

    df_merged.sort_values("Date", inplace=True, ascending=True) 

    
    if len(df_merged) < WINDOW_SIZE:
        print(f"Error: Insufficient complete data after merging. Found {len(df_merged)} days with Rain, Tmax, and Tmin. Expected at least {WINDOW_SIZE}.")
        print(f"  -> This usually means the API has missing data points within the last {FETCH_TOTAL_DAYS} fetched days.")
        return None
    else:
        
        df_window = df_merged.tail(WINDOW_SIZE).copy() 
        print(f"Successfully obtained {len(df_window)} complete records for the prediction window from fetched data.")
        if len(df_window) != WINDOW_SIZE: 
             print(f"Critical Error: Selected window size is {len(df_window)}, expected {WINDOW_SIZE}. Aborting.")
             return None

    
    
    df_features, feature_dates = prepare_input_features(df_window, lat)

    if df_features.empty or len(df_features) != WINDOW_SIZE:
        print("Error: Feature preparation failed or resulted in incorrect number of rows after window selection.")
        return None

    
    pred_et_inv = None 
    try:
        if scaler_et and model_et:
            scaled_features_et = scaler_et.transform(df_features[FEATURE_COLS_FOR_SCALER])
            input_sequence_et = scaled_features_et.reshape(1, WINDOW_SIZE, len(FEATURE_COLS_FOR_SCALER))
            pred_et_scaled = model_et.predict(input_sequence_et) 
            pred_et_inv = inverse_transform_predictions(
                pred_et_scaled, scaler_et, FEATURE_COLS_FOR_SCALER, "ET (mm/day)"
            ) 
        else:
            print("ET scaler or model not available. Skipping ET prediction.")
    except Exception as e:
        print(f"Error during ET prediction/scaling: {e}")
        

    
    pred_rain_inv = None 
    try:
        if scaler_rain and model_rain:
            scaled_features_rain = scaler_rain.transform(df_features[FEATURE_COLS_FOR_SCALER])
            input_sequence_rain = scaled_features_rain.reshape(1, WINDOW_SIZE, len(FEATURE_COLS_FOR_SCALER))
            pred_rain_scaled = model_rain.predict(input_sequence_rain) 
            pred_rain_inv = inverse_transform_predictions(
                pred_rain_scaled, scaler_rain, FEATURE_COLS_FOR_SCALER, "Rainfall (mm)"
            ) 
        else:
            print("Rainfall scaler or model not available. Skipping Rainfall prediction.")
    except Exception as e:
        print(f"Error during Rainfall prediction/scaling: {e}")
        

    
    if pred_et_inv is None and pred_rain_inv is None: 
        print("Both ET and Rainfall predictions failed or were skipped.")
        return None

    
    last_input_date = feature_dates.iloc[-1]
    prediction_dates_dt = [last_input_date + timedelta(days=i) for i in range(1, HORIZON + 1)]

    
    et_predictions_list = pred_et_inv.flatten().tolist() if pred_et_inv is not None else [np.nan] * HORIZON
    rain_predictions_list = pred_rain_inv.flatten().tolist() if pred_rain_inv is not None else [np.nan] * HORIZON

    
    et_predictions_list = [max(0, p) if pd.notna(p) else np.nan for p in et_predictions_list]
    rain_predictions_list = [max(0, p) if pd.notna(p) else np.nan for p in rain_predictions_list]

    results = {
        "farm": farm_name,
        "latitude": lat,
        "longitude": lng,
        "prediction_dates": [d.strftime('%Y-%m-%d') for d in prediction_dates_dt],
        "et_mm_day": et_predictions_list,
        "rain_mm": rain_predictions_list,
        "last_data_date": last_input_date.strftime('%Y-%m-%d')
    }
    return results


if __name__ == "__main__":

    print("--- Running hawaii_web.py directly for testing ---")

    test_farm = "Kahuku Farm"

    if test_farm not in farm_coords:
        print(f"Test farm '{test_farm}' not in farm_coords list. Available: {list(farm_coords.keys())}")
    else:
        model_et, model_rain, scaler_et, scaler_rain = load_models_for_farm(test_farm)
        if not all([model_et, model_rain, scaler_et, scaler_rain]):
            print("Failed to load required models or scalers for testing.")
        else:
            print(f"\nAttempting prediction for test farm: {test_farm}...")
            prediction_results = fetch_and_predict_hawaii(test_farm)

            if prediction_results:
                print("\n--- Prediction Results ---")
                print(f"Farm: {prediction_results['farm']}")
                print(f"Latitude: {prediction_results['latitude']:.4f}, Longitude: {prediction_results['longitude']:.4f}")
                print(f"Last Input Data Date Used: {prediction_results['last_data_date']}")
                print("Forecast:")
                for i in range(HORIZON):
                    pred_date = prediction_results['prediction_dates'][i]
                    et_val = prediction_results['et_mm_day'][i]
                    rain_val = prediction_results['rain_mm'][i]

                    et_str = f"{et_val:.2f}" if pd.notna(et_val) else "N/A"
                    rain_str = f"{rain_val:.2f}" if pd.notna(rain_val) else "N/A"
                    print(f"  - {pred_date}: ET₀ = {et_str} mm/day, Rainfall = {rain_str} mm")
            else:
                print(f"\n❌ Prediction failed for {test_farm}. Check console for detailed error messages.")

    
    
