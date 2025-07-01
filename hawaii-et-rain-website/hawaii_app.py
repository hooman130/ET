"""Streamlit dashboard for displaying Hawaiian ET and rainfall forecasts."""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import streamlit as st
import sqlite3
import schedule
import time
import threading
import functools
from datetime import datetime, timedelta, time as dt_time
import os
import urllib.parse
import json
from dotenv import load_dotenv


from hawaii_web import fetch_and_predict_hawaii, farms, farm_coords, HORIZON

load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Database file is configurable via env; defaults to local file in website dir
DB_FILE = os.getenv('DB_FILE', os.path.join(BASE_DIR, 'hawaii_daily_process.db'))
DEFAULT_HOST = "localhost"
HOST = os.getenv('HOST_IP', DEFAULT_HOST)

# Optional Mapbox token for PyDeck visualizations
MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN')
if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN


farm_names = list(farm_coords.keys())
task_running = False


query_params = st.query_params
api_farm = query_params.get("farm", [None])[0]
api_format = query_params.get("format", ["json"])[0].lower()
if api_farm:
    if api_farm in farm_names:
        with st.spinner(f"Processing API request for {api_farm}..."):
            results = fetch_and_predict_hawaii(api_farm)
        if results:
            output_data = {
                "query_farm": api_farm,
                "prediction_farm": results['farm'],
                "latitude": results['latitude'],
                "longitude": results['longitude'],
                "last_data_date": results['last_data_date'],
                "forecast": [
                    {
                        "date": results['prediction_dates'][i],
                        "et_mm_day": results['et_mm_day'][i] if not np.isnan(results['et_mm_day'][i]) else None,
                        "rain_mm": results['rain_mm'][i] if not np.isnan(results['rain_mm'][i]) else None,
                    } for i in range(HORIZON)
                ]
            }
            if api_format == 'json':
                st.json(output_data)
            elif api_format == 'csv':
                csv_list = []
                for forecast_item in output_data['forecast']:
                    csv_list.append({
                        "farm": output_data['prediction_farm'], "latitude": output_data['latitude'],
                        "longitude": output_data['longitude'], "last_data_date": output_data['last_data_date'],
                        "forecast_date": forecast_item['date'],
                        "et_mm_day": forecast_item['et_mm_day'], "rain_mm": forecast_item['rain_mm']
                    })
                csv_df = pd.DataFrame(csv_list)
                st.write(csv_df.to_csv(index=False))
            else:
                st.json({"error": "Invalid format. Use 'json' or 'csv'."})
        else:
            st.json({"error": "No prediction for this farm."})
        st.stop()
    else:
        st.json({"error": "Invalid farm name."})
        st.stop()




def init_db() -> None:
    """Create the SQLite table used to store processed predictions."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS processed_data
                (timestamp TEXT, farm TEXT, latitude REAL, longitude REAL,
                 pred_date TEXT, et_pred REAL, rain_pred REAL, last_data_date TEXT)'''
    )
    conn.commit()
    conn.close()

def save_to_db(results: dict) -> None:
    """Persist a prediction result dictionary to the database."""
    if not results:
        return
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_data_dt = results['last_data_date']
    lat = results['latitude']
    lon = results['longitude']
    farm = results['farm']

    for i in range(HORIZON):
        pred_dt = results['prediction_dates'][i]
        et_val = results['et_mm_day'][i] if results['et_mm_day'] and not np.isnan(results['et_mm_day'][i]) else None
        rain_val = results['rain_mm'][i] if results['rain_mm'] and not np.isnan(results['rain_mm'][i]) else None
        try:
             c.execute("""
                 INSERT INTO processed_data
                 (timestamp, farm, latitude, longitude, pred_date, et_pred, rain_pred, last_data_date)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
             """, (ts, farm, lat, lon, pred_dt, et_val, rain_val, last_data_dt))
        except sqlite3.Error as e:
             print(f"Database error inserting for {farm} on {pred_dt}: {e}")
    conn.commit()
    conn.close()

def delete_old_db_entries(days_to_keep=1):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        cutoff_date_insertion = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("DELETE FROM processed_data WHERE timestamp < ?", (cutoff_date_insertion,))
        deleted_rows = c.rowcount
        conn.commit()
        conn.close()
        print(f"Deleted {deleted_rows} old entries (older than {days_to_keep} day(s) based on insertion time).")
    except sqlite3.Error as e:
        print(f"Database error during deletion: {e}")


def fetch_latest_data_from_db():
    conn = sqlite3.connect(DB_FILE)
    query = """
    SELECT p.timestamp, p.farm, p.latitude, p.longitude, p.pred_date,
           p.et_pred, p.rain_pred, p.last_data_date
    FROM processed_data p
    INNER JOIN (
        SELECT farm, pred_date, MAX(timestamp) as max_ts
        FROM processed_data
        GROUP BY farm, pred_date
    ) latest ON p.farm = latest.farm AND p.pred_date = latest.pred_date AND p.timestamp = latest.max_ts
    ORDER BY p.farm, p.pred_date;
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error fetching latest data from DB: {e}")
        df = pd.DataFrame()
    conn.close()
    return df




def run_prediction_for_farm(farm_name):
    print(f"Running prediction task for: {farm_name} at {datetime.now()}")
    results = fetch_and_predict_hawaii(farm_name)
    if results:
        save_to_db(results)
        print(f"Successfully processed and saved data for {farm_name}")
    else:
        print(f"Failed to get prediction results for {farm_name}")
    time.sleep(5)

def daily_prediction_task(list_of_farms):
    global task_running
    if not task_running:
        task_running = True
        print(f"--- Starting Daily Prediction Task at {datetime.now()} ---")
        try:
            print("Deleting old database entries (keeping 1 day)...")
            delete_old_db_entries(days_to_keep=1) # Keep 1 day of data

            for farm in list_of_farms:
                run_prediction_for_farm(farm)
            print(f"--- Finished Daily Prediction Task at {datetime.now()} ---")
        except Exception as e:
             print(f"Error in daily_prediction_task: {e}")
        finally:
            task_running = False
    else:
        print("Task is already running. Skipping.")

def schedule_tasks():
    # Time of day for the daily prediction task in HH:MM (24h) format
    schedule_time = os.getenv("SCHEDULE_TIME", "09:37")
    schedule.every().day.at(schedule_time).do(
        functools.partial(daily_prediction_task, farm_names)
    )


    while True:
        schedule.run_pending()
        time.sleep(60)


st.set_page_config(layout="wide", page_title="Hawaii AgroClimatic Forecast")

with st.sidebar:
    try:
        from streamlit_option_menu import option_menu
        selected_page = option_menu(
            "Forecast Menu", ["Overview", "Real-time Forecast", "Historical Data", "API Reference"],
            icons=['house', 'clock-history', 'archive', 'code-slash'],
            menu_icon="cloud-sun-fill", default_index=1,
            styles={
                "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                "icon": {"color": "#007bff", "font-size": "23px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#e9ecef"},
                "nav-link-selected": {"background-color": "#007bff", "color": "white"},
            }
        )
    except ImportError:
        st.error("streamlit-option-menu is not installed. `pip install streamlit-option-menu`")
        selected_page = st.radio("Menu", ["Overview", "Real-time Forecast", "Historical Data", "API Reference"])

init_db()

if 'scheduler_thread_started' not in st.session_state:
    scheduler_thread = threading.Thread(target=schedule_tasks, daemon=True)
    scheduler_thread.start()
    st.session_state.scheduler_thread_started = True
    print("Scheduler thread started.")


if selected_page == "Overview":
    st.title("🌦️ Hawaii AgroClimatic Forecast Dashboard")

    st.image("https://images.unsplash.com/photo-1504509546545-e000Bcb91962?q=80&w=1770&auto=format&fit=crop",
             caption="Lush fields of Hawaii")
    st.markdown("""
    Welcome to the Hawaii AgroClimatic Forecast application. This tool provides valuable 3-day predictions for **Rainfall (mm)** and **Evapotranspiration (ET₀, mm/day)** for key agricultural sites across Hawaii.

    ### How It Works
    Utilizing advanced Long Short-Term Memory (LSTM) neural networks, the system processes recent weather data to generate forecasts.
    - **Data Source:** Real-time atmospheric data (Rainfall, Max/Min Temperature) is sourced from the [HCDP API](https://api.hcdp.ikewai.org/).
    - **Feature Engineering:** Key variables like solar radiation (Ra) and reference evapotranspiration (ET₀, via Hargreaves-Samani equation) are calculated. Time-based features (day, month) are also included.
    - **Prediction Models:** Separate LSTM models, trained on historical data for each specific variable (ET and Rainfall), predict the next 3 days based on a 24-day input sequence.
    - **Data Scaling:** Features are scaled using `StandardScaler` before model input, and predictions are inversely transformed back to their original units.

    ### Application Features
    - **Real-time Forecast:** Select a farm to view the latest 3-day predictions, visualized on an interactive map.
    - **Historical Data:** Access and search recently stored predictions from our database.
    - **API Access:** Programmatically retrieve forecast data via a simple API (see API Reference section).
    - **Daily Updates:** The forecast models run daily to provide the most current information.

    This dashboard is designed to aid farmers and agricultural stakeholders in making informed decisions.
    """)

elif selected_page == "Real-time Forecast":
    st.header("🛰️ Real-time Forecast")
    st.markdown("Select a farm to view its latest 3-day rainfall and evapotranspiration (ET₀) forecast.")

    selected_farm = st.selectbox("Select Farm Location:", farm_names, index=0, help="Choose a farm from the list to see its forecast.")

    if selected_farm:
        lat = farm_coords[selected_farm]["lat"]
        lon = farm_coords[selected_farm]["lng"]

        st.info(f"Fetching and processing data for **{selected_farm}** (Lat: {lat:.4f}, Lon: {lon:.4f}). Please wait...")

        with st.spinner(f"Generating forecast for {selected_farm}... This may take a moment."):
            results = fetch_and_predict_hawaii(selected_farm) # تابع از hawaii_web.py

        if results:
            st.subheader(f"Forecast for: {results['farm']}")
            st.caption(f"Based on input data up to: **{results['last_data_date']}**")

            # نمایش متریک‌ها مثل قبل
            cols_metrics = st.columns(HORIZON)
            for i in range(HORIZON):
                with cols_metrics[i]:
                    st.metric(
                        label=f"{datetime.strptime(results['prediction_dates'][i], '%Y-%m-%d').strftime('%a, %b %d')}",
                        value=f"{results['rain_mm'][i]:.2f} mm" if not np.isnan(results['rain_mm'][i]) else "N/A",
                        delta=f"{results['et_mm_day'][i]:.2f} mm/day ET" if not np.isnan(results['et_mm_day'][i]) else "N/A ET",
                        delta_color="off"
                    )

            pred_df_data = {
                "Date": results['prediction_dates'],
                "Rainfall (mm)": [f"{x:.2f}" if not np.isnan(x) else "N/A" for x in results['rain_mm']],
                "ET₀ (mm/day)": [f"{x:.2f}" if not np.isnan(x) else "N/A" for x in results['et_mm_day']]
            }
            pred_df = pd.DataFrame(pred_df_data).set_index('Date')
            st.dataframe(pred_df, use_container_width=True)

            # --- Map Visualization with PyDeck ColumnLayers ---
            st.subheader("🗺️ Map Visualization (Forecast Bars)")

            map_data_points = []
            bar_spacing_longitude = 0.0008  # فاصله طولی بین میله‌ها - قابل تنظیم
            group_spacing_longitude = 0.0025 # فاصله طولی بین گروه ET و بارش - قابل تنظیم

            # رنگ‌ها
            et_color_day1 = [50, 180, 50, 200]   # Greenish
            et_color_day2 = [100, 200, 100, 200] # Lighter Green
            et_color_day3 = [150, 220, 150, 200] # Even Lighter Green
            rain_color_day1 = [30, 144, 255, 200] # Blue

            # ET Bars (3 days)
            for i in range(HORIZON): # HORIZON should be 3
                et_value = results['et_mm_day'][i]
                if pd.notna(et_value) and et_value > 0: # Only add bar if value exists and is positive
                    offset = (-1 + i) * bar_spacing_longitude # ET Day1, Day2, Day3 next to each other
                    map_data_points.append({
                        "lat": lat,
                        "lon": lon + offset,
                        "value": float(et_value), # Ensure float for PyDeck
                        "type": f"ET Day {i+1}",
                        "date": results['prediction_dates'][i],
                        "color": [et_color_day1, et_color_day2, et_color_day3][i],
                        "tooltip_text": f"<b>Farm:</b> {results['farm']}<br/>"
                                        f"<b>Type:</b> ET Day {i+1}<br/>"
                                        f"<b>Date:</b> {results['prediction_dates'][i]}<br/>"
                                        f"<b>Value:</b> {et_value:.2f} mm/day"
                    })

            # Rainfall Bar (Day 1 only, as requested "1 ستون")
            # Positioned with a gap from the ET bars
            rain_value_day1 = results['rain_mm'][0]
            if pd.notna(rain_value_day1) and rain_value_day1 > 0: # Only add bar if value exists and is positive
                rain_offset = (1 * bar_spacing_longitude) + group_spacing_longitude # Position rain bar to the right of ET group
                map_data_points.append({
                    "lat": lat,
                    "lon": lon + rain_offset, # Offset for rain bar
                    "value": float(rain_value_day1), # Ensure float
                    "type": "Rainfall Day 1",
                    "date": results['prediction_dates'][0],
                    "color": rain_color_day1,
                    "tooltip_text": f"<b>Farm:</b> {results['farm']}<br/>"
                                    f"<b>Type:</b> Rainfall Day 1<br/>"
                                    f"<b>Date:</b> {results['prediction_dates'][0]}<br/>"
                                    f"<b>Value:</b> {rain_value_day1:.2f} mm"
                })

            if not map_data_points:
                st.warning("No positive forecast values are available to display on the map for the selected farm.")
            else:
                map_df = pd.DataFrame(map_data_points)

                # Define PyDeck ColumnLayer
                column_layer = pdk.Layer(
                    "ColumnLayer",
                    data=map_df,
                    get_position="[lon, lat]",
                    get_elevation="value",  # Height of the bar
                    elevation_scale=100,   # Scale factor for elevation (adjust as needed)
                    radius=30,             # Radius of the bars in meters (adjust as needed)
                    get_fill_color="color", # Use the color column from map_df
                    pickable=True,
                    auto_highlight=True,
                    extruded=True, # Gives 3D effect to bars
                )

                # View state for the map
                view_state = pdk.ViewState(
                    latitude=lat,
                    longitude=lon,
                    zoom=14, # Zoom closer to see bars
                    pitch=45, # Angle for 3D view
                    bearing=0
                )

                # Render PyDeck map
                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/satellite-streets-v12', # Using a Mapbox style
                    initial_view_state=view_state,
                    layers=[column_layer],
                    tooltip={"html": "{tooltip_text}"} # Use the pre-formatted tooltip column
                ))

                # Simple Legend
                st.markdown("""
                **Map Legend:**
                <div style="line-height: 1.5;">
                <span style='background-color:rgba(50, 180, 50, 0.7); color:white; padding: 2px 5px; margin-right: 5px;'>&nbsp;ET Day 1&nbsp;</span>
                <span style='background-color:rgba(100, 200, 100, 0.7); color:white; padding: 2px 5px; margin-right: 5px;'>&nbsp;ET Day 2&nbsp;</span>
                <span style='background-color:rgba(150, 220, 150, 0.7); color:white; padding: 2px 5px; margin-right: 5px;'>&nbsp;ET Day 3&nbsp;</span>
                <br/>
                <span style='background-color:rgba(30, 144, 255, 0.7); color:white; padding: 2px 5px; margin-right: 5px;'>&nbsp;Rainfall Day 1&nbsp;</span>
                </div>
                <small>(Bars represent positive forecast values. Height is proportional to the value. Adjust `elevation_scale` and `radius` in code for better visualization if needed.)</small>
                """, unsafe_allow_html=True)
        else:
            st.error(f"❌ Could not retrieve or calculate predictions for {selected_farm}. The data source might be temporarily unavailable or input data might be insufficient. Please check back later or try another station.")


elif selected_page == "Historical Data":
    st.header("📜 Historical Predictions (Condensed View)")
    st.write("Latest forecast for each farm. Each row shows the most recent prediction (rain for Day 1 and ET for three days).")

    df_offline = fetch_latest_data_from_db()
    if df_offline.empty:
        st.warning("No historical data found.")
    else:

        df_latest = (
            df_offline
            .sort_values(['farm', 'pred_date', 'timestamp'], ascending=[True, False, False])
            .groupby('farm')
            .head(3)   
            .sort_values(['farm', 'pred_date'], ascending=[True, True])
        )

        def get_pivot_row(subdf):
            subdf = subdf.sort_values('pred_date')
            rain_val = subdf.iloc[0]['rain_pred']   # Rain for Day 1
            et0 = [np.nan, np.nan, np.nan]
            for idx, row in enumerate(subdf.itertuples()):
                if idx < 3:
                    et0[idx] = row.et_pred
            return pd.Series({
                'Update Time': subdf.iloc[0]['timestamp'],
                'Rain (Day 1)': f"{rain_val:.4f}" if pd.notnull(rain_val) else np.nan,
                'ET-Today': f"{et0[0]:.4f}" if pd.notnull(et0[0]) else np.nan,
                'ET-Tomorrow': f"{et0[1]:.4f}" if pd.notnull(et0[1]) else np.nan,
                'ET-Day After Tomorrow': f"{et0[2]:.4f}" if pd.notnull(et0[2]) else np.nan,
            })
        df_display = (
            df_latest.groupby('farm')
            .apply(get_pivot_row)
            .reset_index()
            .rename(columns={'farm': 'Station'})
        )


        search_term = st.text_input("Search for a farm name:", value="")
        if search_term:
            df_display = df_display[df_display['Station'].str.contains(search_term, case=False, na=False)]

        st.dataframe(df_display, use_container_width=True)

        # --- Simple Map: one marker per farm, with simple popup ---
        import folium
        from streamlit_folium import st_folium

        if not df_display.empty:
            farm_lats = [farm_coords.get(x, {}).get('lat', 20.8) for x in df_display['Station']]
            farm_lons = [farm_coords.get(x, {}).get('lng', -157.5) for x in df_display['Station']]
            m = folium.Map(location=[np.mean(farm_lats), np.mean(farm_lons)], zoom_start=7)
            folium.TileLayer('openstreetmap').add_to(m)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Esri Satellite', overlay=False, control=True
            ).add_to(m)
            for i, row in df_display.iterrows():
                tooltip = (
                    f"<b>{row['Station']}</b><br>"
                    f"Rain (Day 1): {row['Rain (Day 1)']}<br>"
                    f"ET Today: {row['ET-Today']}<br>"
                    f"ET Tomorrow: {row['ET-Tomorrow']}<br>"
                    f"ET Day After Tomorrow: {row['ET-Day After Tomorrow']}"
                )
                lat = farm_coords.get(row['Station'], {}).get('lat', 20.8)
                lon = farm_coords.get(row['Station'], {}).get('lng', -157.5)
                folium.Marker(
                    location=[lat, lon],
                    tooltip=tooltip,
                    icon=folium.Icon(color="blue", icon="cloud")
                ).add_to(m)
            st_folium(m, width=None, height=500, use_container_width=True)

elif selected_page == "API Reference":
    st.header("⚙️ API Reference")
    st.write("Retrieve the latest forecast for any farm as JSON or CSV. Select a farm and the output format to see the API URL and live results.")

    farm_names_sorted = sorted(farm_names)
    selected_farm = st.selectbox("Select Farm:", farm_names_sorted, key="api_ref_farm")
    output_format = st.selectbox("Select Output Format:", ["json", "csv"], key="api_ref_format")


    encoded_farm = urllib.parse.quote(selected_farm)
    base_url = f"http://{HOST}:8501/"
    api_url = f"{base_url}?farm={encoded_farm}&format={output_format}"

    st.markdown("**API URL:**")
    st.code(api_url, language="bash")


    import requests

    try:
        r = requests.get(api_url)
        if r.status_code == 200:
            if output_format == "json":
                st.markdown("**Live JSON Output:**")
                st.json(r.json())
            else:
                st.markdown("**Live CSV Output:**")
                st.code(r.text, language="csv")
        else:
            st.warning(f"API call returned status code {r.status_code}.")
    except Exception as e:
        st.warning(f"Could not fetch live API result: {e}")
