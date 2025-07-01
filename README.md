# Hawaii ET/Rain Forecast

This project provides evapotranspiration and rainfall forecasts for various farms in Hawaii.  
The web application is built with Streamlit and exposes a simple API using Flask.

## Environment Variables

The application relies on several environment variables which can be placed in a `.env` file:

- `HCDP_API_TOKEN` – **required** token for the HCDP data API.
- `FETCH_TOTAL_DAYS` – number of days of historical data to pull from the API (default: `50`).
- `DB_FILE` – path to the SQLite database file (default: `hawaii-et-rain-website/hawaii_daily_process.db`).
- `SCHEDULE_TIME` – daily prediction time in `HH:MM` 24‑hour format (default: `09:37`).
- `MAPBOX_TOKEN` – optional token for Mapbox tiles used in PyDeck maps.

## Running the Streamlit App

```bash
pip install -r hawaii-et-rain-website/requirements.txt
streamlit run hawaii-et-rain-website/hawaii_app.py
```

## Running the API Server

```bash
pip install -r hawaii-et-rain-website/requirements.txt
python hawaii-et-rain-website/hawaii_api.py
```

The API exposes `/api/predict?farm=<FARM_NAME>&format=json|csv`.
