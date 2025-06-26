import os
import math
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from config import START_YEAR, END_YEAR
import pyet as PyET

# from math import sqrt
# from datetime import datetime
import concurrent.futures

# Load the .env file
load_dotenv()

API_TOKEN = os.getenv("HCDP_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("HCDP_API_TOKEN environment variable not set")
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
    {"name": "Hirako Farm", "lat": 20.002619, "lng": -155.694092},
    {"name": "Anoano Farms", "lat": 20.020913, "lng": -155.693966},
]


API_URL = "https://api.hcdp.ikewai.org/raster/timeseries"

# Optional datatypes to fetch for the rainfall model. Rainfall and
# temperature (Tmax/Tmin) are always retrieved and therefore are not
# included here.
ADDITIONAL_DATATYPES = ["relative_humidity"]
parallel = True

headers = {"Authorization": f"Bearer {API_TOKEN}"}

PARTIAL_END_DATE = "2025-03-12"


def extraterrestrial_radiation_mm(doy, latitude_degs):

    lat = math.radians(latitude_degs)

    Gsc = 0.0820
    dr = 1 + 0.033 * math.cos((2 * math.pi / 365) * doy)
    delta = 0.409 * math.sin((2 * math.pi / 365) * doy - 1.39)
    omega_s = math.acos(-math.tan(lat) * math.tan(delta))

    Ra_MJ = (
        (24 * 60 / math.pi)
        * Gsc
        * dr
        * (
            omega_s * math.sin(lat) * math.sin(delta)
            + math.cos(lat) * math.cos(delta) * math.sin(omega_s)
        )
    )

    Ra_mm = 0.408 * Ra_MJ
    return Ra_mm


def fetch_daily_data_for_year(
    lat, lng, datatype, start_date, end_date, aggregation=None
):
    params = {
        "start": start_date,
        "end": end_date,
        "lat": lat,
        "lng": lng,
        "extent": "statewide",
        "datatype": datatype,
        "period": "day",
    }

    if datatype == "rainfall":
        params["production"] = "new"

    if datatype == "temperature" and aggregation:
        params["aggregation"] = aggregation

    response = requests.get(API_URL, headers=headers, params=params)
    if response.status_code != 200:

        print(
            f"Error fetching data ({datatype}, aggregation={aggregation}, range={start_date}-{end_date}):"
        )
        print(response.status_code, response.text)
        return pd.DataFrame(columns=["Date", "Value"])
    else:

        data = response.json()
        df = pd.DataFrame(list(data.items()), columns=["Date", "Value"])

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def process_farm(farm):
    farm_name = farm["name"]
    lat = farm["lat"]
    lng = farm["lng"]

    farm_folder_name = farm_name.replace(" ", "_").replace("'", "")
    print(f"Processing data for: {farm_name}")

    all_years_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        start_date = f"{year}-01-01"
        end_date = (
            PARTIAL_END_DATE
            if (year == END_YEAR and PARTIAL_END_DATE)
            else f"{year}-12-31"
        )
        print(f"  Year: {year}, Range: {start_date} to {end_date}")

        df_rain = fetch_daily_data_for_year(lat, lng, "rainfall", start_date, end_date)
        df_tmax = fetch_daily_data_for_year(
            lat, lng, "temperature", start_date, end_date, aggregation="max"
        )
        df_tmin = fetch_daily_data_for_year(
            lat, lng, "temperature", start_date, end_date, aggregation="min"
        )

        df_tmean = fetch_daily_data_for_year(
            lat, lng, "temperature", start_date, end_date, aggregation="mean"
        )

        optional_data = {}
        for dtype in ADDITIONAL_DATATYPES:
            df_extra = fetch_daily_data_for_year(lat, lng, dtype, start_date, end_date)
            if not df_extra.empty and df_extra["Value"].notna().any():
                df_extra = df_extra.rename(columns={"Value": dtype})
                optional_data[dtype] = df_extra
            else:
                print(f"    -> {dtype} data missing or empty for {year}")

        df_rain = df_rain.rename(columns={"Value": "Rainfall (mm)"})
        df_tmax = df_tmax.rename(columns={"Value": "Tmax (°C)"})
        df_tmin = df_tmin.rename(columns={"Value": "Tmin (°C)"})
        df_tmean = df_tmean.rename(columns={"Value": "Tmean (°C)"})

        df_merge = pd.merge(df_rain, df_tmax, on="Date", how="outer")
        df_merge = pd.merge(df_merge, df_tmin, on="Date", how="outer")
        df_merge = pd.merge(df_merge, df_tmean, on="Date", how="outer")
        df_merge.sort_values("Date", inplace=True)

        df_merge["Tavg (°C)"] = df_merge[["Tmax (°C)", "Tmin (°C)"]].mean(axis=1)

        # Use Tmean, then fall back to Tavg, else null if both are null
        df_merge["Tmean_final (°C)"] = df_merge.apply(
            lambda row: (
                row["Tmean (°C)"]
                if pd.notnull(row["Tmean (°C)"])
                else (row["Tavg (°C)"] if pd.notnull(row["Tavg (°C)"]) else np.nan)
            ),
            axis=1,
        )

        df_merge["doy"] = df_merge["Date"].dt.dayofyear
        df_merge["Ra_mm"] = df_merge["doy"].apply(
            lambda doy: extraterrestrial_radiation_mm(doy, lat)
        )

        for dtype, df_extra in optional_data.items():
            df_merge = pd.merge(df_merge, df_extra, on="Date", how="left")
            if dtype == "relative_humidity":
                df_merge[dtype] = df_merge[dtype] / 100.0

        def compute_et(row):
            if (
                pd.notnull(row["Tmax (°C)"])
                and pd.notnull(row["Tmin (°C)"])
                and pd.notnull(row["Tmean_final (°C)"])
            ):
                diff = row["Tmax (°C)"] - row["Tmin (°C)"]
                if diff < 0:
                    return None
                return (
                    0.0023
                    * (row["Tmean_final (°C)"] + 17.8)
                    * np.sqrt(diff)
                    * row["Ra_mm"]
                )
            else:
                return None

        def compute_et2(row):
            """
            Computes reference evapotranspiration (ET0) using PyET (Hargreaves).
            Requires: Tmin (°C), Tmax (°C), Tmean (°C), lat,
            """
            if (
                pd.notnull(row["Tmin (°C)"])
                and pd.notnull(row["Tmax (°C)"])
                and pd.notnull(row["Tmean_final (°C)"])
            ):
                try:
                    et0 = PyET.hargreaves(
                        row["Tmean_final (°C)"],
                        row["Tmax (°C)"],
                        row["Tmin (°C)"],
                        lat,
                        k=0.0135,
                        method=0,
                        clip_zero=True,
                    )

                    return et0
                except Exception:
                    return None
            else:
                return None

        df_merge["ET (mm/day)"] = df_merge.apply(compute_et, axis=1)
        df_merge["ET2 (mm/day)"] = df_merge.apply(compute_et2, axis=1)
        df_merge = df_merge.rename(columns={"Ra_mm": "Ra (mm/day)"})

        base_cols = [
            "Date",
            "Rainfall (mm)",
            "Tmax (°C)",
            "Tmin (°C)",
            "Tmean (°C)",  # from api
            "Tavg (°C)",  # average of tmin and tmax
            "Tmean_final (°C)",  # Tmean if available, else Tavg
            "ET (mm/day)",
            # "ET2 (mm/day)",
            "Ra (mm/day)",
        ]

        final_df = df_merge[base_cols + list(optional_data.keys())].copy()

        final_df["Date"] = pd.to_datetime(final_df["Date"]).dt.strftime("%Y-%m-%d")
        final_df["Latitude"] = lat
        final_df["Longitude"] = lng
        final_df["Station"] = farm_name

        if final_df.empty:
            print(f"    -> No data found for {year}, skipping file creation.")
            return

        station_dir = os.path.join("farm_data", farm_folder_name, str(year))
        os.makedirs(station_dir, exist_ok=True)
        csv_path = os.path.join(station_dir, "daily_data.csv")
        final_df.to_csv(csv_path, index=False)
        print(f"    -> Data saved to: {csv_path}")

        all_years_data.append(final_df)

    if all_years_data:
        station_dir = os.path.join("farm_data", farm_folder_name)
        combined_df = pd.concat(all_years_data, ignore_index=True)
        combined_df.sort_values("Date", inplace=True)

        combined_csv_path = os.path.join(station_dir, "all_years_data.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"--> All years combined data saved to: {combined_csv_path}")
    print("")


def main():
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process_farm, farms)
    else:
        for farm in farms:
            process_farm(farm)


if __name__ == "__main__":
    main()
