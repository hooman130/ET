import os
import math
import requests
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime

# -------------------------------
# 1. Configuration
# -------------------------------
API_TOKEN = (
    "1b8a6439c85b8e42e211b68ea68ac198"  # Replace with your actual HCDP API token
)


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

# Base API URL for timeseries data
API_URL = "https://api.hcdp.ikewai.org/raster/timeseries"

# Headers with API Authorization
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Define the range of years you want to process
START_YEAR = 2000
END_YEAR = 2025  # inclusive (adjust as needed)

# If you want the final year to end on a partial date (like 2024-03-12), define it here
PARTIAL_END_DATE = "2025-03-12"


# -------------------------------
# 2. Helper functions
# -------------------------------
def extraterrestrial_radiation_mm(doy, latitude_degs):
    """
    Compute daily extraterrestrial radiation (Ra) in mm/day
    for a given day of year and latitude (in degrees) using the FAO-56 approach.
    """
    lat = math.radians(latitude_degs)
    Gsc = 0.082  # Solar constant (MJ/m^2/min)
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
    """
    Fetches daily data from the HCDP API for the given parameters and returns a DataFrame.

    For rainfall, the parameter "production": "new" is included.
    For temperature, if an aggregation is provided (e.g., "max" or "min"), it is added.
    """
    params = {
        "start": start_date,
        "end": end_date,
        "lat": lat,
        "lng": lng,
        "extent": "statewide",
        "datatype": datatype,
        "period": "day",
    }
    # Include production parameter only for rainfall.
    if datatype == "rainfall":
        params["production"] = "new"
    # For temperature, add the aggregation parameter if provided.
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
        data = response.json()  # Expected to be a dictionary with ISO-8601 date key
        df = pd.DataFrame(list(data.items()), columns=["Date", "Value"])
        # Convert Date column to timezone-naive datetimes
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


# -------------------------------
# 3. Main function: loop over years, fetch data, compute ET, and save CSV
# -------------------------------
def main():
    for farm in farms:
        farm_name = farm["name"]
        lat = farm["lat"]
        lng = farm["lng"]

        # Clean folder name (remove spaces, apostrophes, etc.)
        farm_folder_name = farm_name.replace(" ", "_").replace("'", "")
        print(f"Processing data for: {farm_name}")

        # We'll keep track of a list of DataFrames for each year.
        all_years_data = []

        for year in range(START_YEAR, END_YEAR + 1):
            start_date = f"{year}-01-01"
            # Use a partial end date for the final year if provided.
            end_date = (
                PARTIAL_END_DATE
                if (year == END_YEAR and PARTIAL_END_DATE)
                else f"{year}-12-31"
            )
            print(f"  Year: {year}, Range: {start_date} to {end_date}")
            
            # Fetch daily data for rainfall, Tmax, and Tmin.
            df_rain = fetch_daily_data_for_year(lat, lng, "rainfall", start_date, end_date)
            df_tmax = fetch_daily_data_for_year(lat, lng, "temperature", start_date, end_date, aggregation="max")
            df_tmin = fetch_daily_data_for_year(lat, lng, "temperature", start_date, end_date, aggregation="min")
            
            # Rename the "Value" columns.
            df_rain = df_rain.rename(columns={"Value": "Rainfall (mm)"})
            df_tmax = df_tmax.rename(columns={"Value": "Tmax (°C)"})
            df_tmin = df_tmin.rename(columns={"Value": "Tmin (°C)"})
            
            # Merge the DataFrames on the "Date" column using an outer join.
            df_merge = pd.merge(df_rain, df_tmax, on="Date", how="outer")
            df_merge = pd.merge(df_merge, df_tmin, on="Date", how="outer")
            df_merge.sort_values("Date", inplace=True)
            
            # Compute Tmean.
            df_merge["Tmean"] = (df_merge["Tmax (°C)"] + df_merge["Tmin (°C)"]) / 2.0

            # Compute day-of-year and Ra (extraterrestrial radiation) for each date.
            df_merge["doy"] = df_merge["Date"].dt.dayofyear
            df_merge["Ra_mm"] = df_merge["doy"].apply(
                lambda doy: extraterrestrial_radiation_mm(doy, lat)
            )

            # Compute ET using the Hargreaves–Samani method.
            def compute_et(row):
                if (
                    pd.notnull(row["Tmax (°C)"])
                    and pd.notnull(row["Tmin (°C)"])
                    and pd.notnull(row["Tmean"])
                ):
                    diff = row["Tmax (°C)"] - row["Tmin (°C)"]
                    if diff < 0:
                        return None
                    return 0.0023 * (row["Tmean"] + 17.8) * np.sqrt(diff) * row["Ra_mm"]
                else:
                    return None

            df_merge["ET (mm/day)"] = df_merge.apply(compute_et, axis=1)

            # Prepare final DataFrame with desired columns.
            df_merge = df_merge.rename(columns={"Ra_mm": "Ra (mm/day)"})
            
            final_df = df_merge[[
                "Date",
                "Rainfall (mm)",
                "Tmax (°C)",
                "Tmin (°C)",
                "ET (mm/day)",
                "Ra (mm/day)"
            ]].copy()
            # Convert Date to simple string YYYY-MM-DD
            final_df["Date"] = pd.to_datetime(final_df["Date"]).dt.strftime("%Y-%m-%d")
            
            # Add columns for station details.
            final_df["Latitude"] = lat
            final_df["Longitude"] = lng
            final_df["Station"] = farm_name

            if final_df.empty:
                print(f"    -> No data found for {year}, skipping file creation.")
                continue

            # Create folder for the farm and year, then save the DataFrame as CSV.
            station_dir = os.path.join("farm_data", farm_folder_name, str(year))
            os.makedirs(station_dir, exist_ok=True)
            csv_path = os.path.join(station_dir, "daily_data.csv")
            final_df.to_csv(csv_path, index=False)
            print(f"    -> Data saved to: {csv_path}")

            # Append to all_years_data list for final concatenation later.
            all_years_data.append(final_df)

        # After all years are processed, concatenate all DataFrames for a station.
        if all_years_data:
            station_dir = os.path.join("farm_data", farm_folder_name)
            combined_df = pd.concat(all_years_data, ignore_index=True)
            combined_df.sort_values("Date", inplace=True)

            # Save the combined DataFrame to a single CSV containing all data for that station.
            combined_csv_path = os.path.join(station_dir, "all_years_data.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"--> All years combined data saved to: {combined_csv_path}")

        print("")  # Blank line after each farm


if __name__ == "__main__":
    main()
