# Shared configuration for ET and rainfall models

STATION_FOLDERS = [
    "Kahuku_Farm",
    "Nozawa_Farms",
    "Kuilima_Farms",
    "Cabaero_Farms",
    "Kupaa_Farms",
    "MAO_Organic_Farms_(new_site)",
    "MAO_Organic_Farms_(original_site)",
    "2K_Farm_LLC",
    "Wong_Hon_Hin_Inc",
    "Hawaii_Taro_Farm_LLC",
    "Hawaii_Seed_Pro_LLC_Farm",
    "Cabaero_Farm",
    "Kupaa_Farms2",
    "Hirako_Farm",
    "Hirako_Farm1",
    "Anoano_Farms",
]

TRAIN_PER_FARM = True
MODELS_DIR = "models"
BASE_DIR = "farm_data"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

START_YEAR = 2000
END_YEAR = 2025

WINDOW_SIZE = 24
HORIZON = 3

MAX_WORKERS = 8
RANDOM_SEED = 42
