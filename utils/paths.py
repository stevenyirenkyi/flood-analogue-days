import os

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_env_var(var_name: str):
    ENV_VAR = os.environ.get(var_name)
    if not ENV_VAR:
        raise Exception(f"{var_name} environment variable not set")

    return ENV_VAR


BASE_DATA_DIR = Path(get_env_var("BASE_DATA_DIR"))

TEMP_CLIMATE_DATA_DIR = BASE_DATA_DIR / "temp"

GLOBAL_DATA_DIR = BASE_DATA_DIR / "level_0"

PRECIPITATION_DIR = BASE_DATA_DIR / "level_1" / "precipitation"
WAVE_DIR = BASE_DATA_DIR / "level_1" / "wave"
DAC_DIR = BASE_DATA_DIR / "level_1" / "dac"
GENERATED_TIDES_DIR = BASE_DATA_DIR / "level_1" / "tide"
SLA_DIR = BASE_DATA_DIR / "level_1" / "sla"
WIND_DIR = BASE_DATA_DIR / "level_1" / "wind"








PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA_DIR = PROJECT_ROOT / "data"
HARMONISED_DATA_DIR = Path(PROJECT_DATA_DIR / "harmonised_data")



# --- FILES ---

FLOOD_MONTHS_CSV = PROJECT_DATA_DIR / "flood_occurence_year_month.csv"
