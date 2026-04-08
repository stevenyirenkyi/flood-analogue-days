import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Literal, TypedDict, Unpack, NotRequired
from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA_DIR = PROJECT_ROOT / "data"

HARMONISED_DATA_DIR = PROJECT_DATA_DIR / "harmonised_data"
FLOOD_MONTHS_CSV = PROJECT_DATA_DIR / "flood_occurence_year_month.csv"


def load_flood_drivers(end_date=None):
    TIDE_DIR = f"{HARMONISED_DATA_DIR}/tide_stats_on_era5_grid/*.nc"
    WAVE_DIR = f"{HARMONISED_DATA_DIR}/wave_run_up.nc"
    DAC_DIR = f"{HARMONISED_DATA_DIR}/dac_stats_on_era5_grid/*.nc"
    SEA_DIR = f"{HARMONISED_DATA_DIR}/sea.nc"

    tide = xr.open_mfdataset(TIDE_DIR, chunks="auto")
    wave = xr.open_dataset(WAVE_DIR, chunks="auto")
    dac = xr.open_mfdataset(DAC_DIR, chunks="auto")
    sea = xr.open_mfdataset(SEA_DIR, chunks="auto")

    components = {
        "sea": sea,
        "dac": dac,
        "wave": wave,
        "tide": tide,
    }

    end_date = end_date if end_date else "2024-12-31"

    components = {k: v.sel(longitude=[0.2, 0.7, 1.2],
                           latitude=[5.0, 5.5],
                           time=slice("1992-09-03", end_date))
                  for k, v in components.items()}

    return components


class LoadFloodMonthsArgs(TypedDict):
    subset: NotRequired[None | Literal["nadmo", "me", "lagoon",
                                       "not_proven_false"]]


def load_flood_months(subset=None) -> list | None:
    flood_months = pd.read_csv(FLOOD_MONTHS_CSV)

    if subset == "nadmo":
        data = flood_months[flood_months["source"] == "nadmo"]
    elif subset == "lagoon":
        data = flood_months[flood_months["type"] == "lagoon"]
    elif subset == "me":
        data = flood_months[flood_months["source"] == "me"]
    elif subset == "not_proven_false":
        data = flood_months[~flood_months["proven_false"].astype(bool)]
    elif subset is None:
        data = flood_months
    else:
        raise Exception("...")

    return data["month_str"].to_list()


def ensure_month_str_coord(da: xr.DataArray):
    if "month_str" not in da.coords:
        return da.assign_coords(month_str=("time", da["time"].dt.strftime("%Y-%m").data))

    return da


@lru_cache(maxsize=None)
def load_flood_months_cached(**kwargs):
    frozen_kwargs = tuple(sorted((k, str(v)) for k, v, in kwargs.items()))
    return load_flood_months(**dict(frozen_kwargs))
