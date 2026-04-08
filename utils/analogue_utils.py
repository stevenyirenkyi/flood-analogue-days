import xarray as xr
import pandas as pd
import utils.flood_data_loader as flood_data_loader


def calculate_dcfi(end_date):
    def norm(da: xr.DataArray):
        return (da - da.min(dim="time")) / (da.max(dim="time") - da.min(dim="time"))

    def bfill_nan(da: xr.DataArray):
        return da.bfill(dim="time", limit=1)

    drivers = flood_data_loader.load_flood_drivers(end_date=end_date)

    dac = drivers["dac"]
    sea = drivers["sea"]
    tide = drivers["tide"]
    wave = drivers["wave"]

    dac["max_rate_of_change"] = bfill_nan(dac["max_rate_of_change"])
    tide["max_rate_of_change"] = bfill_nan(tide["max_rate_of_change"])
    wave["run_up_roc"] = bfill_nan(wave["run_up_roc"])

    da_roc = (dac["max_rate_of_change"]
              + tide["max_rate_of_change"]
              + wave["run_up_roc"])
    da_mag = dac["max"] + sea["sla"] + tide["max"] + wave["run_up"]

    da_roc_norm = norm(da_roc)
    da_mag_norm = norm(da_mag)
    dcfi_da = (da_mag_norm + da_roc_norm).load()

    return dcfi_da, {"cwl_roc": da_roc,
                     "cwl": da_mag,
                     "dac": dac,
                     "tide": tide,
                     "wave": wave,
                     "sea": sea}


def calculate_daily_water_level(da: xr.DataArray, temporal_agg="max",
                                spatial_agg="mean", latlon: tuple | None = None):
    # TODO: Change function name to ecwl_timeseries and allow selection of temporal resolution
    # Possible lat values are [5. , 5.5, 6. ]
    # Possible lon values are [0.2, 0.7, 1.2]
    da = da.assign_coords(
        day_str=("time", da["time"].dt.strftime("%Y-%m-%d").data))

    if latlon:
        lat, lon = latlon
        data = da.sel(latitude=lat, longitude=lon)
    elif spatial_agg == "max":
        data = da.max(["latitude", "longitude"], skipna=True)
    elif spatial_agg == "mean":
        data = da.mean(["latitude", "longitude"], skipna=True)
    else:
        raise ValueError()

    if temporal_agg == "max":
        data = data.groupby("day_str").max(dim="time", skipna=True)
    elif temporal_agg == "mean":
        data = data.groupby("day_str").mean("time", skipna=True)

    valid = data.notnull()
    data = data.where(valid, drop=True)
    # month = pd.to_datetime(data["day_str"])

    # date = data["day_str"].values

    return pd.DataFrame({
        "water_level": data.values,
        "day_str": data["day_str"],
        "month_str": pd.to_datetime(data["day_str"].values).strftime("%Y-%m")
    })
