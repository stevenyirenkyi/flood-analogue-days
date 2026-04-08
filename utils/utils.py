import os
import xarray as xr
from xarray import DataArray
import numpy as np

from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path


load_dotenv()


def get_study_bbox():
    min_lon = 0.5
    max_lon = 1.1
    min_lat = 5.5
    max_lat = 6.1

    return min_lon, max_lon, min_lat, max_lat


def get_restricted_bbox():
    min_lon = 0.86
    max_lon = 1.08
    min_lat = 5.7

    return min_lon, max_lon, min_lat


def compute_intracell_spatial_gradient(group_ds: xr.Dataset, data_var: str) -> dict[str, xr.DataArray] | None:
    """
    Calculate spatial gradient magnitude within a coarse grid cell using a vectorized manual unstacking.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray] | None
        (mean_gradient, max_gradient) both in [data_units/degree],
        or None if calculation fails
    """
    if ('original_lat' not in group_ds.coords or 'original_lon' not in group_ds.coords or
            'stacked_latitude_longitude' not in group_ds.dims):
        return None

    try:
        # shape: (time, stacked_latitude_longitude)
        data_values = group_ds[data_var].values
        original_lats = group_ds['original_lat'].values
        original_lons = group_ds['original_lon'].values

        # 1. Determine Grid and Check Minimum Points
        unique_lats = np.unique(original_lats)
        unique_lons = np.unique(original_lons)

        if len(unique_lats) < 2 or len(unique_lons) < 2:
            return None

        # 2. Vectorize Index Mapping (Perform once outside the time loop)
        lat_map = {lat: i for i, lat in enumerate(unique_lats)}
        lon_map = {lon: i for i, lon in enumerate(unique_lons)}

        lat_indices = np.array([lat_map[lat] for lat in original_lats])
        lon_indices = np.array([lon_map[lon] for lon in original_lons])

        # Pre-allocate results for both mean and max
        mean_gradient_time_series = np.empty(len(group_ds.time))
        max_gradient_time_series = np.empty(len(group_ds.time))

        # 3. Time Loop for Reshaping and Gradient Calculation
        for time_idx in range(len(group_ds.time)):
            data_2d = np.full((len(unique_lats), len(unique_lons)), np.nan)

            # Vectorized assignment: much faster than nested loops with np.where
            data_2d[lat_indices, lon_indices] = data_values[time_idx, :]

            # 4. Create DataArray for this time step
            data_da = xr.DataArray(
                data_2d,
                dims=['fine_lat', 'fine_lon'],
                coords={'fine_lat': unique_lats, 'fine_lon': unique_lons}
            )

            # 5. Calculate spatial gradients
            lat_deriv = data_da.differentiate('fine_lat')
            lon_deriv = data_da.differentiate('fine_lon')

            # Compute gradient magnitude
            grad_mag = np.sqrt(lat_deriv**2 + lon_deriv**2)

            # Store both mean and max for this time step
            mean_gradient_time_series[time_idx] = grad_mag.mean().values
            max_gradient_time_series[time_idx] = grad_mag.max().values

        # 6. Create the final output DataArrays
        mean_gradient = xr.DataArray(
            mean_gradient_time_series,
            dims=['time'],
            coords={'time': group_ds.time},
        )

        max_gradient = xr.DataArray(
            max_gradient_time_series,
            dims=['time'],
            coords={'time': group_ds.time},
        )

        return {"mean": mean_gradient,
                "max": max_gradient}

    except Exception as e:
        print(f"Intracell spatial gradient calculation failed: {e}")
        return None


def compute_rate_of_change(da: xr.DataArray, time_coord: str = "time") -> xr.DataArray:
    """
    Compute the rate of change of a DataArray along the given time coordinate.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time dimension.
    time_coord : str, optional
        Name of the time dimension (default: "time").

    Returns
    -------
    xr.DataArray
        Rate of change with respect to time (same dimensions as input, except time is shorter by 1).
    """
    dt = da[time_coord].diff(time_coord) / np.timedelta64(1, "h")
    dy = da.diff(dim=time_coord)                                   # Δy
    rate_of_change = dy / dt.broadcast_like(dy)                    # dy/dt
    return rate_of_change.assign_coords({time_coord: da[time_coord][1:]})


def compute_spatial_aggregates(ds: xr.Dataset, data_var: str) -> xr.Dataset:
    mean = ds[data_var].mean(dim="stacked_latitude_longitude")
    max_ = ds[data_var].max(dim="stacked_latitude_longitude")
    min_ = ds[data_var].min(dim="stacked_latitude_longitude")
    range_ = max_ - min_

    rate_of_change = compute_rate_of_change(ds[data_var])

    mean_rate_of_change = rate_of_change.mean(dim="stacked_latitude_longitude")
    max_rate_of_change = rate_of_change.max(dim="stacked_latitude_longitude")

    rate_of_avg_change_dt = mean["time"].diff("time") / np.timedelta64(1, "h")
    rate_of_avg_change = mean.diff(dim="time") / rate_of_avg_change_dt
    rate_of_avg_change = rate_of_avg_change.assign_coords(time=mean.time[1:])

    spatial_gradient = compute_intracell_spatial_gradient(ds, data_var)

    output_data = {
        "mean": mean,
        "max": max_,
        "min": min_,
        "range": range_,
        "mean_rate_of_change": mean_rate_of_change,
        "max_rate_of_change": max_rate_of_change,
        "rate_of_avg_change": rate_of_avg_change,
    }
    if spatial_gradient is not None:
        output_data["spatial_gradient_max"] = spatial_gradient["max"]
        output_data["spatial_gradient_mean"] = spatial_gradient["mean"]

    return xr.Dataset(output_data)


def process_yearly_spatial_stats(ds: xr.Dataset, path: Path, data_var):
    for year in tqdm(range(1992, 2026), desc="Processing years"):
        _ds = ds.sel(time=ds["time"].dt.year == year)
        aggregated_list = []

        for (latitude, longitude), group in _ds.groupby(["latitude", "longitude"]):
            _agg = compute_spatial_aggregates(group, data_var)
            _agg = _agg.assign_coords(latitude=latitude, longitude=longitude)
            aggregated_list.append(_agg)

        agg: xr.Dataset = xr.concat(aggregated_list, dim="points")
        agg = agg.set_coords(["latitude", "longitude"])
        agg = agg.set_index(points=["latitude", "longitude"]).unstack("points")
        agg = agg.sortby("time")

        agg.to_netcdf(f"{path}/{year}.nc")


def get_all_files(directory_path: Path):
    if not directory_path.exists():
        return []
    if not directory_path.is_dir():
        return []

    return [f.name for f in directory_path.iterdir()
            if f.is_file()]


def assign_to_nearest_grid(values, grid, tol=0.25):
    values = np.array(values)
    grid = np.array(grid)
    assigned = np.full_like(values, np.nan, dtype=float)

    for i, val in enumerate(values):
        diffs = val - grid

        mask = (diffs >= -tol) & (diffs < tol)
        if np.any(mask):
            nearest_idx = np.argmin(np.abs(diffs[mask]))
            assigned[i] = grid[mask][nearest_idx]

    return assigned


def assign_nearest_era5_grid(ds: xr.Dataset, era5_lats, era5_lons):
    input_lats = ds["latitude"].values
    input_lons = ds["longitude"].values

    lat_bin = assign_to_nearest_grid(input_lats, era5_lats)
    lon_bin = assign_to_nearest_grid(input_lons, era5_lons)

    ds = ds.assign_coords(
        latitude=("latitude", lat_bin),
        longitude=("longitude", lon_bin),
        original_lat=("latitude", input_lats),
        original_lon=("longitude", input_lons)
    )

    ds = ds.dropna(dim="latitude", subset=["latitude"])
    ds = ds.dropna(dim="longitude", subset=["longitude"])

    return ds