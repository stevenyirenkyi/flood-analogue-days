import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray as rxr
import matplotlib.ticker as mticker
import xarray as xr

from cartopy.mpl.geoaxes import GeoAxes
from typing import cast
from global_land_mask import globe
import matplotlib.patheffects as PathEffects



STUDY_AREA_EXTENT = [-0.2, 2, 4.5, 6.1]
fig_size = (5.5, 5)
MAP_LATS, MAP_LONS = [5.5, 5], [0.2, 0.7, 1.2, 1.7]

KETA_LAT, KETA_LON = 5.917, 0.986
DZITA_LAT, DZITA_LON = 5.771, 0.773
ANLOGA_LAT, ANLOGA_LON = 5.792811, 0.899024


location_marker_size = 4

MAP_FONT_SIZES = {"LEGEND": 10,
                  "TITLE": 11}


def add_surface_types(ax: GeoAxes, exclude=[]):
    default_features = {
        "land":     (cfeature.LAND, {"facecolor": "#FBF6E2"}),
        "rivers":   (cfeature.RIVERS, {"facecolor": "#B0EAFC"}),
        "ocean":    (cfeature.OCEAN, {"facecolor": "#B0EAFC"}),
        "borders":  (cfeature.BORDERS, {"edgecolor": "#DBC8A5"}),
        "coastline": (cfeature.COASTLINE, {"edgecolor": "#DBC8A5"}),
    }

    for name, (feature, kwargs) in default_features.items():
        if name not in exclude:
            ax.add_feature(feature, **kwargs)


def add_towns(ax: GeoAxes, offset= 1.0, projection=ccrs.PlateCarree()):
    text_lon_shift = -0.1 * offset
    text_lat_shift = 0.03 * offset

    plot_params = {"marker": "o",
                   "markersize": location_marker_size,
                   "transform": projection}
    text_params = {"transform": projection,
                   "fontsize": 10}


    # ax.plot(KETA_LON, KETA_LAT, **plot_params)
    t = ax.text(KETA_LON + text_lon_shift,
            KETA_LAT + text_lat_shift,
            'Keta',
            **text_params)
    add_halos(t)

    # ax.plot(DZITA_LON, DZITA_LAT, **plot_params)
    t = ax.text(ANLOGA_LON + text_lon_shift,
            ANLOGA_LAT + text_lat_shift,
            'Anloga',
            **text_params)
    add_halos(t)

def add_halos(text, linewidth=2.5):
    text.set_path_effects(
        [PathEffects.withStroke(linewidth=linewidth, foreground="white")])

def add_halo_to_legend(legend):
    for text in legend.get_texts():
        text.set_path_effects([
            PathEffects.withStroke(linewidth=2.5, foreground="white")
        ])

def format_gridlines_and_spines(ax: GeoAxes, lats=None, lons=None, linewidth=0.0):
    gl = ax.gridlines(draw_labels=True, linewidth=linewidth)
    gl.top_labels = gl.right_labels = False

    if lons is None:
        gl.xlocator = mticker.FixedLocator(MAP_LONS)
    else:
        gl.xlocator = mticker.FixedLocator(lons)

    if lats is None:
        gl.ylocator = mticker.FixedLocator(MAP_LATS)
    else:
        gl.ylocator = mticker.FixedLocator(lats)


    for spine in ax.spines.values():
        spine.set_visible(False)

# def format_legend()


def make_map(extent: list,
             lats: list | None = None,
             lons: list | None = None,
             figsize=(10, 5),
             gridline_width=0.0,
             show_towns=True):
    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    ax = cast(GeoAxes, ax)
    ax.set_extent(extent, crs=projection)

    add_surface_types(ax)
    format_gridlines_and_spines(ax, lats, lons,
                                gridline_width)
    if show_towns:
        add_towns(ax)

    return fig, ax, projection


def split_surface_type(ds: xr.Dataset, short_coords=False):
    elevation = rxr.open_rasterio("../data/elevation.tiff", masked=True)
    elevation = elevation.squeeze().drop_vars(  # type:ignore
        ["spatial_ref", "band"])
    elevation = elevation.rename({"x": "longitude", "y": "latitude"})

    # not clipping ds will leave NaNs at edges
    min_lat = float(elevation["latitude"].min())
    max_lat = float(elevation["latitude"].max())
    min_lon = float(elevation["longitude"].min())
    max_lon = float(elevation["longitude"].max())

    clipped_ds = ds.copy(deep=True)
    if short_coords:
        clipped_ds = clipped_ds.rename({"lat": "latitude", "lon": "longitude"})

    clipped_ds = clipped_ds.sel(
        latitude=slice(min_lat, max_lat),
        longitude=slice(min_lon, max_lon),
    ).copy(deep=True)

    elevation_interp = elevation.interp(
        latitude=clipped_ds["latitude"],
        longitude=clipped_ds["longitude"],
        method="linear",
    )

    # land mask from global_land_mask
    lon_grid, lat_grid = np.meshgrid(
        clipped_ds["longitude"], clipped_ds["latitude"])
    is_land = globe.is_land(lat_grid, lon_grid)

    # water mask from EOTPO1 elevation
    is_water = elevation_interp <= 0

    # category definition
    sea_mask = ~is_land
    land_mask = is_land & ~is_water
    lagoon_mask = is_land & is_water

    classes = np.full_like(is_land, fill_value=0, dtype=np.uint8)
    classes[land_mask] = 1
    classes[lagoon_mask] = 2

    surface_mask = xr.DataArray(
        classes,
        coords={
            "latitude": clipped_ds["latitude"],
            "longitude": clipped_ds["longitude"],
        },
        dims=["latitude", "longitude"],
        name="surface_mask",
    )

    sea_data = clipped_ds.where(surface_mask == 0)
    lagoon_data = clipped_ds.where(surface_mask == 2)
    land_data = clipped_ds.where(surface_mask == 1)

    return sea_data, lagoon_data, land_data


def set_title(title: str):
    plt.title(title, fontsize=11)
