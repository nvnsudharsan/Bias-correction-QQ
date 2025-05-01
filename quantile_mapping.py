import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import gamma
import os

def define_region_seasons(lat_range=None, region=None):
    if region == "India" or (lat_range and 8 <= np.mean(lat_range) <= 35):
        return {
            "Pre-Monsoon": [3, 4, 5],
            "Monsoon": [6, 7, 8, 9],
            "Post-Monsoon": [10, 11],
            "Winter": [12, 1, 2]
        }
    elif region == "Australia" or (lat_range and -40 <= np.mean(lat_range) <= -10):
        return {
            "Summer": [12, 1, 2],
            "Autumn": [3, 4, 5],
            "Winter": [6, 7, 8],
            "Spring": [9, 10, 11]
        }
    elif region == "US" or (lat_range and 25 <= np.mean(lat_range) <= 50):
        return {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Fall": [9, 10, 11]
        }
    else:
        return {
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "JJAS": [6, 7, 8, 9],
            "ON": [10, 11]
        }

def load_and_preprocess(path_hist, path_fut, path_obs, varname_hist, varname_obs, varname_fut,
                         lat_slice, lon_slice, new_lat, new_lon, period_obs, period_hist, kelvin_offset=273.15):
    hist = xr.open_dataset(path_hist)[varname_hist]
    fut = xr.open_dataset(path_fut)[varname_fut]
    obs = xr.open_dataset(path_obs)[varname_obs]

    obs = obs.sel(time=~((obs.time.dt.month == 2) & (obs.time.dt.day == 29)))
    hist = hist.sel(time=~((hist.time.dt.month == 2) & (hist.time.dt.day == 29)))
    fut = fut.sel(time=~((fut.time.dt.month == 2) & (fut.time.dt.day == 29)))

    obs = obs.sel(time=slice(*period_obs))
    hist = hist.sel(time=slice(*period_hist))

    hist = hist[:, lat_slice[0]:lat_slice[1], lon_slice[0]:lon_slice[1]].interp(lat=new_lat, lon=new_lon)
    fut = fut[:, lat_slice[0]:lat_slice[1], lon_slice[0]:lon_slice[1]].interp(lat=new_lat, lon=new_lon)

    obs = obs + kelvin_offset
    obs = obs.interp(lat=new_lat, lon=new_lon)

    return hist.load(), fut.load(), obs.load()

def separate_seasons(data, lat_range=None, region=None):
    season_def = define_region_seasons(lat_range=lat_range, region=region)
    return {name: data.sel(time=np.in1d(data['time.month'], months)) for name, months in season_def.items()}

def fit_gamma_cdf(data):
    try:
        params = gamma.fit(data)
        cdf = gamma.cdf(data, *params)
        cdf = np.clip(cdf, 0, 0.9999)
        return params, cdf
    except:
        return None, None

def apply_cdf_matching(hist_season, fut_season, obs_season, lat_dim, lon_dim, time_dim):
    corrected_hist = xr.full_like(hist_season, fill_value=np.nan)
    corrected_fut = xr.full_like(fut_season, fill_value=np.nan)

    for i in range(lat_dim):
        for j in range(lon_dim):
            if np.isnan(obs_season[:, i, j]).all():
                continue
            hist_params, hist_cdf = fit_gamma_cdf(hist_season[:, i, j])
            obs_params, _ = fit_gamma_cdf(obs_season[:, i, j])
            fut_params, fut_cdf = fit_gamma_cdf(fut_season[:, i, j])

            if None in (hist_params, obs_params, fut_params, hist_cdf, fut_cdf):
                continue

            corrected_hist[:, i, j] = gamma.ppf(hist_cdf, *obs_params)
            shift = fut_season[:, i, j] - gamma.ppf(fut_cdf, *hist_params)
            corrected_fut[:, i, j] = gamma.ppf(fut_cdf, *obs_params) + shift

        print(f"Completed latitude index: {i}")
    return corrected_hist, corrected_fut

def run_bias_correction(hist, fut, obs, save_path=None, lat_range=None, region=None):
    hist_seasons = separate_seasons(hist, lat_range=lat_range, region=region)
    fut_seasons = separate_seasons(fut, lat_range=lat_range, region=region)
    obs_seasons = separate_seasons(obs, lat_range=lat_range, region=region)

    corrected_hist_all = []
    corrected_fut_all = []

    for season in hist_seasons:
        print(f"Processing {season}...")
        hist_corr, fut_corr = apply_cdf_matching(
            hist_seasons[season], fut_seasons[season], obs_seasons[season],
            lat_dim=hist.sizes['lat'], lon_dim=hist.sizes['lon'], time_dim=hist_seasons[season].sizes['time']
        )
        corrected_hist_all.append(hist_corr)
        corrected_fut_all.append(fut_corr)

    hist_corrected = xr.concat(corrected_hist_all, dim='time')
    fut_corrected = xr.concat(corrected_fut_all, dim='time')

    if save_path:
        hist_corrected.to_netcdf(os.path.join(save_path, "hist_corrected.nc"))
        fut_corrected.to_netcdf(os.path.join(save_path, "fut_corrected.nc"))

    return hist_corrected, fut_corrected
