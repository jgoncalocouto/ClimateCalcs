# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 13:41:51 2025

@author: COO1AV
"""

# epw_ladybug_export.py
# Usage:
#   python epw_ladybug_export.py /path/to/file.epw --outdir ./out
#
# What it does (Ladybug-first):
# - Uses ladybug.epw.EPW to parse everything it can (location, datetimes, series)
# - Exports:
#     * headers_ladybug.json (rich location + raw header if available)
#     * timeseries.csv and timeseries.parquet (timestamp-indexed)
#     * stats.json (basic per-column stats)
# - If Ladybug isn't installed, falls back to a robust pure-Python parser (same outputs).

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from plotly.colors import qualitative
import pandas as pd

# ----------------- Ladybug-first loader -----------------
def parse_with_ladybug(epw_path: str):
    try:
        from ladybug.epw import EPW  # type: ignore
    except Exception as e:
        return None, str(e)

    epw = EPW(epw_path)

    # datetimes and data series
    times = [
        datetime(datetime.now().year, dt.month, dt.day, (dt.hour % 24))  # turn 24 into 0
        for dt in epw.dry_bulb_temperature.datetimes
    ]

    # Collect all possible EPW fields present on the object
    series_map = {}
    # Known field names in ladybug.epw (where present)
    field_names = [
        'dry_bulb_temperature', 'dew_point_temperature', 'relative_humidity',
        'atmospheric_station_pressure', 'extraterrestrial_horizontal_radiation',
        'extraterrestrial_direct_normal_radiation', 'horizontal_infrared_radiation_intensity',
        'global_horizontal_radiation', 'direct_normal_radiation', 'diffuse_horizontal_radiation',
        'global_horizontal_illuminance', 'direct_normal_illuminance', 'diffuse_horizontal_illuminance',
        'zenith_luminance', 'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
        'visibility', 'ceiling_height', 'present_weather_observation', 'present_weather_codes',
        'precipitable_water', 'aerosol_optical_depth', 'snow_depth', 'days_since_last_snow',
        'albedo', 'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ]
    for name in field_names:
        if hasattr(epw, name):
            series = getattr(epw, name)
            # Many LB series are arrays of numbers; coerce to list
            try:
                series_map[name] = list(series)
            except Exception:
                try:
                    series_map[name] = [float(x) if x is not None else None for x in series]
                except Exception:
                    pass

    # Build DataFrame
    df = pd.DataFrame(series_map, index=pd.to_datetime(times))
    df.index.name = "timestamp"

    # Location object
    loc = epw.location
    location_dict = {}
    for attr in dir(loc):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(loc, attr)
            # keep basic types only
            if isinstance(val, (str, int, float, type(None))):
                location_dict[attr] = val
        except Exception:
            pass
        
    # Simple stats
    df_stats = dataframe_stats(df)
    return df,location_dict,df_stats


def num(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)
    
def fmt(v, unit=""):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:,.2f}{unit}" if unit else f"{v:,.2f}"
    except Exception:
        return "—"
    
def label_from_location(location_dict, fallback):
    if not location_dict:
        return fallback
    city = str(location_dict.get("city","")).strip()
    country = str(location_dict.get("country","")).strip()
    station = str(location_dict.get("station_name","")).strip() or str(location_dict.get("station","")).strip()
    if city and country:
        return f"{city}, {country}"
    if station:
        return station
    return fallback

def parse_one(epw_path: Path):
    df, location_dict, df_stats = parse_with_ladybug(str(epw_path))
    if df is None or df.empty:
        raise ValueError(f"Parsed dataframe is empty for: {epw_path.name}")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df, location_dict, df_stats

def make_color_map(labels):
    palette = qualitative.D3 + qualitative.Plotly + qualitative.Set3
    return {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}

def daily_stats(series):
    g = pd.DataFrame({"val": num(series)}); g["day"] = g.index.day
    return g.groupby("day")["val"].agg(["mean","min","max"]).reset_index()

def monthly_stats(series):
    g = pd.DataFrame({"val": num(series)}); g["month"] = g.index.month
    return g.groupby("month")["val"].agg(["mean","min","max"]).reset_index()

def diurnal_stats(series):
    g = pd.DataFrame({"val": num(series)}); g["hour"] = g.index.hour
    return g.groupby("hour")["val"].agg(["mean","min","max"]).reset_index()

def degree_days(series, base_c, kind="HDD"):
    t = num(series)
    if kind == "HDD":
        dd_hourly = np.clip(base_c - t, 0, None)
    else:
        dd_hourly = np.clip(t - base_c, 0, None)
    dd_daily = dd_hourly.resample("D").sum() / 24.0
    return dd_daily

def comfort_mask_fixed(temp_series, rh_series, tmin, tmax, rhmin=None, rhmax=None):
    t = num(temp_series)
    ok_t = (t >= tmin) & (t <= tmax)
    if rh_series is None or rhmin is None or rhmax is None:
        return ok_t
    rh = num(rh_series)
    ok_rh = (rh >= rhmin) & (rh <= rhmax)
    return ok_t & ok_rh

# ---- ASHRAE 55 Adaptive ----
def prevailing_mean_outdoor(daily_mean_outdoor, alpha=0.8):
    daily = pd.Series(daily_mean_outdoor).copy()
    daily.index = pd.to_datetime(daily.index)
    dm_shift = daily.shift(1)
    alpha_ewm = 1 - float(alpha)
    rm = dm_shift.ewm(alpha=alpha_ewm, adjust=False).mean()
    return rm

def adaptive_band_from_outdoor(daily_mean_outdoor, alpha=0.8, acceptability="80"):
    T_rm = prevailing_mean_outdoor(daily_mean_outdoor, alpha=alpha)
    t_comf = 17.8 + 0.31 * T_rm
    delta = 2.5 if str(acceptability) == "90" else 3.5
    band = pd.DataFrame({"t_comf": t_comf, "lo": t_comf - delta, "hi": t_comf + delta})
    return band

def wind_rose_speed_binned(df, ws_col, wd_col, speed_bins, n_sectors=16):
    if ws_col not in df.columns or wd_col not in df.columns:
        return pd.DataFrame(columns=["sector","speed_bin","count"])
    wr = df[[ws_col, wd_col]].dropna().copy()
    if wr.empty: return pd.DataFrame(columns=["sector","speed_bin","count"])
    wr.columns = ["ws", "wd"]
    wr["wd"] = (pd.to_numeric(wr["wd"], errors="coerce") % 360).round()
    wr["ws"] = pd.to_numeric(wr["ws"], errors="coerce")
    wr = wr.dropna(subset=["wd","ws"])
    sector_size = 360 / n_sectors
    wr["sector"] = (wr["wd"] // sector_size) * sector_size
    labels = [f"{speed_bins[i]}–{speed_bins[i+1]}" for i in range(len(speed_bins)-1)]
    wr["speed_bin"] = pd.cut(wr["ws"], bins=speed_bins, labels=labels, include_lowest=True, right=False)
    rose = wr.groupby(["sector","speed_bin"]).size().reset_index(name="count")
    return rose

def hour_of_day_heatmap(series, bins):
    vals = num(series).dropna()
    dfh = pd.DataFrame({"val": vals})
    dfh["hour"] = dfh.index.hour
    dfh["bin"] = pd.cut(dfh["val"], bins=bins, include_lowest=True)
    pivot = dfh.pivot_table(index="hour", columns="bin", values="val", aggfunc="count").fillna(0)
    return pivot

def adaptive_mask_hourly(temp_series, daily_band_df):
    hourly = pd.Series(num(temp_series), index=pd.to_datetime(temp_series.index))
    days = hourly.index.normalize()
    lo_map = daily_band_df["lo"]; hi_map = daily_band_df["hi"]
    lo_hourly = pd.Series(lo_map.reindex(days).values, index=hourly.index)
    hi_hourly = pd.Series(hi_map.reindex(days).values, index=hourly.index)
    return (hourly >= lo_hourly) & (hourly <= hi_hourly), lo_hourly, hi_hourly

def dataframe_stats(df):
    # Simple stats
    df_stats = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        df_stats[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()) if s.count() else None,
            "std": float(s.std()) if s.count() else None,
            "min": float(s.min()) if s.count() else None,
            "p50": float(s.quantile(0.5)) if s.count() else None,
            "max": float(s.max()) if s.count() else None,
        }
        return df_stats