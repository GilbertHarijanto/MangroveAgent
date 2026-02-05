from typing import Literal
import logging
import time
import os
import re
import datetime as dt
import requests
import pandas as pd
import ee

from datetime import datetime, timedelta
from typing import Optional, Literal
from langchain_openai import ChatOpenAI
try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit optional for non-UI usage
    st = None


# --- Logging ---
logger = logging.getLogger("mangrove_agent")


def log_event(level: int, message: str, **fields) -> None:
    context = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.log(level, f"{message} | {context}" if context else message)


def get_request_id(state: dict | None) -> str:
    if not state:
        return "unknown"
    return state.get("request_id") or "unknown"


def cache_data(ttl: int):
    def decorator(func):
        if st is None:
            return func
        return st.cache_data(ttl=ttl, show_spinner=False)(func)
    return decorator


@cache_data(ttl=7 * 24 * 60 * 60)
def geocode_location(location_name: str):
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json"
    r = requests.get(url, headers={"User-Agent": "LangGraph-Mangrove-Agent"})
    data = r.json()
    if not data:
        raise ValueError(f"Could not geocode location: {location_name}")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return (lat, lon)


def resolve_gps_from_location_node(state: dict) -> dict:
    location = state.get("location")
    state_abbr = state.get("state")

    log_event(
        logging.INFO,
        "resolve_gps_start",
        request_id=get_request_id(state),
        location=location,
        state=state_abbr,
    )

    if not location:
        log_event(
            logging.WARNING,
            "resolve_gps_missing_location",
            request_id=get_request_id(state),
        )
        return {"gps": None}

    try:
        query = location if not state_abbr else f"{location}, {state_abbr}"
        lat, lon = geocode_location(query)
        log_event(
            logging.INFO,
            "resolve_gps_success",
            request_id=get_request_id(state),
            lat=lat,
            lon=lon,
        )
        return {"gps": (lat, lon)}
    except Exception as e:
        log_event(
            logging.ERROR,
            "resolve_gps_error",
            request_id=get_request_id(state),
            error=str(e),
        )
        return {"gps": None}


def load_stations(csv_path: str, state_abbr: str) -> list[dict]:
    df = pd.read_csv(csv_path)

    # Ensure state column exists and is properly formatted
    if "state" not in df.columns:
        raise ValueError("Missing 'state' column in station CSV.")

    if not state_abbr:  # 💡 Fix: don't attempt upper() on None
        return []

    filtered = df[df["state"].str.upper() == state_abbr.upper()]
    return filtered.to_dict(orient="records")  # for LLM ranking


def ask_llm_rank_stations(location: str, station_list: list[dict], variable: str) -> list[dict]:
    # --- LLM Setup ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
    station_names = [s['name'] for s in station_list]

    prompt = f"""
    The user is asking about: {location}
    Sensor type: {variable}

    Here are some stations in the same U.S. state:
    {station_names}

    Rank the top 5 stations that are most relevant or closest to the location. Respond with a list of names (copy them exactly from the list).
    """

    response = llm.invoke(prompt).content
    # Raw LLM Output
    log_event(
        logging.INFO,
        "ranked_stations_response",
        variable=variable,
        response=response,
    )

    # Normalize and remove list numbering (e.g., "1. Station Name")
    ranked_names = [
        re.sub(r"^\d+\.\s*", "", name.strip().lower())
        for name in response.split("\n")
        if name.strip()
    ]

    matched = []
    for ranked_name in ranked_names:
        for s in station_list:
            if ranked_name in s["name"].lower():
                matched.append(s)
                break
    return matched[:5]


@cache_data(ttl=7 * 24 * 60 * 60)
def select_station_by_location_cached(location: str, state_abbr: str) -> dict:
    wind_stations = load_stations(
        "stations/wind_speed_stations.csv", state_abbr)
    water_stations = load_stations(
        "stations/water_level_stations.csv", state_abbr)

    if not wind_stations and not water_stations:
        return {
            "station_ids": {},
            "station_candidates": {},
            "data_available": []
        }

    ranked_wind = ask_llm_rank_stations(location, wind_stations, "wind")
    ranked_water = ask_llm_rank_stations(
        location, water_stations, "water level")

    return {
        "station_ids": {
            "wind": ranked_wind[0]["station_id"],
            "water": ranked_water[0]["station_id"]
        },
        "station_candidates": {
            "wind": [s["station_id"] for s in ranked_wind],
            "water": [s["station_id"] for s in ranked_water]
        }
    }


def select_station_by_location_node(state: dict) -> dict:
    location = state.get("location")
    state_abbr = state.get("state")

    if not location or not state_abbr:
        return {
            "station_ids": {},
            "station_candidates": {},
            "data_available": []
        }

    return select_station_by_location_cached(location, state_abbr)


@cache_data(ttl=5 * 60)
def fetch_wind_speed(station_id: str) -> Optional[float]:
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "date": "latest",
        "station": station_id,
        "product": "wind",
        "units": "english",
        "time_zone": "gmt",
        "format": "json"
    }

    try:
        r = requests.get(url, params=params, timeout=5)
        if not r.ok:
            log_event(
                logging.WARNING,
                "noaa_wind_bad_response",
                station_id=station_id,
                status_code=r.status_code,
            )
            return None

        data = r.json()
        if "data" not in data or not data["data"]:
            log_event(
                logging.WARNING,
                "noaa_wind_no_data",
                station_id=station_id,
            )
            return None

        wind_speed = float(data["data"][0]["s"])
        return wind_speed
    except Exception as e:
        log_event(
            logging.ERROR,
            "noaa_wind_fetch_error",
            station_id=station_id,
            error=str(e),
        )
        return None


@cache_data(ttl=5 * 60)
def fetch_water_level(station_id: str) -> Optional[float]:
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "date": "latest",
        "station": station_id,
        "product": "water_level",
        "datum": "MLLW",
        "units": "english",
        "time_zone": "gmt",
        "format": "json"
    }

    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        if "data" in data and data["data"]:
            return float(data["data"][0]["v"])
        log_event(
            logging.WARNING,
            "noaa_water_no_data",
            station_id=station_id,
        )
        return None
    except Exception as e:
        log_event(
            logging.ERROR,
            "noaa_water_fetch_error",
            station_id=station_id,
            error=str(e),
        )
        return None


def try_fetch_with_fallback(fetch_func, candidate_ids: list[str]) -> Optional[float]:
    for station_id in candidate_ids:
        result = fetch_func(station_id)
        if result is not None:
            return result
    return None


def fetch_environmental_data_node(state: dict) -> dict:
    stations = state.get("station_ids", {})
    candidates = state.get("station_candidates", {})

    wind_ids = candidates.get(
        "wind", [stations.get("wind")]) if "wind" in stations else []
    water_ids = candidates.get(
        "water", [stations.get("water")]) if "water" in stations else []

    wind = try_fetch_with_fallback(
        fetch_wind_speed, wind_ids) if wind_ids else None
    water = try_fetch_with_fallback(
        fetch_water_level, water_ids) if water_ids else None

    return {
        "environmental_data": {
            "wind_speed": wind,
            "water_level": water
        }
    }


@cache_data(ttl=6 * 60 * 60)
def get_cleaned_weekly_ndvi_series(lat: float, lon: float, days_back: int = 70) -> list[float]:
    point = ee.Geometry.Point([lon, lat])
    study_area = point.buffer(16000)

    today = dt.date.today()
    start = ee.Date(today.strftime('%Y-%m-%d')).advance(-days_back, 'day')

    def add_ndvi(image):
        ndvi = image.normalizedDifference(
            ['sur_refl_b02', 'sur_refl_b01']).rename('NDVI')
        return image.addBands(ndvi)

    def extract_mean(image):
        mean = image.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=study_area,
            scale=500,
            maxPixels=1e9,
            bestEffort=True
        ).get('NDVI')
        return ee.Feature(None, {'ndvi': mean})

    ndvi_series = (
        ee.ImageCollection('MODIS/061/MOD09GA')
        .filterBounds(study_area)
        .filterDate(start, ee.Date(today.strftime('%Y-%m-%d')))
        .map(add_ndvi)
        .map(extract_mean)
        .filter(ee.Filter.notNull(['ndvi']))
    )

    ndvi_list = ndvi_series.aggregate_array('ndvi').getInfo()
    ndvi_floats = [v / 10000 if v > 1 else v for v in ndvi_list]

    if len(ndvi_floats) < 56:
        raise ValueError(
            f"Not enough data. Got {len(ndvi_floats)} daily values, need at least 56.")

    ndvi_floats = ndvi_floats[-56:]
    weekly_ndvi = [
        sum(ndvi_floats[i:i + 7]) / 7 for i in range(0, 56, 7)
    ]

    if len(weekly_ndvi) < 8:
        raise ValueError(
            f"Only {len(weekly_ndvi)} weekly values found, need 8.")

    return weekly_ndvi


@cache_data(ttl=60 * 60)
def fetch_weekly_noaa_lags_chunked(station_id: str, product: str, total_days: int = 56) -> list[float]:

    end = datetime.utcnow().date()
    start = end - timedelta(days=total_days)

    # Break range into 3 chunks (e.g., 18 + 19 + 19 days)
    chunk_starts = [start + timedelta(days=i * 18) for i in range(3)]
    chunk_ends = [min(s + timedelta(days=18), end) for s in chunk_starts]

    all_dfs = []

    for chunk_start, chunk_end in zip(chunk_starts, chunk_ends):
        params = {
            "begin_date": chunk_start.strftime("%Y%m%d"),
            "end_date": chunk_end.strftime("%Y%m%d"),
            "station": station_id,
            "product": product,
            "datum": "MLLW" if product == "water_level" else None,
            "interval": "h",  # Hourly granularity
            "units": "english",
            "time_zone": "gmt",
            "format": "json"
        }

        try:
            r = requests.get("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
                             params={k: v for k, v in params.items()
                                     if v is not None},
                             timeout=15)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                continue

            df = pd.DataFrame(data)
            df["t"] = pd.to_datetime(df["t"])
            df.set_index("t", inplace=True)

            value_col = "s" if product == "wind" else "v"
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

            all_dfs.append(df)

        except Exception as e:
            log_event(
                logging.WARNING,
                "noaa_chunk_failed",
                product=product,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                error=str(e),
            )

    if not all_dfs:
        log_event(
            logging.WARNING,
            "noaa_no_valid_data",
            product=product,
        )
        return []

    full_df = pd.concat(all_dfs).sort_index()

    # Resample into weekly means (week ends Sunday by default)
    value_col = "s" if product == "wind" else "v"
    weekly = full_df[value_col].resample("W").mean().dropna()

    if len(weekly) < 8:
        log_event(
            logging.WARNING,
            "noaa_insufficient_weekly_values",
            product=product,
            count=len(weekly),
            expected=8,
        )
        return weekly.tolist()  # Return whatever we have

    return weekly.tail(8).tolist()


def build_feature_vector_node(state: dict) -> dict:
    try:
        # Fetch 8 weeks (current + 7 lags)
        wind_vals = fetch_weekly_noaa_lags_chunked(
            state["station_ids"]["wind"], "wind")
        water_vals = fetch_weekly_noaa_lags_chunked(
            state["station_ids"]["water"], "water_level")
        ndvi_vals = get_cleaned_weekly_ndvi_series(*state["gps"])

        if len(wind_vals) < 8 or len(water_vals) < 8 or len(ndvi_vals) < 8:
            log_event(
                logging.WARNING,
                "feature_vector_insufficient_weekly_data",
                request_id=get_request_id(state),
            )
            return {"feature_vector": [], "feature_df": None}

        # Build DataFrame (already in chronological order: oldest → newest)
        df = pd.DataFrame({
            "tide_verified": water_vals,
            "wind_speed": wind_vals,
            "ndvi": ndvi_vals
        })

        # Add artificial weekly dates (most recent week = today)
        base_date = pd.to_datetime("today").normalize()
        df["date"] = [base_date -
                      pd.Timedelta(weeks=i) for i in reversed(range(len(df)))]
        df.set_index("date", inplace=True)

        # Add lag features (1–7)
        for col in ["tide_verified", "wind_speed", "ndvi"]:
            for lag in range(1, 8):
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Drop NaNs → only final row will be complete
        latest_row = df.dropna().iloc[-1]

        # Select 21 lag features in model training order
        # Get current values first
        features = latest_row[["tide_verified", "wind_speed"]].tolist()

        # Add lags
        features += latest_row[
            [f"{col}_lag_{i}" for col in ["tide_verified", "wind_speed", "ndvi"]
                for i in range(1, 8)]
        ].tolist()

        log_event(
            logging.INFO,
            "feature_vector_built",
            request_id=get_request_id(state),
            feature_count=len(features),
        )

        return {
            "feature_vector": features,
            "feature_df": df
        }

    except Exception as e:
        log_event(
            logging.ERROR,
            "feature_vector_error",
            request_id=get_request_id(state),
            error=str(e),
        )
        return {"feature_vector": [], "feature_df": None}


def fetch_ndvi_lags_node(state: dict) -> dict:
    gps = state.get("gps")
    if not gps:
        log_event(
            logging.WARNING,
            "ndvi_lags_missing_gps",
            request_id=get_request_id(state),
        )
        return {"ndvi_lags": []}
    lat, lon = gps
    ndvi_lags = get_cleaned_weekly_ndvi_series(lat, lon)[:7]
    log_event(
        logging.INFO,
        "ndvi_lags_complete",
        request_id=get_request_id(state),
        count=len(ndvi_lags),
    )
    return {"ndvi_lags": ndvi_lags}


def fetch_weekly_lags_node(state: dict) -> dict:
    stations = state.get("station_ids", {})
    wind = fetch_weekly_noaa_lags_chunked(stations.get("wind"), "wind")
    water = fetch_weekly_noaa_lags_chunked(
        stations.get("water"), "water_level")
    log_event(
        logging.INFO,
        "noaa_lags_complete",
        request_id=get_request_id(state),
        wind_count=len(wind),
        water_count=len(water),
    )
    return {"wind_lags": wind, "water_lags": water}


def route_by_goal(state: dict) -> Literal["forecast", "research"]:
    if state.get("goal") == "forecast":
        return "forecast"
    return "research"
