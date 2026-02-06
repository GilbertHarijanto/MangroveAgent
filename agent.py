from typing import Literal
import logging
import os
import re
import time
import uuid
import datetime as dt
import requests
import pandas as pd
import joblib
import ee
from threading import Lock

from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Literal
from typing import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_pinecone import Pinecone
from langgraph.graph import StateGraph, END

from nodes import (
    resolve_gps_from_location_node,
    select_station_by_location_node,
    fetch_weekly_lags_node,
    fetch_ndvi_lags_node,
    fetch_environmental_data_node,
    build_feature_vector_node,
    route_by_goal
)


# --- Logging ---
logger = logging.getLogger("mangrove_agent")
if not logger.handlers:
    log_level = os.getenv("MANGROVE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


def log_event(level: int, message: str, **fields) -> None:
    """Log a structured event with key-value context to the mangrove_agent logger."""
    context = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.log(level, f"{message} | {context}" if context else message)


def get_request_id(state: Optional[dict]) -> str:
    """Extract request_id from graph state; return 'unknown' if missing or empty."""
    if not state:
        return "unknown"
    return state.get("request_id") or "unknown"


def timed_node(node_name: str, fn):
    """Wrap a LangGraph node with timing and structured error logging."""

    def _inner(state: dict) -> dict:
        start = time.perf_counter()
        request_id = get_request_id(state)
        try:
            result = fn(state)
            duration_ms = int((time.perf_counter() - start) * 1000)
            log_event(
                logging.INFO,
                "node_complete",
                request_id=request_id,
                node=node_name,
                latency_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            log_event(
                logging.ERROR,
                "node_error",
                request_id=request_id,
                node=node_name,
                latency_ms=duration_ms,
                error=str(e),
            )
            raise

    return _inner


def cache_data(ttl: int, maxsize: int = 1024):
    """
    Wrap a function with an in-memory TTL cache (FastAPI/CLI compatible).
    Uses cachetools.TTLCache and a lock for thread-safe access.
    """

    def decorator(func):
        cache = TTLCache(maxsize=maxsize, ttl=ttl)
        lock = Lock()
        return cached(cache=cache, key=hashkey, lock=lock)(func)

    return decorator


# --- API Keys and Auth ---
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_INDEX_NAME"] = "mangrove-index"

# Earth Engine setup
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    # Replace with your GCP EE project
    ee.Initialize(project='ee-lgharijanto123')

# --- Pydantic Intent Schema ---
class QueryIntent(BaseModel):
    goal: Literal["forecast", "research"] = Field(
        description="""
        'forecast': the user wants a contextual overview or analysis. May include NDVI forecasts and real-time data if a location is mentioned.
        'research': the user is asking a general knowledge question not tied to a specific place or data fetch.
        """
    )
    include_plot: bool = Field(
        default=False,
        description="""
        True ONLY when the user explicitly asks for a plot/chart/graph/visualization (e.g. 'plot', 'chart', 'graph', 'visualize').
        False for 'compare', 'vs', 'between', or general questions. Default False when unsure.
        """
    )
    plot_metric: Optional[Literal["ndvi", "wind_speed", "water_level", "all"]] = Field(
        default=None,
        description="""
        The metric the user wants plotted. Use:
        - 'ndvi' for NDVI requests
        - 'wind_speed' for wind speed requests
        - 'water_level' for water level/tide requests
        - 'all' if they ask for all metrics
        Return None if not requested.
        """
    )
    location: Optional[str] = Field(
        description="The geographic location specified in the query, if any. Return None if not found."
    )
    state: Optional[str] = Field(
        description="""
    The 2-letter U.S. state abbreviation (e.g., 'FL', 'TX') associated with the location,
    if known or inferable from the query. Helps scope station lookups regionally.
    Return None if the query doesn't reference a U.S. state.
    """
    )

# --- LLM Setup (initialized lazily after API key is available) ---
llm: Optional[ChatOpenAI] = None
structured_llm: Optional[Runnable] = None
INTENT_MODEL = os.getenv("MANGROVE_INTENT_MODEL", "gpt-4o-mini")


system_prompt = SystemMessage(content="""
You are an intelligent assistant that extracts user intent from natural language.

Return:
- goal:
    - 'forecast' if the user is asking for an overview, condition, or analysis of mangroves, especially in a specific location.
    - 'research' if the user asks general questions like "What are mangroves?" or "Why are they important?"

- location:
    - If the user specifies a geographic location (e.g., city, region, island, or landmark), extract it.
    - Return None if no clear location is mentioned.

- state:
    - If the location is in the United States, extract the 2-letter state abbreviation (e.g., "FL" for Florida, "LA" for Louisiana).
    - Return None if the location is outside the U.S. or cannot be inferred.

- include_plot:
    - True ONLY if the user uses explicit visualization words: "plot", "chart", "graph", "visualize", "visualization", "show me a chart/graph", "draw", "display a graph".
    - False for "compare", "vs", "between", or general analysis questions. Default to False when unsure.

- plot_metric:
    - 'ndvi' for NDVI requests
    - 'wind_speed' for wind speed requests
    - 'water_level' for water level/tide requests
    - 'all' if they ask for all metrics
    - Return None if not requested

Only use the allowed values for 'goal'. Do not guess location if it's unclear.
""")


class State(TypedDict):
    # === User input & intent extraction ===
    user_query: str                             # Always provided by the user
    goal: Optional[str]                         # 'forecast' or 'research'
    include_plot: Optional[bool]
    plot_metric: Optional[str]
    location: Optional[str]                     # e.g. "Key West"
    state: Optional[str]                        # e.g. "FL"
    request_id: Optional[str]                   # request/session id

    # === Location resolution ===
    gps: Optional[Tuple[float, float]]          # (lat, lon)

    # === Station selection ===
    station_ids: Optional[Dict[str, str]]       # {wind: id, water: id}
    station_candidates: Optional[Dict[str, List[str]]]  # fallback ids

    # === Real-time fetches
    # {'wind_speed': val, 'water_level': val}
    environmental_data: Optional[Dict[str, float]]

    # === Weekly history (optional: for inspection only)
    wind_lags: Optional[List[float]]
    water_lags: Optional[List[float]]
    ndvi_lags: Optional[List[float]]

    # === Feature engineering ===
    feature_vector: Optional[List[float]]       # Final model input
    feature_df: Optional["pd.DataFrame"]        # Optional for debugging

    # === Model output ===
    ndvi_prediction: Optional[float]
    summary: Optional[str]
    final_output: Optional[str]


# --- XGBoost and Scaler Model ---
xgb_model = joblib.load("models/xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

# --- Memory and Retriever ---
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
embeddings: Optional[OpenAIEmbeddings] = None
retriever = None
research_chain = None


def get_research_chain(llm: ChatOpenAI):
    """Build or return the cached RAG chain (Pinecone retriever + conversational chain)."""
    global embeddings, retriever, research_chain
    if research_chain is None:
        embeddings = OpenAIEmbeddings()
        db = Pinecone.from_existing_index(
            index_name="mangrove-index", embedding=embeddings)
        retriever = db.as_retriever()
        research_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory)
    return research_chain


def extract_intent_node(state: dict) -> dict:
    """LangGraph node: extract goal, location, state, include_plot, plot_metric from user query via LLM."""
    user_query = state["user_query"]

    intent = structured_llm.invoke(
        [system_prompt, HumanMessage(content=user_query)]
    )

    return {
        "goal": intent.goal,
        "include_plot": intent.include_plot,
        "plot_metric": intent.plot_metric,
        "location": intent.location,
        "state": intent.state
    }


@cache_data(ttl=5 * 60)
def fetch_wind_speed(station_id: str) -> Optional[float]:
    """Fetch latest wind speed (mph) from NOAA Tides & Currents API for the given station. Cached 5 min."""
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
    """Fetch latest water level (ft, MLLW) from NOAA Tides & Currents API for the given station. Cached 5 min."""
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
    """Try candidate station IDs in order until a valid value is returned; otherwise None."""
    for station_id in candidate_ids:
        result = fetch_func(station_id)
        if result is not None:
            return result
    return None


@cache_data(ttl=6 * 60 * 60)
def get_cleaned_weekly_ndvi_series(lat: float, lon: float, days_back: int = 70) -> list[float]:
    """Fetch MODIS NDVI around (lat, lon), aggregate to 8 weekly means (last 56 days). Cached 6h."""
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
    """Fetch NOAA time series for station/product in chunks, resample to weekly means, return last 8. Cached 1h."""
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


def predict_ndvi_node(state: dict) -> dict:
    """LangGraph node: run XGBoost model on feature_vector and return ndvi_prediction."""
    try:
        features = state.get("feature_vector")
        if not features or len(features) != 23:
            log_event(
                logging.WARNING,
                "prediction_invalid_feature_vector",
                request_id=get_request_id(state),
                feature_count=0 if not features else len(features),
            )
            return {"ndvi_prediction": None}

        X = pd.DataFrame([features], columns=scaler.feature_names_in_)

        X_scaled = scaler.transform(X)

        pred = xgb_model.predict(X_scaled)[0]
        log_event(
            logging.INFO,
            "prediction_complete",
            request_id=get_request_id(state),
            ndvi_prediction=pred,
        )

        return {"ndvi_prediction": float(pred)}

    except Exception as e:
        log_event(
            logging.ERROR,
            "prediction_error",
            request_id=get_request_id(state),
            error=str(e),
        )
        return {"ndvi_prediction": None}


summary_prompt = ChatPromptTemplate.from_template("""
You are a coastal ecology assistant.

Given the following data:
- User question: {user_query}
- NDVI prediction: {ndvi_prediction}
- Wind speed: {wind_speed}
- Water level: {water_level}

Generate a clear, concise 5-6 sentence summary of the mangrove condition and any notable environmental trends.
- Always include all the numbers IF provided.
- **DO NOT MENTION MISSING OR UNAVAILABLE DATA**
- If the query depends on geographic specificity and no location was found, consider asking the user to clarify a region.
""")
summary_chain: Optional[Runnable] = None


def get_summary_chain(llm: ChatOpenAI) -> Runnable:
    """Build or return the cached summary chain (prompt | llm)."""
    global summary_chain
    if summary_chain is None:
        summary_chain = summary_prompt | llm
    return summary_chain


def generate_summary_node(state: State) -> dict:
    """LangGraph node: generate natural-language summary from NDVI, wind, water using the summary LLM."""
    try:
        env = state.get("environmental_data") or {}
        wind = env.get("wind_speed")
        water = env.get("water_level")
        ndvi = state.get("ndvi_prediction")

        def _round_metric(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            return round(float(value), 3)

        response = get_summary_chain(llm).invoke({
            "user_query": state["user_query"],
            "ndvi_prediction": _round_metric(ndvi),
            "wind_speed": _round_metric(wind),
            "water_level": _round_metric(water)
        })

        return {
            "summary": response.content
        }

    except Exception as e:
        log_event(
            logging.ERROR,
            "summary_error",
            request_id=get_request_id(state),
            error=str(e),
        )
        return {"summary": "Unable to generate summary."}


# --- Graph Setup ---
workflow = StateGraph(State)
workflow.add_node("extract_intent", timed_node(
    "extract_intent", extract_intent_node))
workflow.add_node("exit_early", timed_node("exit_early", lambda state: {
    "final_output": "[ROUTED TO RESEARCH AGENT]"}))

workflow.set_entry_point("extract_intent")
workflow.add_conditional_edges("extract_intent", route_by_goal, {
    "forecast": "select_stations",
    "research": "exit_early"
})

workflow.add_node("select_stations", timed_node(
    "select_stations", select_station_by_location_node))
workflow.add_node("resolve_gps", timed_node(
    "resolve_gps", resolve_gps_from_location_node))
workflow.add_node("fetch_environmental_data", timed_node(
    "fetch_environmental_data", fetch_environmental_data_node))
workflow.add_node("fetch_weekly_lags", timed_node(
    "fetch_weekly_lags", fetch_weekly_lags_node))

workflow.add_edge("select_stations", "resolve_gps")
workflow.add_edge("select_stations", "fetch_environmental_data")
workflow.add_edge("select_stations", "fetch_weekly_lags")

workflow.add_node("fetch_ndvi_lags", timed_node(
    "fetch_ndvi_lags", fetch_ndvi_lags_node))
workflow.add_edge("resolve_gps", "fetch_ndvi_lags")

workflow.add_node("build_feature_vector", timed_node(
    "build_feature_vector", build_feature_vector_node))
workflow.add_node("predict_ndvi", timed_node(
    "predict_ndvi", predict_ndvi_node))
workflow.add_node("generate_summary", timed_node(
    "generate_summary", generate_summary_node))

workflow.add_edge("fetch_ndvi_lags", "build_feature_vector")
workflow.add_edge("fetch_weekly_lags", "build_feature_vector")
workflow.add_edge("build_feature_vector", "predict_ndvi")
workflow.add_edge("predict_ndvi", "generate_summary")
workflow.add_edge("fetch_environmental_data", "generate_summary")
workflow.add_edge("generate_summary", END)

app = workflow.compile()


def forecast_chain(user_query: str, request_id: Optional[str] = None) -> str:
    """Run the forecast graph and return the summary text (or fallback message)."""
    state = {"user_query": user_query}
    if request_id:
        state["request_id"] = request_id
    result = app.invoke(state)
    return result.get("summary", "No forecast available.")


def _weekly_dates(count: int) -> list[str]:
    """Return a list of ISO date strings for the last `count` weeks (most recent last)."""
    if count <= 0:
        return []
    base_date = datetime.utcnow().date()
    return [
        (base_date - timedelta(weeks=(count - 1 - i))).isoformat()
        for i in range(count)
    ]


def build_graph_data(state: dict, intent: QueryIntent) -> Optional[dict]:
    """Build chart-ready dict with series (wind, water, ndvi) from state; filter by intent.plot_metric if set."""
    wind = state.get("wind_lags") or []
    water = state.get("water_lags") or []
    ndvi = state.get("ndvi_lags") or []

    if not any([wind, water, ndvi]):
        return None

    count = max(len(wind), len(water), len(ndvi))
    dates = _weekly_dates(count)

    def to_points(values: list[float]) -> list[dict]:
        if not values:
            return []
        aligned_dates = dates[-len(values):]
        return [
            {"t": date, "v": round(float(value), 3)}
            for date, value in zip(aligned_dates, values)
        ]

    series = [
        {"name": "wind_speed", "unit": "mph", "points": to_points(wind)},
        {"name": "water_level", "unit": "ft", "points": to_points(water)},
        {"name": "ndvi", "unit": "index", "points": to_points(ndvi)},
    ]

    if intent.plot_metric and intent.plot_metric != "all":
        series = [s for s in series if s["name"] == intent.plot_metric]

    return {
        "series": series,
        "location": intent.location,
        "state": intent.state,
        "station_ids": state.get("station_ids") or {},
        "gps": state.get("gps"),
    }


def run_research_chain(user_query: str) -> str:
    """Run the RAG research chain for general mangrove questions; return answer string."""
    raw_response = get_research_chain(llm).invoke({"question": user_query})
    return raw_response.get("answer") if isinstance(raw_response, dict) else str(raw_response)


def extract_goal_and_location(user_query: str):
    """Extract intent (goal) from user query via structured LLM; return intent.goal."""
    intent = structured_llm.invoke(
        [system_prompt, HumanMessage(content=user_query)]
    )
    return intent.goal


def run_agent(user_query: str):
    """Run full agent (intent -> forecast or research) and return dict with final_output."""
    request_id = str(uuid.uuid4())
    log_event(
        logging.INFO,
        "request_start",
        request_id=request_id,
        goal="auto",
    )

    goal = extract_goal_and_location(user_query)
    log_event(
        logging.INFO,
        "intent_detected",
        request_id=request_id,
        goal=goal,
    )

    if goal == "forecast":
        final_output = forecast_chain(user_query, request_id=request_id)

        memory.save_context({"input": user_query}, {"output": final_output})

    elif goal == "research":
        final_output = run_research_chain(user_query)

    else:
        log_event(
            logging.WARNING,
            "intent_unrecognized",
            request_id=request_id,
            goal=goal,
        )
        final_output = "Unrecognized goal."

    return {"final_output": final_output}


# def run_mangrove_agent(user_query: str) -> str:
#     """
#     Wrapper for Streamlit or notebook usage.
#     Returns only the final response string, or an error message.
#     """
#     response = run_agent(user_query)

#     return response["final_output"]

# --- Main Logic Refactored for Dynamic API Key ---
def build_llm(api_key: str):
    """Initialize llm and structured_llm (intent extraction) with the given API key; return llm."""
    global llm, structured_llm
    os.environ["OPENAI_API_KEY"] = api_key  # update environment
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o", temperature=0.8)
    intent_llm = ChatOpenAI(
        openai_api_key=api_key,
        model=INTENT_MODEL,
        temperature=0,
    )
    structured_llm = intent_llm.with_structured_output(QueryIntent)
    return llm


def run_mangrove_agent(user_query: str, api_key: str) -> str:
    """Public entry point: build LLM, run agent, return plain-text reply or error string."""
    try:
        llm = build_llm(api_key)
        return _run_with_llm(user_query, llm)
    except Exception as e:
        return f"Error processing query: {str(e)}"


def run_mangrove_agent_structured(user_query: str, api_key: str) -> dict:
    """Public entry point: build LLM, run agent, return dict with reply, intent, include_plot, plot_metric, graph_data."""
    try:
        llm = build_llm(api_key)
        return _run_with_llm_structured(user_query, llm)
    except Exception as e:
        return {"reply": f"Error processing query: {str(e)}", "intent": None, "graph_data": None}


def _run_with_llm(user_query: str, llm: ChatOpenAI) -> str:
    """Run intent extraction and forecast/research path; return plain-text reply."""
    request_id = str(uuid.uuid4())
    log_event(
        logging.INFO,
        "request_start",
        request_id=request_id,
        goal="auto",
    )

    intent_chain = structured_llm or llm.with_structured_output(QueryIntent)
    intent = intent_chain.invoke(
        [system_prompt, HumanMessage(content=user_query)]
    )
    log_event(
        logging.INFO,
        "intent_detected",
        request_id=request_id,
        goal=intent.goal,
        include_plot=intent.include_plot,
        plot_metric=intent.plot_metric,
        location=intent.location,
        state=intent.state,
    )

    if intent.goal == "forecast":
        result = app.invoke({
            "user_query": user_query,
            "request_id": request_id,
        })
        memory.save_context({"input": user_query}, {
                            "output": result["summary"]})
        log_event(
            logging.INFO,
            "forecast_response",
            request_id=request_id,
        )
        return result["summary"]

    elif intent.goal == "research":
        raw_response = get_research_chain(llm).invoke({"question": user_query})
        answer = raw_response.get("answer") if isinstance(
            raw_response, dict) else str(raw_response)
        log_event(
            logging.INFO,
            "research_response",
            request_id=request_id,
        )
        return answer

    else:
        log_event(
            logging.WARNING,
            "intent_unrecognized",
            request_id=request_id,
            goal=intent.goal,
        )
        return "Unrecognized goal."


def _run_with_llm_structured(user_query: str, llm: ChatOpenAI) -> dict:
    """Run intent extraction and forecast/research path; return dict with reply, intent, graph_data, etc."""
    request_id = str(uuid.uuid4())
    log_event(
        logging.INFO,
        "request_start",
        request_id=request_id,
        goal="auto",
    )

    intent_chain = structured_llm or llm.with_structured_output(QueryIntent)
    intent = intent_chain.invoke(
        [system_prompt, HumanMessage(content=user_query)]
    )
    log_event(
        logging.INFO,
        "intent_detected",
        request_id=request_id,
        goal=intent.goal,
        location=intent.location,
        state=intent.state,
    )

    if intent.goal == "forecast":
        result = app.invoke({
            "user_query": user_query,
            "request_id": request_id,
        })
        memory.save_context({"input": user_query}, {
                            "output": result["summary"]})
        log_event(
            logging.INFO,
            "forecast_response",
            request_id=request_id,
        )
        graph_data = build_graph_data(result, intent)
        return {
            "reply": result["summary"],
            "intent": intent.goal,
            "include_plot": intent.include_plot,
            "plot_metric": intent.plot_metric,
            "graph_data": graph_data,
        }

    if intent.goal == "research":
        raw_response = get_research_chain(llm).invoke({"question": user_query})
        answer = raw_response.get("answer") if isinstance(
            raw_response, dict) else str(raw_response)
        log_event(
            logging.INFO,
            "research_response",
            request_id=request_id,
        )
        return {
            "reply": answer,
            "intent": intent.goal,
            "include_plot": intent.include_plot,
            "plot_metric": intent.plot_metric,
            "graph_data": None,
        }

    log_event(
        logging.WARNING,
        "intent_unrecognized",
        request_id=request_id,
        goal=intent.goal,
    )
    return {
        "reply": "Unrecognized goal.",
        "intent": intent.goal,
        "include_plot": intent.include_plot,
        "plot_metric": intent.plot_metric,
        "graph_data": None,
    }


def print_chat_history():
    """Log the in-memory conversation history (memory.chat_memory.messages) to the mangrove_agent logger."""
    log_event(logging.INFO, "chat_history_start")
    for msg in memory.chat_memory.messages:
        role = "👤 User" if msg.type == "human" else "🤖 Assistant"
        log_event(
            logging.INFO,
            "chat_message",
            role=role,
            content=msg.content,
        )


# if __name__ == "__main__":
#     memory.clear()

#     test_queries = [
#         "What are mangroves?",
#         "How are mangroves doing in Florida?",
#         "What is the impact of climate change on mangrove coverage?",
#         "How are mangroves doing in Vermont?"
#     ]

#     for query in test_queries:
#         print(f"💬 {query}")
#         response = run_agent(query)
#         print(f"🤖 {response['final_output']}")
#         print()

#     print_chat_history()
