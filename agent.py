from typing import Literal
import os
import re
import datetime as dt
import requests
import pandas as pd
import joblib
import ee

from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Literal
from typing import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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


# --- API Keys and Auth ---
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_INDEX_NAME'] = 'mangrove-index'

# Earth Engine setup
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    # Replace with your GCP EE project
    ee.Initialize(project='ee-lgharijanto123')

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.8)


# --- Pydantic Intent Schema ---
class QueryIntent(BaseModel):
    goal: Literal["forecast", "research"] = Field(
        description="""
        'forecast': the user wants a contextual overview or analysis. May include NDVI forecasts and real-time data if a location is mentioned.
        'research': the user is asking a general knowledge question not tied to a specific place or data fetch.
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

Only use the allowed values for 'goal'. Do not guess location if it's unclear.
""")

structured_llm = llm.with_structured_output(QueryIntent)


class State(TypedDict):
    # === User input & intent extraction ===
    user_query: str                             # Always provided by the user
    goal: Optional[str]                         # 'forecast' or 'research'
    location: Optional[str]                     # e.g. "Key West"
    state: Optional[str]                        # e.g. "FL"

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
embeddings = OpenAIEmbeddings()
db = Pinecone.from_existing_index(
    index_name="mangrove-index", embedding=embeddings)
retriever = db.as_retriever()
research_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory)


def extract_intent_node(state: dict) -> dict:
    user_query = state["user_query"]

    intent = structured_llm.invoke([system_prompt, user_query])

    return {
        "goal": intent.goal,
        "location": intent.location,
        "state": intent.state
    }


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
            print(f"[{station_id}] Bad response: {r.status_code}")
            return None

        data = r.json()
        if "data" not in data or not data["data"]:
            print(f"[{station_id}] No 'data' in wind response")
            return None

        wind_speed = float(data["data"][0]["s"])
        return wind_speed
    except Exception as e:
        print(f"[{station_id}] Wind speed fetch error: {e}")
        return None


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
        print(f"[{station_id}] No data in response.")
        return None
    except Exception as e:
        print(f"[{station_id}] Water level fetch error: {e}")
        return None


def try_fetch_with_fallback(fetch_func, candidate_ids: list[str]) -> Optional[float]:
    for station_id in candidate_ids:
        result = fetch_func(station_id)
        if result is not None:
            return result
    return None


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

    weekly_ndvi = []
    for i in range(0, len(ndvi_floats) - 56 + 56, 7):
        chunk = ndvi_floats[i:i+7]
        if len(chunk) == 7:
            weekly_ndvi.append(sum(chunk) / 7)

    if len(weekly_ndvi) < 8:
        raise ValueError(
            f"Only {len(weekly_ndvi)} weekly values found, need 8.")

    return weekly_ndvi[-8:]


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
            print(f"Chunk {chunk_start}–{chunk_end} failed: {e}")

    if not all_dfs:
        print(f"No valid data collected for {product}.")
        return []

    full_df = pd.concat(all_dfs).sort_index()

    # Resample into weekly means (week ends Sunday by default)
    value_col = "s" if product == "wind" else "v"
    weekly = full_df[value_col].resample("W").mean().dropna()

    if len(weekly) < 8:
        print(
            f"Only {len(weekly)} weekly values found for {product}, expected 8.")
        return weekly.tolist()  # Return whatever we have

    return weekly.tail(8).tolist()


def predict_ndvi_node(state: dict) -> dict:
    try:
        features = state.get("feature_vector")
        if not features or len(features) != 23:
            print("Invalid or missing feature vector")
            return {"ndvi_prediction": None}

        X = pd.DataFrame([features], columns=scaler.feature_names_in_)

        X_scaled = scaler.transform(X)

        pred = xgb_model.predict(X_scaled)[0]
        print("NDVI Prediction:", pred)

        return {"ndvi_prediction": float(pred)}

    except Exception as e:
        print("Prediction error:", e)
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
summary_chain: Runnable = summary_prompt | llm


def generate_summary_node(state: State) -> dict:
    try:
        response = summary_chain.invoke({
            "user_query": state["user_query"],
            "ndvi_prediction": state["ndvi_prediction"],
            "wind_speed": state["environmental_data"]["wind_speed"],
            "water_level": state["environmental_data"]["water_level"]
        })

        return {
            "summary": response.content
        }

    except Exception as e:
        print("[summary] Error:", e)
        return {"summary": "Unable to generate summary."}


# --- Graph Setup ---
workflow = StateGraph(State)
workflow.add_node("extract_intent", extract_intent_node)
workflow.add_node("exit_early", lambda state: {
                  "final_output": "[ROUTED TO RESEARCH AGENT]"})

workflow.set_entry_point("extract_intent")
workflow.add_conditional_edges("extract_intent", route_by_goal, {
    "forecast": "select_stations",
    "research": "exit_early"
})

workflow.add_node("select_stations", select_station_by_location_node)
workflow.add_node("resolve_gps", resolve_gps_from_location_node)
workflow.add_node("fetch_environmental_data", fetch_environmental_data_node)
workflow.add_node("fetch_weekly_lags", fetch_weekly_lags_node)

workflow.add_edge("select_stations", "resolve_gps")
workflow.add_edge("select_stations", "fetch_environmental_data")
workflow.add_edge("select_stations", "fetch_weekly_lags")

workflow.add_node("fetch_ndvi_lags", fetch_ndvi_lags_node)
workflow.add_edge("resolve_gps", "fetch_ndvi_lags")

workflow.add_node("build_feature_vector", build_feature_vector_node)
workflow.add_node("predict_ndvi", predict_ndvi_node)
workflow.add_node("generate_summary", generate_summary_node)

workflow.add_edge("fetch_ndvi_lags", "build_feature_vector")
workflow.add_edge("fetch_weekly_lags", "build_feature_vector")
workflow.add_edge("build_feature_vector", "predict_ndvi")
workflow.add_edge("predict_ndvi", "generate_summary")
workflow.add_edge("generate_summary", END)

app = workflow.compile()


def forecast_chain(user_query: str) -> str:
    result = app.invoke({"user_query": user_query})
    return result.get("summary", "No forecast available.")


def run_research_chain(user_query: str) -> str:
    raw_response = research_chain.invoke({"question": user_query})
    return raw_response.get("answer") if isinstance(raw_response, dict) else str(raw_response)


def extract_goal_and_location(user_query: str):
    intent = structured_llm.invoke([system_prompt, user_query])
    return intent.goal


def run_agent(user_query: str):
    print(f"🧪 User query: {user_query}")

    goal = extract_goal_and_location(user_query)
    print(f"🔍 Detected goal: {goal}")

    if goal == "forecast":
        final_output = forecast_chain(user_query)

        memory.save_context({"input": user_query}, {"output": final_output})

    elif goal == "research":
        final_output = run_research_chain(user_query)

    else:
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
    os.environ['OPENAI_API_KEY'] = api_key  # update environment
    return ChatOpenAI(openai_api_key=api_key, model="gpt-4o", temperature=0.8)


def run_mangrove_agent(user_query: str, api_key: str) -> str:
    try:
        llm = build_llm(api_key)
        return _run_with_llm(user_query, llm)
    except Exception as e:
        return f"Error processing query: {str(e)}"


def _run_with_llm(user_query: str, llm: ChatOpenAI) -> str:
    print(f"🧪 User query: {user_query}")  # ✅ always log input

    intent_chain = llm.with_structured_output(QueryIntent)
    intent = intent_chain.invoke([system_prompt, user_query])
    print(f"🔍 Detected goal: {intent.goal}")  # ✅ log goal

    if intent.goal == "forecast":
        result = app.invoke({"user_query": user_query})
        memory.save_context({"input": user_query}, {
                            "output": result["summary"]})
        print(f"🌤️ Forecast response: {result['summary']}")  # ✅ log output
        return result["summary"]

    elif intent.goal == "research":
        chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory)
        raw_response = chain.invoke({"question": user_query})
        answer = raw_response.get("answer") if isinstance(
            raw_response, dict) else str(raw_response)
        print(f"📚 Research response: {answer}")  # ✅ log output
        return answer

    else:
        print(f"Unknown goal: {intent.goal}")
        return "Unrecognized goal."


def print_chat_history():
    print("🧠 Chat History:")
    for msg in memory.chat_memory.messages:
        role = "👤 User" if msg.type == "human" else "🤖 Assistant"
        print(f"{role}: {msg.content}\n")


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
