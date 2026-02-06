from typing import Optional
import json
import os
import re
import time

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import run_mangrove_agent_structured
from backend.plot_tools import build_state_ndvi_bar
from backend.state_utils import extract_states
from backend.store import (
    init_db,
    create_session,
    add_message,
    get_sessions,
    get_messages,
    get_latest_graph_data,
    delete_session,
    rename_session,
)


app = FastAPI(title="MangroveAgent API")
LAST_GRAPH_DATA: dict[str, dict] = {} # In-memory cache: session_id → last graph_data for “plot later” without re-running the agent.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)


class ChatStreamRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)


class GraphPoint(BaseModel):
    t: str
    v: float


class GraphSeries(BaseModel):
    name: str
    unit: Optional[str] = None
    points: list[GraphPoint]


class GraphData(BaseModel):
    chart_type: Optional[str] = None
    series: list[GraphSeries]
    location: Optional[str] = None
    state: Optional[str] = None
    station_ids: dict
    gps: Optional[list[float]] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Optional[str] = None
    include_plot: Optional[bool] = None
    plot_metric: Optional[str] = None
    graph_data: Optional[GraphData] = None


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


def sse(event: str, data: dict | str) -> str:
    """Format a single Server-Sent Event (event name, newline, data line, blank line). Dict data is JSON-encoded."""
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


def stream_reply(text: str, chunk_size: int = 24, delay_s: float = 0.03):
    """Yield SSE 'chunk' events for reply text (chunk_size chars per event, delay_s between) for streaming display."""
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        yield sse("chunk", {"text": chunk})
        time.sleep(delay_s)


@app.on_event("startup")
def startup() -> None:
    """Initialize the SQLite chat history database on application startup."""
    init_db()


@app.post("/sessions", response_model=SessionResponse)
def create_session_endpoint() -> SessionResponse:
    """Create a new chat session (new UUID) and return its session_id."""
    session_id = create_session()
    return SessionResponse(session_id=session_id)


@app.get("/sessions/{session_id}")
def get_session_history(session_id: str) -> dict:
    """Return session_id and list of messages for the given session; 404 if not found."""
    messages = get_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": messages}


@app.get("/sessions")
def list_sessions() -> dict:
    """Return all sessions (session_id, created_at, title) ordered by created_at descending."""
    return {"sessions": get_sessions()}


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str) -> dict:
    """Delete the session and all its messages; return session_id and deleted=True."""
    delete_session(session_id)
    return {"session_id": session_id, "deleted": True}


@app.patch("/sessions/{session_id}")
def rename_session_endpoint(session_id: str, payload: RenameRequest) -> dict:
    """Update the session title; return session_id and new title."""
    rename_session(session_id, payload.title.strip())
    return {"session_id": session_id, "title": payload.title.strip()}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Handle non-streaming chat: create/use session, add user message, run agent or quick paths (bar chart, plot-only), return reply and optional graph_data."""
    session_id = create_session(payload.session_id)
    add_message(session_id, "user", payload.message)

    plot_request = re.search(r"\b(plot|chart|graph|visualize)\b",
                             payload.message, re.IGNORECASE) is not None
    states = extract_states(payload.message)
    wants_bar = re.search(r"\b(bar|compare|between)\b",
                          payload.message, re.IGNORECASE) is not None
    if plot_request and wants_bar and len(states) >= 2:
        chart = build_state_ndvi_bar(states[:2])
        if chart["series"][0]["points"]:
            reply = (
                f"Here is a bar chart of average NDVI over the last 8 weeks "
                f"for {states[0]} vs {states[1]}."
            )
        else:
            reply = "No NDVI data available for those states."
        add_message(session_id, "assistant", reply, chart)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            intent="forecast",
            include_plot=True,
            plot_metric="ndvi",
            graph_data=chart,
        )
    metric_match = re.search(r"\b(ndvi|wind|wind speed|water level|tide|tides|water)\b",
                             payload.message, re.IGNORECASE)
    requested_metric = None
    if metric_match:
        term = metric_match.group(1).lower()
        if "wind" in term:
            requested_metric = "wind_speed"
        elif "tide" in term or "water" in term:
            requested_metric = "water_level"
        elif "ndvi" in term:
            requested_metric = "ndvi"
    def filter_graph(graph: dict) -> dict:
        if not requested_metric or requested_metric == "all":
            return graph
        filtered = graph.copy()
        filtered["series"] = [
            s for s in graph.get("series", [])
            if s.get("name") == requested_metric
        ]
        return filtered

    if plot_request and session_id in LAST_GRAPH_DATA:
        reply = "Here is the latest chart from your previous forecast."
        graph_data = filter_graph(LAST_GRAPH_DATA[session_id])
        add_message(session_id, "assistant", reply, graph_data)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            intent="forecast",
            include_plot=True,
            plot_metric=requested_metric,
            graph_data=graph_data,
        )
    if plot_request and session_id not in LAST_GRAPH_DATA:
        cached_graph = get_latest_graph_data(session_id)
        if cached_graph:
            LAST_GRAPH_DATA[session_id] = cached_graph
            reply = "Here is the latest chart from your previous forecast."
            graph_data = filter_graph(cached_graph)
            add_message(session_id, "assistant", reply, graph_data)
            return ChatResponse(
                session_id=session_id,
                reply=reply,
                intent="forecast",
                include_plot=True,
                plot_metric=requested_metric,
                graph_data=graph_data,
            )
        reply = (
            "I don't have forecast data to plot yet. Ask a forecast question "
            "with a location, for example: 'How are mangroves doing in Florida?'"
        )
        add_message(session_id, "assistant", reply)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            intent="forecast",
            include_plot=True,
            plot_metric=requested_metric,
            graph_data=None,
        )

    result = run_mangrove_agent_structured(payload.message, payload.api_key)
    reply = result.get("reply") or ""
    intent = result.get("intent")
    graph_data = result.get("graph_data")
    include_plot = result.get("include_plot")
    plot_metric = result.get("plot_metric")

    if graph_data:
        LAST_GRAPH_DATA[session_id] = graph_data

    add_message(session_id, "assistant", reply, graph_data if include_plot else None)

    if plot_request and not include_plot and session_id in LAST_GRAPH_DATA:
        return ChatResponse(
            session_id=session_id,
            reply="Here is the latest chart from your previous forecast.",
            intent="forecast",
            include_plot=True,
            graph_data=LAST_GRAPH_DATA[session_id],
        )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        intent=intent,
        include_plot=include_plot,
        plot_metric=plot_metric,
        graph_data=graph_data if include_plot else None,
    )


@app.post("/chat/stream")
def chat_stream_endpoint(payload: ChatStreamRequest):
    """Stream chat reply as Server-Sent Events: session, chunk(s), graph (optional), done, or error. Uses OPENAI_API_KEY from env."""

    def event_stream():
        session_id = create_session(payload.session_id)
        add_message(session_id, "user", payload.message)
        yield sse("session", {"session_id": session_id})

        plot_request = re.search(
            r"\b(plot|chart|graph|visualize)\b", payload.message, re.IGNORECASE
        ) is not None
        states = extract_states(payload.message)
        wants_bar = (
            re.search(r"\b(bar|compare|between)\b", payload.message, re.IGNORECASE)
            is not None
        )
        if plot_request and wants_bar and len(states) >= 2:
            chart = build_state_ndvi_bar(states[:2])
            if chart["series"][0]["points"]:
                reply = (
                    f"Here is a bar chart of average NDVI over the last 8 weeks "
                    f"for {states[0]} vs {states[1]}."
                )
            else:
                reply = "No NDVI data available for those states."
            add_message(session_id, "assistant", reply, chart)
            yield from stream_reply(reply)
            yield sse("graph", {"graph_data": chart, "plot_metric": "ndvi"})
            yield sse("done", {"intent": "forecast", "include_plot": True, "plot_metric": "ndvi"})
            return

        metric_match = re.search(
            r"\b(ndvi|wind|wind speed|water level|tide|tides|water)\b",
            payload.message,
            re.IGNORECASE,
        )
        requested_metric = None
        if metric_match:
            term = metric_match.group(1).lower()
            if "wind" in term:
                requested_metric = "wind_speed"
            elif "tide" in term or "water" in term:
                requested_metric = "water_level"
            elif "ndvi" in term:
                requested_metric = "ndvi"

        def filter_graph(graph: dict) -> dict:
            if not requested_metric or requested_metric == "all":
                return graph
            filtered = graph.copy()
            filtered["series"] = [
                s for s in graph.get("series", []) if s.get("name") == requested_metric
            ]
            return filtered

        if plot_request and session_id in LAST_GRAPH_DATA:
            reply = "Here is the latest chart from your previous forecast."
            graph_data = filter_graph(LAST_GRAPH_DATA[session_id])
            add_message(session_id, "assistant", reply, graph_data)
            yield from stream_reply(reply)
            yield sse("graph", {"graph_data": graph_data, "plot_metric": requested_metric})
            yield sse("done", {"intent": "forecast", "include_plot": True, "plot_metric": requested_metric})
            return
        if plot_request and session_id not in LAST_GRAPH_DATA:
            cached_graph = get_latest_graph_data(session_id)
            if cached_graph:
                LAST_GRAPH_DATA[session_id] = cached_graph
                reply = "Here is the latest chart from your previous forecast."
                graph_data = filter_graph(cached_graph)
                add_message(session_id, "assistant", reply, graph_data)
                yield from stream_reply(reply)
                yield sse("graph", {"graph_data": graph_data, "plot_metric": requested_metric})
                yield sse("done", {"intent": "forecast", "include_plot": True, "plot_metric": requested_metric})
                return
            reply = (
                "I don't have forecast data to plot yet. Ask a forecast question "
                "with a location, for example: 'How are mangroves doing in Florida?'"
            )
            add_message(session_id, "assistant", reply)
            yield from stream_reply(reply)
            yield sse("done", {"intent": "forecast", "include_plot": True, "plot_metric": requested_metric})
            return

        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                yield sse("error", {"message": "OPENAI_API_KEY not set"})
                return
            result = run_mangrove_agent_structured(payload.message, api_key)
        except Exception as e:
            yield sse("error", {"message": str(e)})
            return

        reply = result.get("reply") or ""
        intent = result.get("intent")
        graph_data = result.get("graph_data")
        include_plot = result.get("include_plot")
        plot_metric = result.get("plot_metric")

        if graph_data:
            LAST_GRAPH_DATA[session_id] = graph_data

        add_message(session_id, "assistant", reply, graph_data if include_plot else None)

        yield from stream_reply(reply)

        if graph_data and include_plot:
            yield sse("graph", {"graph_data": graph_data, "plot_metric": plot_metric})

        yield sse("done", {
            "intent": intent,
            "include_plot": include_plot,
            "plot_metric": plot_metric,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
