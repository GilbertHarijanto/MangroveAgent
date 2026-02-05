import json
from pathlib import Path

from agent import get_cleaned_weekly_ndvi_series


def load_state_centroids() -> dict[str, list[float]]:
    path = Path(__file__).resolve().parents[1] / "data" / "state_centroids.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def average_ndvi_for_state(state_abbr: str) -> float | None:
    centroids = load_state_centroids()
    coords = centroids.get(state_abbr)
    if not coords:
        return None
    lat, lon = coords
    weekly = get_cleaned_weekly_ndvi_series(lat, lon)
    if not weekly:
        return None
    return round(sum(weekly) / len(weekly), 3)


def build_state_ndvi_bar(states: list[str]) -> dict:
    points = []
    for state_abbr in states:
        avg = average_ndvi_for_state(state_abbr)
        if avg is None:
            continue
        points.append({"t": state_abbr, "v": avg})

    return {
        "chart_type": "bar",
        "series": [
            {
                "name": "ndvi",
                "unit": "index",
                "points": points,
            }
        ],
        "state": None,
        "location": None,
        "station_ids": {},
        "gps": None,
    }
