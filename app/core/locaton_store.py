import json
from pathlib import Path

LOCATION_FILE = Path("location_store.json")


def update_location(lat: float, lon: float):
    LOCATION_FILE.write_text(json.dumps({"lat": lat, "lon": lon}))


def get_last_location():
    if not LOCATION_FILE.exists():
        return {"lat": None, "lon": None}
    try:
        return json.loads(LOCATION_FILE.read_text())
    except json.JSONDecodeError:
        return {"lat": None, "lon": None}
