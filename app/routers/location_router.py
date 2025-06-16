# app/routers/location_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from ..core.locaton_store import update_location


router = APIRouter(tags=["Location"], prefix="/location")


class LocationSchema(BaseModel):
    lat: float
    lon: float


@router.post("/update")
async def update_client_location(data: LocationSchema):
    update_location(data.lat, data.lon)
    return {"status": "ok", "message": "location updated"}
