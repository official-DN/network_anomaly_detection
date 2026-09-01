"""Model information and performance API routes."""

from fastapi import APIRouter
from backend.services.model_service import get_all_model_info, get_all_model_performance
from backend.services.detection_service import detection_service

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("")
async def list_models():
    """List all registered models and their status."""
    infos = await get_all_model_info()
    return {
        "success": True,
        "data": [m.model_dump() for m in infos],
        "loaded": detection_service.get_available_models(),
    }


@router.get("/performance")
async def model_performance():
    """Get performance metrics for all models."""
    metrics = await get_all_model_performance()
    return {"success": True, "data": [m.model_dump() for m in metrics]}
