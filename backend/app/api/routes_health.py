"""
Status check endpoint mapping model loading states and execution devices.
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from app.utils.logging_utils import get_logger

router = APIRouter()
logger = get_logger("api_health")

@router.get("/health")
def health_check(request: Request):
    """
    Inquire backend operational health status.
    """
    try:
        registry = getattr(request.app.state, "model_registry", None)
        if registry is not None and registry.predictors:
            first_pred = next(iter(registry.predictors.values()))
            return {
                "status": "healthy",
                "device": str(first_pred.device),
                "category": first_pred.category,
                "threshold": float(first_pred.threshold),
                "loaded_categories": list(registry.predictors.keys()),
                "thresholds": {cat: float(p.threshold) for cat, p in registry.predictors.items()},
                "model_loaded": True
            }
        else:
            return {
                "status": "degraded",
                "device": "none",
                "category": "none",
                "threshold": 0.0,
                "loaded_categories": [],
                "thresholds": {},
                "model_loaded": False
            }
    except Exception as err:
        logger.error(f"Health check failed: {str(err)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Unexpected health check error: {str(err)}"}
        )
