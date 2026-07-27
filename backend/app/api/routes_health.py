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
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is not None:
            return {
                "status": "healthy",
                "device": str(predictor.device),
                "category": predictor.category,
                "threshold": predictor.threshold,
                "model_loaded": True
            }
        else:
            return {
                "status": "degraded",
                "device": "none",
                "category": "none",
                "threshold": 0.0,
                "model_loaded": False
            }
    except Exception as err:
        logger.error(f"Health check failed: {str(err)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Unexpected health check error: {str(err)}"}
        )
