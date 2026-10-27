"""
FastAPI router serving the list of successfully loaded/trained categories.
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from app.utils.logging_utils import get_logger

router = APIRouter()
logger = get_logger("api_categories")

@router.get("/categories")
def get_categories(request: Request):
    """
    Inquire the list of active categories configured with loaded model checkpoints.
    """
    try:
        registry = getattr(request.app.state, "model_registry", None)
        if registry is None:
            logger.warning("Request for categories received but model registry is not initialized.")
            return []
        
        # Return only categories that are successfully preloaded
        return list(registry.predictors.keys())
    except Exception as err:
        logger.error(f"Failed to query model registry categories: {str(err)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Unexpected categories query error: {str(err)}"}
        )
