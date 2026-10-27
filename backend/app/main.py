"""
DefectLens AI Backend Application Entrypoint.

This module initializes the FastAPI application, registers API routers
for health check and anomaly prediction, and configures CORS middleware.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_health import router as health_router
from app.api.routes_predict import router as predict_router
from app.api.routes_categories import router as categories_router
from app.inference.model_registry import ModelRegistry
from app.utils.logging_utils import get_logger

logger = get_logger("backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the ModelRegistry preloading all active category checkpoints
    try:
        registry = ModelRegistry()
        registry.load_models()
        app.state.model_registry = registry
        logger.info(f"ModelRegistry loaded with categories: {list(registry.predictors.keys())}")
    except Exception as e:
        app.state.model_registry = None
        logger.error(f"Failed to initialize ModelRegistry during startup: {str(e)}")
        logger.warning("Application running in degraded mode (inference predictions will return 503).")
    yield
    # Clean up registry and release resources
    if hasattr(app.state, "model_registry") and app.state.model_registry is not None:
        del app.state.model_registry
        logger.info("ModelRegistry resources cleaned up during shutdown lifespan.")

# Initialize the FastAPI app with metadata suitable for a portfolio project
app = FastAPI(
    title="DefectLens AI API",
    description="Industrial visual anomaly detection API using Student-Teacher Feature Pyramid Matching (STFPM)",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints from routers
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(predict_router, prefix="/api", tags=["predict"])
app.include_router(categories_router, prefix="/api", tags=["categories"])

@app.get("/")
def read_root():
    """
    Root endpoint to confirm the API is online and running.
    """
    try:
        return {
            "message": "Welcome to DefectLens AI API. Use /api/health or /api/predict endpoints."
        }
    except Exception as err:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Unexpected root error: {str(err)}"}
        )
