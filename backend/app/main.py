"""
DefectLens AI Backend Application Entrypoint.

This module initializes the FastAPI application, registers API routers
for health check and anomaly prediction, and configures CORS middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_health import router as health_router
from app.api.routes_predict import router as predict_router

# Initialize the FastAPI app with metadata suitable for a portfolio project
app = FastAPI(
    title="DefectLens AI API",
    description="Industrial visual anomaly detection API using Student-Teacher Feature Pyramid Matching (STFPM)",
    version="1.0.0"
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

@app.get("/")
def read_root():
    """
    Root endpoint to confirm the API is online and running.
    """
    return {
        "message": "Welcome to DefectLens AI API. Use /api/health or /api/predict endpoints."
    }
