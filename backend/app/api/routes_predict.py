"""
Prediction API endpoint executing anomaly classification on uploaded images.
"""

import io
from PIL import Image
from fastapi import APIRouter, Request, UploadFile, File, Query, status
from fastapi.responses import JSONResponse
from app.utils.logging_utils import get_logger

router = APIRouter()
logger = get_logger("api_predict")

@router.post("/predict")
async def predict_anomaly(
    request: Request, 
    file: UploadFile = File(...),
    category: str = Query(..., description="Target inspection category name")
):
    """
    Accept image uploads and return STFPM classification and anomaly map data
    for the specified inspection target category.
    """
    try:
        # 1. Retrieve the model registry from application state
        registry = getattr(request.app.state, "model_registry", None)
        if registry is None or not registry.predictors:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"message": "Model registry is not initialized or no models are loaded."}
            )

        # 2. Look up the requested category in the registry
        predictor = registry.get_predictor(category)
        if predictor is None:
            available = list(registry.predictors.keys())
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": f"Category '{category}' is invalid or not loaded. Available categories: {available}"}
            )

        # 3. Validate MIME content type
        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Uploaded file is not a valid image format."}
            )

        # 4. Read file stream and verify Pillow image parsing
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": f"Failed to parse uploaded image: {str(e)}"}
            )

        # 5. Execute prediction using the category predictor
        try:
            result = predictor.predict(img)
            
            # Serialize 2D NumPy array anomaly map to nested list
            anomaly_map_list = result["anomaly_map"].tolist()
            
            return {
                "is_anomaly": int(result["is_anomaly"]),
                "anomaly_score": float(result["anomaly_score"]),
                "anomaly_map": anomaly_map_list,
                "threshold": float(predictor.threshold),
                "inference_time": float(result["inference_time"]),
                "category": category
            }
        except Exception as pred_err:
            logger.error(f"Inference prediction error for category '{category}': {str(pred_err)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"Inference prediction error: {str(pred_err)}"}
            )

    except Exception as err:
        logger.error(f"Unexpected prediction route error: {str(err)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Unexpected server error: {str(err)}"}
        )
