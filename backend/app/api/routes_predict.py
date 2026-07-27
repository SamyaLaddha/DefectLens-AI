"""
Prediction API endpoint executing anomaly classification on uploaded images.
"""

import io
from PIL import Image
from fastapi import APIRouter, Request, UploadFile, File, status
from fastapi.responses import JSONResponse
from app.utils.logging_utils import get_logger

router = APIRouter()
logger = get_logger("api_predict")

@router.post("/predict")
async def predict_anomaly(request: Request, file: UploadFile = File(...)):
    """
    Accept image uploads and return STFPM classification and anomaly map data.
    """
    try:
        # 1. Retrieve the shared predictor instance from application state
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"message": "Model checkpoint is not loaded. Ensure checkpoints are available."}
            )

        # 2. Validate MIME content type
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Uploaded file is not a valid image format."}
            )

        # 3. Read file stream and verify Pillow image parsing
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": f"Failed to parse uploaded image: {str(e)}"}
            )

        # 4. Execute prediction using the loaded model pipeline
        try:
            result = predictor.predict(img)
            
            # Serialize 2D NumPy array anomaly map to nested list
            anomaly_map_list = result["anomaly_map"].tolist()
            
            return {
                "is_anomaly": int(result["is_anomaly"]),
                "anomaly_score": float(result["anomaly_score"]),
                "anomaly_map": anomaly_map_list,
                "threshold": float(predictor.threshold),
                "inference_time": float(result["inference_time"])
            }
        except Exception as pred_err:
            logger.error(f"Inference prediction error: {str(pred_err)}")
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
