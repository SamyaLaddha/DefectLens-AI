"""
Prediction API endpoint executing anomaly classification on uploaded images.
Supports single-image prediction and multi-image batch inspection.
"""

import io
from typing import List, Optional
from PIL import Image
from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, Request, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse
from app.utils.logging_utils import get_logger

# Prevent decompression bomb attacks via explicit pixel limit (~4500x4500 px)
Image.MAX_IMAGE_PIXELS = 20_000_000

router = APIRouter()
logger = get_logger("api_predict")

# Batch inspection constraints
MIN_BATCH_SIZE = 4
MAX_BATCH_SIZE = 12
MAX_FILE_BYTES = 10 * 1024 * 1024       # 10 MB per file
MAX_TOTAL_BYTES = 50 * 1024 * 1024      # 50 MB total payload

@router.post("/predict")
async def predict_anomaly(
    request: Request, 
    file: UploadFile = File(...),
    category: str = Query(..., description="Target inspection category name")
):
    """
    Accept single image upload and return STFPM classification and anomaly map data
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


@router.post("/predict/batch")
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    image_ids: Optional[List[str]] = Form(None),
    category: str = Query(..., description="Target inspection category name")
):
    """
    Accept multi-image batch upload (4 to 12 images) for an inspection target category.
    Processes each image independently through the category's STFPM model offloaded
    to a worker threadpool, preserving non-blocking event loop execution.
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

        # 3. Enforce strict batch size boundaries (4 to 12 images)
        if len(files) < MIN_BATCH_SIZE:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "message": f"Batch inspection requires at least {MIN_BATCH_SIZE} images (received {len(files)}). For single-image inspection, use /api/predict."
                }
            )
        if len(files) > MAX_BATCH_SIZE:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "message": f"Batch inspection allows at most {MAX_BATCH_SIZE} images per batch (received {len(files)})."
                }
            )

        # 4. Validate client image IDs if supplied
        if image_ids is not None:
            # Flatten or validate matching length
            if len(image_ids) != len(files):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "message": f"Number of image_ids ({len(image_ids)}) does not match number of files ({len(files)})."
                    }
                )

        results = []
        total_processed = 0
        total_normal = 0
        total_anomalous = 0
        total_failed = 0
        total_payload_bytes = 0

        # 5. Process each image independently in sequential order
        for idx, file in enumerate(files):
            assigned_id = image_ids[idx] if (image_ids is not None and idx < len(image_ids)) else f"img_{idx}"
            filename = file.filename or f"image_{idx}"

            # Step 5a: Read file stream and enforce size limits
            try:
                contents = await file.read()
                file_size = len(contents)
                total_payload_bytes += file_size

                if file_size == 0:
                    raise ValueError("Uploaded file is empty (0 bytes)")
                if file_size > MAX_FILE_BYTES:
                    raise ValueError(f"File size ({file_size / (1024*1024):.2f}MB) exceeds maximum {MAX_FILE_BYTES / (1024*1024):.0f}MB limit")
                if total_payload_bytes > MAX_TOTAL_BYTES:
                    raise ValueError(f"Total batch payload exceeds {MAX_TOTAL_BYTES / (1024*1024):.0f}MB limit")

                # Step 5b: Parse and validate image dimensions
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                width, height = img.size
                if width < 16 or height < 16:
                    raise ValueError(f"Image dimensions ({width}x{height}) are smaller than minimum allowed 16x16")
                if width > 4500 or height > 4500:
                    raise ValueError(f"Image dimensions ({width}x{height}) exceed maximum allowed 4500x4500")

            except Exception as parse_err:
                logger.warning(f"Batch image {idx} ({filename}) parse failure: {str(parse_err)}")
                results.append({
                    "index": idx,
                    "image_id": assigned_id,
                    "filename": filename,
                    "status": "error",
                    "error_message": f"Failed to parse uploaded image: {str(parse_err)}",
                    "is_anomaly": None,
                    "anomaly_score": None,
                    "threshold": float(predictor.threshold),
                    "inference_time": 0.0,
                    "anomaly_map": None
                })
                total_failed += 1
                continue

            # Step 5c: Offload PyTorch inference to worker threadpool to protect event loop
            try:
                pred_res = await run_in_threadpool(predictor.predict, img)
                anomaly_map = pred_res["anomaly_map"]
                anomaly_map_list = anomaly_map.tolist() if hasattr(anomaly_map, "tolist") else list(anomaly_map)

                is_anomaly = int(pred_res["is_anomaly"])
                if is_anomaly == 1:
                    total_anomalous += 1
                else:
                    total_normal += 1
                total_processed += 1

                results.append({
                    "index": idx,
                    "image_id": assigned_id,
                    "filename": filename,
                    "status": "success",
                    "is_anomaly": is_anomaly,
                    "anomaly_score": float(pred_res["anomaly_score"]),
                    "threshold": float(predictor.threshold),
                    "inference_time": float(pred_res["inference_time"]),
                    "anomaly_map": anomaly_map_list
                })
            except Exception as infer_err:
                logger.error(f"Batch inference error for image {idx} ({assigned_id}): {str(infer_err)}")
                results.append({
                    "index": idx,
                    "image_id": assigned_id,
                    "filename": filename,
                    "status": "error",
                    "error_message": f"Inference prediction error: {str(infer_err)}",
                    "is_anomaly": None,
                    "anomaly_score": None,
                    "threshold": float(predictor.threshold),
                    "inference_time": 0.0,
                    "anomaly_map": None
                })
                total_failed += 1

        # 6. Compute overall batch verdict state
        if total_anomalous > 0 and total_failed == 0:
            batch_status = "DEFECT_DETECTED"
        elif total_anomalous > 0 and total_failed > 0:
            batch_status = "INCOMPLETE_DEFECT_DETECTED"
        elif total_failed > 0 and total_anomalous == 0:
            batch_status = "INCOMPLETE"
        else:
            batch_status = "ALL_CLEAR"

        return {
            "category": category,
            "threshold": float(predictor.threshold),
            "batch_status": batch_status,
            "total_submitted": len(files),
            "total_processed": total_processed,
            "total_normal": total_normal,
            "total_anomalous": total_anomalous,
            "total_failed": total_failed,
            "results": results
        }

    except Exception as err:
        logger.error(f"Unexpected batch route error: {str(err)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Unexpected server error: {str(err)}"}
        )
