import io
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import (
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    VersionResponse,
)
from app.risk import evaluate_risk_and_recommendation
from app.predictor import get_predictor
from app.middleware import RequestTrackingMiddleware
from src.utils.logger import setup_logger

logger = setup_logger("api_main")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024 # 10 MB

app = FastAPI(
    title="SiteSafe Vision API",
    description=(
        "Production AI-Powered Construction PPE Compliance Screening Service. "
        "Classifies worker-centered crops into FULL_PPE, PARTIAL_PPE, and NO_PPE, "
        "and computes calibrated risk levels and safety recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware
app.add_middleware(RequestTrackingMiddleware)


@app.on_event("startup")
async def startup_event():
    logger.info("SiteSafe Vision FastAPI application starting up...")
    predictor = get_predictor()
    if predictor.is_loaded:
        logger.info(f"Loaded champion model '{predictor.model_name}' successfully.")
    else:
        logger.warning("Production model weights not found on startup.")


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "SiteSafe Vision API",
        "status": "online",
        "docs": "/docs",
        "version": "1.0.0",
        "description": "AI-Powered Construction PPE Compliance Screening System",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health & Status"])
async def health_check():
    predictor = get_predictor()
    status_str = "healthy" if predictor.is_loaded else "degraded"
    return HealthResponse(
        status=status_str,
        model_loaded=predictor.is_loaded,
        model_name=getattr(predictor, "model_name", "UNKNOWN"),
        version="1.0.0",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["Health & Status"])
async def get_metadata():
    predictor = get_predictor()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot retrieve metadata.",
        )

    meta = predictor.metadata
    return MetadataResponse(
        model_name=meta.get("model_name", getattr(predictor, "model_name", "mobilenet_v3_large")),
        model_version=meta.get("model_version", "1.0.0"),
        architecture=meta.get("architecture", "torchvision.models.mobilenet_v3_large"),
        training_commit=meta.get("training_commit", "UNKNOWN"),
        dataset_version=meta.get("dataset_version", "1.0.0"),
        class_mapping=predictor.class_to_idx,
        input_size=meta.get("input_size", 224),
        normalization=meta.get("normalization", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
        validation_metrics=meta.get("validation_metrics"),
        artifact_sha256=meta.get("artifact_sha256"),
    )


@app.get("/version", response_model=VersionResponse, tags=["Health & Status"])
async def get_version():
    predictor = get_predictor()
    git_commit = predictor.metadata.get("training_commit", "UNKNOWN")
    return VersionResponse(
        application_name="SiteSafe Vision",
        version="1.0.0",
        api_version="v1",
        git_commit=git_commit,
        dvc_status="TRACKED",
    )


def validate_and_open_image(file_bytes: bytes, filename: str, content_type: str) -> Image.Image:
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded. Image must contain valid byte data.",
        )

    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds max allowed limit of {MAX_FILE_BYTES // (1024*1024)}MB.",
        )

    # Validate image decoding
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        # Re-open after verify
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unreadable image file: {str(e)}",
        )

    w, h = img.size
    if w < 16 or h < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions too small ({w}x{h}px). Minimum required size is 16x16px.",
        )

    return img


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(
    request: Request,
    file: UploadFile = File(..., description="Worker crop image file (JPEG/PNG/WebP)"),
):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    predictor = get_predictor()

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is currently unavailable.",
        )

    file_bytes = await file.read()
    image = validate_and_open_image(file_bytes, file.filename, file.content_type)

    pred_res = predictor.predict(image)
    risk_level, rec = evaluate_risk_and_recommendation(
        pred_res["prediction"],
        pred_res["confidence"],
    )

    return PredictionResponse(
        prediction=pred_res["prediction"],
        confidence=pred_res["confidence"],
        risk_level=risk_level,
        recommendation=rec,
        model_version=pred_res["model_version"],
        request_id=req_id,
        probabilities=pred_res["probabilities"],
        inference_latency_ms=pred_res["inference_latency_ms"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="List of worker crop image files"),
):
    batch_req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    predictor = get_predictor()

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is currently unavailable.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in batch upload request.",
        )

    results = []
    for f in files:
        item_req_id = str(uuid.uuid4())
        f_bytes = await f.read()
        image = validate_and_open_image(f_bytes, f.filename, f.content_type)
        pred_res = predictor.predict(image)
        risk_level, rec = evaluate_risk_and_recommendation(
            pred_res["prediction"],
            pred_res["confidence"],
        )
        results.append(
            PredictionResponse(
                prediction=pred_res["prediction"],
                confidence=pred_res["confidence"],
                risk_level=risk_level,
                recommendation=rec,
                model_version=pred_res["model_version"],
                request_id=item_req_id,
                probabilities=pred_res["probabilities"],
                inference_latency_ms=pred_res["inference_latency_ms"],
            )
        )

    return BatchPredictionResponse(
        batch_request_id=batch_req_id,
        total_images=len(results),
        results=results,
    )
