import io
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.middleware import RequestTrackingMiddleware
from app.predictor import get_predictor
from app.risk import evaluate_risk_and_recommendation
from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
    VersionResponse,
)
from src.utils.logger import setup_logger

logger = setup_logger("api_main")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SiteSafe Vision | Construction PPE Screening</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(30, 41, 59, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --accent: #f59e0b;
      --full: #10b981;
      --partial: #f59e0b;
      --none: #ef4444;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: radial-gradient(circle at 50% 0%, #1e293b 0%, var(--bg) 75%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }
    .container { max-width: 960px; width: 100%; }
    .header { text-align: center; margin-bottom: 2rem; }
    .header h1 { font-size: 2.25rem; font-weight: 800; letter-spacing: -0.025em; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
    .header p { color: var(--text-muted); font-size: 1rem; margin-top: 0.5rem; }
    .nav-links { display: flex; justify-content: center; gap: 1rem; margin-top: 0.75rem; }
    .nav-links a { color: var(--primary); text-decoration: none; font-size: 0.875rem; font-weight: 500; padding: 0.25rem 0.75rem; border-radius: 9999px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); transition: all 0.2s; }
    .nav-links a:hover { background: rgba(59,130,246,0.2); }
    .disclaimer { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); color: #fde68a; padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.8125rem; margin-bottom: 1.5rem; line-height: 1.4; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
    .card h2 { font-size: 1.125rem; font-weight: 600; margin-bottom: 1rem; }
    .dropzone { border: 2px dashed rgba(255,255,255,0.15); border-radius: 10px; padding: 2rem 1rem; text-align: center; cursor: pointer; transition: all 0.2s; background: rgba(15, 23, 42, 0.4); position: relative; }
    .dropzone:hover { border-color: var(--primary); background: rgba(59,130,246,0.05); }
    .dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
    .preview-img { width: 100%; max-height: 260px; object-fit: contain; border-radius: 8px; margin-top: 1rem; display: none; }
    .btn { display: inline-block; width: 100%; padding: 0.75rem; background: #2563eb; color: #fff; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 1rem; transition: background 0.2s; }
    .btn:hover { background: #1d4ed8; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .badge { display: inline-block; padding: 0.375rem 0.875rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 700; text-transform: uppercase; }
    .badge-FULL_PPE { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-PARTIAL_PPE { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-NO_PPE { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .risk-badge { font-weight: 700; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; }
    .risk-LOW { background: #064e3b; color: #6ee7b7; }
    .risk-MEDIUM { background: #78350f; color: #fde68a; }
    .risk-HIGH { background: #7f1d1d; color: #fca5a5; }
    .metric-row { display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0; padding-bottom: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .bar-wrap { margin-top: 0.5rem; }
    .bar-label { display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 0.25rem; color: var(--text-muted); }
    .bar-bg { width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease-out; }
    .rec-box { background: rgba(15, 23, 42, 0.6); border-left: 4px solid var(--primary); padding: 0.875rem; border-radius: 4px; font-size: 0.875rem; margin-top: 1rem; line-height: 1.5; }
    .loading { display: none; text-align: center; color: var(--primary); font-size: 0.875rem; margin-top: 1rem; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🦺 SiteSafe Vision</h1>
      <p>AI-Powered Construction PPE Compliance Screening System</p>
      <div class="nav-links">
        <a href="/docs" target="_blank">📖 Swagger API Docs</a>
        <a href="/health" target="_blank">🩺 Health Status</a>
        <a href="https://github.com/Anshika-111105/SiteSafe-Vision-AI-Powered-Construction-PPE-Screening-System" target="_blank">🐙 GitHub Repository</a>
      </div>
    </div>

    <div class="disclaimer">
      ⚠️ <strong>SAFETY & REGULATORY DISCLAIMER:</strong> SiteSafe Vision is an AI-based image screening aid. It does not perform certified safety audits, autonomous entry gating, or legally binding inspections. Designed to assist safety supervisors on construction sites.
    </div>

    <div class="grid">
      <div class="card">
        <h2>Worker Image Upload</h2>
        <div class="dropzone" id="dropzone">
          <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp">
          <p id="dropText">📁 Drag & drop worker crop image here, or <strong>click to browse</strong></p>
          <img id="preview" class="preview-img" alt="Preview">
        </div>
        <button id="submitBtn" class="btn" disabled>🔍 Screen Worker Image</button>
        <div id="loading" class="loading">⚙️ Analyzing worker PPE compliance...</div>
      </div>

      <div class="card">
        <h2>Screening Assessment Results</h2>
        <div id="resultsPlaceholder" style="text-align: center; color: var(--text-muted); padding: 3rem 1rem;">
          Upload a worker crop image and click "Screen Worker Image" to view real-time compliance results.
        </div>
        <div id="resultsPanel" style="display: none;">
          <div class="metric-row">
            <span>Compliance Status:</span>
            <span id="predBadge" class="badge"></span>
          </div>
          <div class="metric-row">
            <span>Model Confidence:</span>
            <strong id="confVal"></strong>
          </div>
          <div class="metric-row">
            <span>Risk Tier:</span>
            <span id="riskBadge" class="risk-badge"></span>
          </div>
          <div class="rec-box" id="recText"></div>

          <div style="margin-top: 1.25rem;">
            <div style="font-size: 0.8125rem; font-weight: 600; margin-bottom: 0.5rem;">Class Probability Distribution</div>
            <div class="bar-wrap">
              <div class="bar-label"><span>FULL_PPE</span><span id="pFull"></span></div>
              <div class="bar-bg"><div class="bar-fill" id="bFull" style="background:#10b981; width:0%;"></div></div>
            </div>
            <div class="bar-wrap">
              <div class="bar-label"><span>PARTIAL_PPE</span><span id="pPartial"></span></div>
              <div class="bar-bg"><div class="bar-fill" id="bPartial" style="background:#f59e0b; width:0%;"></div></div>
            </div>
            <div class="bar-wrap">
              <div class="bar-label"><span>NO_PPE</span><span id="pNone"></span></div>
              <div class="bar-bg"><div class="bar-fill" id="bNone" style="background:#ef4444; width:0%;"></div></div>
            </div>
          </div>
          <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 1rem; text-align: right;" id="metaText"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const dropText = document.getElementById('dropText');
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const resultsPanel = document.getElementById('resultsPanel');
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');

    let selectedFile = null;

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (re) => {
          preview.src = re.target.result;
          preview.style.display = 'block';
          dropText.style.display = 'none';
          submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
      }
    });

    submitBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      loading.style.display = 'block';
      submitBtn.disabled = true;

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const resp = await fetch('/predict', { method: 'POST', body: formData });
        const data = await resp.json();
        loading.style.display = 'none';
        submitBtn.disabled = false;

        if (resp.ok) {
          resultsPlaceholder.style.display = 'none';
          resultsPanel.style.display = 'block';

          const badge = document.getElementById('predBadge');
          badge.className = 'badge badge-' + data.prediction;
          badge.textContent = data.prediction.replace('_', ' ');

          document.getElementById('confVal').textContent = (data.confidence * 100).toFixed(1) + '%';

          const risk = document.getElementById('riskBadge');
          risk.className = 'risk-badge risk-' + data.risk_level;
          risk.textContent = data.risk_level + ' RISK';

          document.getElementById('recText').textContent = data.recommendation;

          const probs = data.probabilities || {};
          const full = (probs['FULL_PPE'] || 0) * 100;
          const partial = (probs['PARTIAL_PPE'] || 0) * 100;
          const none = (probs['NO_PPE'] || 0) * 100;

          document.getElementById('pFull').textContent = full.toFixed(1) + '%';
          document.getElementById('bFull').style.width = full + '%';
          document.getElementById('pPartial').textContent = partial.toFixed(1) + '%';
          document.getElementById('bPartial').style.width = partial + '%';
          document.getElementById('pNone').textContent = none.toFixed(1) + '%';
          document.getElementById('bNone').style.width = none + '%';

          document.getElementById('metaText').textContent = 'Model: ' + (data.model_name || 'mobilenet_v3') + ' | Latency: ' + data.inference_latency_ms + 'ms';
        } else {
          alert('Error: ' + (data.detail || 'Prediction failed'));
        }
      } catch (err) {
        loading.style.display = 'none';
        submitBtn.disabled = false;
        alert('Network Error: ' + err.message);
      }
    });
  </script>
</body>
</html>
"""


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
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=DASHBOARD_HTML)
    return {
        "service": "SiteSafe Vision API",
        "status": "online",
        "docs": "/docs",
        "ui": "/ui",
        "version": "1.0.0",
        "description": "AI-Powered Construction PPE Compliance Screening System",
    }


@app.get("/ui", response_class=HTMLResponse, tags=["General"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["General"])
async def web_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/health", response_model=HealthResponse, tags=["Health & Status"])
async def health_check():
    predictor = get_predictor()

    status_str = "healthy" if predictor.is_loaded else "degraded"
    return HealthResponse(
        status=status_str,
        model_loaded=predictor.is_loaded,
        model_name=getattr(predictor, "model_name", "UNKNOWN"),
        version="1.0.0",
        timestamp_utc=datetime.now(UTC).isoformat(),
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
            detail=f"Corrupted or unreadable image file: {e!s}",
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
    files: list[UploadFile] = File(..., description="List of worker crop image files"),
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
