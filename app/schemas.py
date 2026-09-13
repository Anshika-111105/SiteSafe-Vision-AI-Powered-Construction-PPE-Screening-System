from typing import Any

from pydantic import BaseModel, Field


class ProbabilityBreakdown(BaseModel):
    FULL_PPE: float = Field(..., ge=0.0, le=1.0, description="Confidence for Full PPE compliance")
    PARTIAL_PPE: float = Field(..., ge=0.0, le=1.0, description="Confidence for Partial PPE compliance")
    NO_PPE: float = Field(..., ge=0.0, le=1.0, description="Confidence for No PPE compliance")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Target predicted class (FULL_PPE, PARTIAL_PPE, NO_PPE)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of top prediction")
    risk_level: str = Field(..., description="Site safety risk classification (LOW, MEDIUM, HIGH)")
    recommendation: str = Field(..., description="Actionable safety guidance for site supervisors")
    model_version: str = Field(..., description="Semantic version of the serving model")
    request_id: str = Field(..., description="Unique UUID tracking the request lifecycle")
    probabilities: dict[str, float] = Field(..., description="Class probability distribution")
    inference_latency_ms: float = Field(..., ge=0.0, description="Pure model inference duration in milliseconds")


class BatchPredictionResponse(BaseModel):
    batch_request_id: str = Field(..., description="Unique UUID for batch request")
    total_images: int = Field(..., ge=1, description="Number of successfully processed images")
    results: list[PredictionResponse] = Field(..., description="List of individual image prediction results")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health state ('healthy', 'degraded', 'unhealthy')")
    model_loaded: bool = Field(..., description="Indicates whether the production model weights are active in memory")
    model_name: str = Field(..., description="Name of the active model architecture")
    version: str = Field(..., description="API microservice version")
    timestamp_utc: str = Field(..., description="Current ISO-8601 UTC server timestamp")


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    architecture: str
    training_commit: str
    dataset_version: str
    class_mapping: dict[str, int]
    input_size: int
    normalization: dict[str, list[float]]
    validation_metrics: dict[str, Any] | None = None
    artifact_sha256: str | None = None


class VersionResponse(BaseModel):
    application_name: str = "SiteSafe Vision"
    version: str = "1.0.0"
    api_version: str = "v1"
    git_commit: str = "UNKNOWN"
    dvc_status: str = "TRACKED"
