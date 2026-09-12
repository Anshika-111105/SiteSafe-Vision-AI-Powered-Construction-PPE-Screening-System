import pytest
from pydantic import ValidationError
from app.schemas import PredictionResponse, HealthResponse, MetadataResponse, VersionResponse


def test_prediction_response_schema_valid():
    res = PredictionResponse(
        prediction="FULL_PPE",
        confidence=0.9542,
        risk_level="LOW",
        recommendation="COMPLIANT: Standard PPE detected.",
        model_version="1.0.0",
        request_id="test-uuid-1234",
        probabilities={"FULL_PPE": 0.9542, "PARTIAL_PPE": 0.0321, "NO_PPE": 0.0137},
        inference_latency_ms=12.45,
    )
    assert res.prediction == "FULL_PPE"
    assert res.confidence == 0.9542
    assert res.risk_level == "LOW"


def test_prediction_response_confidence_bounds():
    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction="FULL_PPE",
            confidence=1.5, # Invalid > 1.0
            risk_level="LOW",
            recommendation="Test",
            model_version="1.0.0",
            request_id="test-uuid",
            probabilities={"FULL_PPE": 1.0, "PARTIAL_PPE": 0.0, "NO_PPE": 0.0},
            inference_latency_ms=10.0,
        )


def test_health_response_schema():
    h = HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name="mobilenet_v3_large",
        version="1.0.0",
        timestamp_utc="2026-09-12T12:00:00Z",
    )
    assert h.status == "healthy"
    assert h.model_loaded is True
