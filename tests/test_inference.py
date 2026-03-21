"""Tests for the inference service layer."""

from app.inference import predict
from app.schemas import PredictionRequest


def test_predict_returns_placeholder_response() -> None:
    """Ensure the placeholder prediction flow returns the expected structure."""
    response = predict(PredictionRequest())
    assert response.prediction == 0.0
    assert response.model_version == "0.0.0-placeholder"

