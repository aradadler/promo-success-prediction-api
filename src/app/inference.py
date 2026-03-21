"""Inference service layer for model loading and prediction orchestration."""

from app.features import build_feature_vector
from app.model import DummyModel, load_model
from app.schemas import PredictionRequest, PredictionResponse


def predict(payload: PredictionRequest) -> PredictionResponse:
    """Run a placeholder prediction flow and return a typed response."""
    model: DummyModel = load_model()
    feature_vector = build_feature_vector(payload)
    prediction = model.predict(feature_vector)
    return PredictionResponse(prediction=prediction, model_version=model.version)

