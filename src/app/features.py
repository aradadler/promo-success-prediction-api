"""Feature engineering utilities for transforming raw request data."""

from app.schemas import PredictionRequest


def build_feature_vector(payload: PredictionRequest) -> list[float]:
    """Convert the request payload into a model-ready feature vector."""
    # Replace this placeholder with validated feature engineering logic.
    return [payload.example_feature]

