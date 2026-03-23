"""Inference service layer for trained model prediction orchestration."""

import pandas as pd

from app.features import build_features
from app.model import get_model
from app.schemas import PredictionRequest, PredictionResponse


def predict_promotion_success(request: PredictionRequest) -> PredictionResponse:
    """Generate a promotion success prediction from validated request inputs."""
    request_frame = pd.DataFrame(
        [
            {
                "price": request.price,
                "discount_pct": request.discount_pct,
                "baseline_units": request.baseline_units,
                "cogs": request.cogs,
                "cannibalization_pct": request.cannibalization_pct,
                "duration_weeks": request.duration_weeks,
            }
        ]
    )

    model_features = build_features(request_frame)
    model = get_model()
    success_probability = float(model.predict_proba(model_features)[0, 1])

    predicted_label = "SUCCESS" if success_probability >= 0.5 else "FAILURE"

    if success_probability >= 0.8 or success_probability <= 0.2:
        confidence = "HIGH"
    elif success_probability >= 0.6 or success_probability <= 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return PredictionResponse(
        success_probability=success_probability,
        predicted_label=predicted_label,
        confidence=confidence,
    )
