"""Pydantic schemas for API request and response models."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Validated request schema for pre-promotion prediction inputs."""

    price: float = Field(..., gt=0, description="Regular unit price before discount.")
    discount_pct: float = Field(..., ge=0, le=1, description="Discount percentage expressed as a decimal.")
    baseline_units: int = Field(..., gt=0, description="Expected baseline unit sales without the promotion.")
    cogs: float = Field(..., gt=0, description="Cost of goods sold per unit.")
    cannibalization_pct: float = Field(
        ...,
        ge=0,
        le=1,
        description="Expected share of promoted sales that cannibalize existing demand.",
    )
    duration_weeks: int = Field(..., gt=0, description="Promotion duration in weeks.")


class PredictionResponse(BaseModel):
    """Response schema for prediction outputs."""

    success_probability: float = Field(..., description="Predicted probability that the promotion will succeed.")
    predicted_label: str = Field(..., description="Human-readable classification label derived from the score.")
    confidence: str = Field(..., description="Qualitative confidence band for the prediction.")
    interpretation: str = Field(..., description="Human-readable interpretation of the predicted probability.")
    recommendation: str = Field(..., description="Suggested action based on the predicted outcome.")
