"""Pydantic schemas for API request and response models."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Placeholder request schema for inference input."""

    # Add feature fields here once the contract is defined.
    example_feature: float = Field(default=0.0, description="Temporary placeholder field.")


class PredictionResponse(BaseModel):
    """Placeholder response schema for inference output."""

    prediction: float = Field(..., description="Predicted success score or probability.")
    model_version: str = Field(..., description="Version of the loaded inference artifact.")

