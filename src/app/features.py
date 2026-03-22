"""Deterministic feature engineering utilities for approved model inputs."""

from __future__ import annotations

import pandas as pd

from .schemas import PredictionRequest


RAW_INPUT_COLUMNS = [
    "price",
    "discount_pct",
    "baseline_units",
    "cogs",
    "cannibalization_pct",
    "duration_weeks",
]

MODEL_FEATURE_COLUMNS = [
    "price",
    "discount_pct",
    "baseline_units",
    "cogs",
    "cannibalization_pct",
    "duration_weeks",
    "promo_margin_per_unit",
    "discount_to_margin_ratio",
    "baseline_profit",
]


def build_features(raw_inputs: pd.DataFrame) -> pd.DataFrame:
    """Build the approved model feature set from raw pre-promotion inputs."""
    missing_columns = [column for column in RAW_INPUT_COLUMNS if column not in raw_inputs.columns]
    if missing_columns:
        raise ValueError(f"Missing required raw input columns: {missing_columns}")

    # Limit inputs to the approved raw fields so simulation-only columns never enter the model.
    features = raw_inputs.loc[:, RAW_INPUT_COLUMNS].copy()

    promo_price = features["price"] * (1.0 - features["discount_pct"])
    margin_per_unit = features["price"] - features["cogs"]
    safe_margin = margin_per_unit.clip(lower=1e-6)

    # These engineered features match the dataset generator and are available before launch.
    features["promo_margin_per_unit"] = promo_price - features["cogs"]
    features["discount_to_margin_ratio"] = (features["price"] * features["discount_pct"]) / safe_margin
    features["baseline_profit"] = features["baseline_units"] * margin_per_unit

    return features.loc[:, MODEL_FEATURE_COLUMNS]


def build_feature_vector(payload: PredictionRequest) -> list[float]:
    """Convert a single request payload into the ordered model feature vector."""
    row = pd.DataFrame(
        [
            {
                "price": payload.price,
                "discount_pct": payload.discount_pct,
                "baseline_units": payload.baseline_units,
                "cogs": payload.cogs,
                "cannibalization_pct": payload.cannibalization_pct,
                "duration_weeks": payload.duration_weeks,
            }
        ]
    )
    return build_features(row).iloc[0].tolist()
