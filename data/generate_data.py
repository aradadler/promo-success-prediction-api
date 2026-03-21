"""Synthetic promotion dataset generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROWS = 10_000
DEFAULT_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent / "promotions.csv"


def generate_base_features(num_rows: int, seed: int) -> pd.DataFrame:
    """Generate the base promotion inputs."""
    rng = np.random.default_rng(seed)

    price = rng.uniform(5.0, 25.0, size=num_rows)
    discount_pct = rng.uniform(0.05, 0.35, size=num_rows)
    baseline_units = rng.integers(200, 5001, size=num_rows)
    cogs_ratio = rng.uniform(0.35, 0.75, size=num_rows)
    cogs = price * cogs_ratio
    cannibalization_pct = rng.uniform(0.0, 0.30, size=num_rows)
    duration_weeks = rng.integers(1, 5, size=num_rows)

    return pd.DataFrame(
        {
            "price": price,
            "discount_pct": discount_pct,
            "baseline_units": baseline_units,
            "cogs": cogs,
            "cannibalization_pct": cannibalization_pct,
            "duration_weeks": duration_weeks,
        }
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used by the simulator and downstream models."""
    promo_price = df["price"] * (1.0 - df["discount_pct"])
    promo_margin_per_unit = promo_price - df["cogs"]
    margin_per_unit = df["price"] - df["cogs"]

    safe_margin = np.maximum(margin_per_unit.to_numpy(), 1e-6)
    discount_to_margin_ratio = (df["price"] * df["discount_pct"]) / safe_margin
    baseline_profit = df["baseline_units"] * margin_per_unit

    enriched = df.copy()
    enriched["promo_margin_per_unit"] = promo_margin_per_unit
    enriched["discount_to_margin_ratio"] = discount_to_margin_ratio
    enriched["baseline_profit"] = baseline_profit
    return enriched


def simulate_outcomes(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Simulate uplift and profit outcomes from promotion inputs."""
    rng = np.random.default_rng(seed + 1)

    # The uplift model is calibrated to produce a more usable label balance
    # while preserving realistic directional relationships.
    discount_signal = 0.26 + (2.10 * df["discount_pct"])
    cannibalization_penalty = 0.52 * df["cannibalization_pct"]
    margin_penalty = 0.08 * np.clip(df["discount_to_margin_ratio"] - 1.10, a_min=0.0, a_max=None)
    duration_effect = 0.03 * (df["duration_weeks"] - 1)
    noise = rng.normal(loc=0.0, scale=0.02, size=len(df))

    uplift_pct = discount_signal - cannibalization_penalty - margin_penalty + duration_effect + noise
    uplift_pct = np.clip(uplift_pct, a_min=0.0, a_max=1.25)

    expected_units = df["baseline_units"] * (1.0 + uplift_pct) * (1.0 - 0.30 * df["cannibalization_pct"])
    incremental_units = expected_units - df["baseline_units"]
    baseline_profit_reference = df["baseline_units"] * (df["price"] - df["cogs"])
    promo_profit = expected_units * df["promo_margin_per_unit"]
    incremental_profit = promo_profit - baseline_profit_reference

    discount_investment = df["baseline_units"] * df["price"] * df["discount_pct"]
    safe_investment = np.maximum(discount_investment.to_numpy(), 1e-6)
    roi = incremental_profit / safe_investment

    simulated = df.copy()
    simulated["uplift_pct"] = uplift_pct
    simulated["expected_units"] = expected_units
    simulated["incremental_units"] = incremental_units
    simulated["promo_profit"] = promo_profit
    simulated["baseline_profit_reference"] = baseline_profit_reference
    simulated["incremental_profit"] = incremental_profit
    simulated["roi"] = roi
    simulated["success_label"] = (simulated["roi"] >= 0.20).astype(int)
    return simulated


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric fields for cleaner persisted output."""
    rounded = df.copy()
    float_columns = rounded.select_dtypes(include=["float64", "float32"]).columns
    rounded[float_columns] = rounded[float_columns].round(4)
    return rounded


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact preview and label distribution."""
    print("First few rows:")
    print(df.head().to_string(index=False))
    print("\nSuccess label distribution:")
    print(df["success_label"].value_counts(normalize=False).sort_index().to_string())


def build_dataset(num_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate the complete synthetic promotion dataset."""
    dataset = generate_base_features(num_rows=num_rows, seed=seed)
    dataset = add_engineered_features(dataset)
    dataset = simulate_outcomes(dataset, seed=seed)
    return finalize_dataset(dataset)


def main() -> None:
    """Generate, save, and summarize the synthetic promotions dataset."""
    dataset = build_dataset()
    save_dataset(dataset, OUTPUT_PATH)
    print_summary(dataset)


if __name__ == "__main__":
    main()
