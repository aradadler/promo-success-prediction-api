# Promotion Success Prediction API

## Overview

This project is a production-minded machine learning system for a common retail planning question: **is a promotion likely to succeed before it goes live?**

In this project, a promotion is considered successful when **ROI is at least 20%**. That business definition is converted into a supervised classification problem, allowing a model to estimate promotion success from inputs that are known before launch.

The result is a portfolio project that connects business logic, synthetic data design, leakage-safe feature engineering, baseline model training, artifact persistence, and a FastAPI path for future inference.

## Business Problem / Objective

Retail teams make promotion decisions before they know the outcome. They have to choose discount levels, duration, pricing strategy, and expected volume with limited visibility into whether the campaign will actually create incremental value.

This project addresses that decision point directly. Instead of analyzing promotions only after the fact, it asks whether a promotion is likely to meet a business threshold **before launch**.

The business rule is straightforward:

```text
success = 1 if ROI >= 20%
success = 0 otherwise
```

That turns a business KPI into a machine learning target the system can train on, evaluate, and eventually serve through an API.

## Value Proposition

This project is designed to show how ML can support business planning, not just reporting.

- It helps teams make better promotion decisions earlier in the planning cycle.
- It demonstrates how business logic can be translated into an operational ML workflow.
- It shows a realistic path from dataset creation to model artifact persistence to future API-based inference.
- It frames ML as a decision-support tool for commercial teams, not just a technical exercise.

## Core ML Framing

The modeling task is:

```text
P(success | pre-promotion inputs)
```

Where:

- `success = 1` when ROI is at least 20%
- `success = 0` otherwise

This is a binary classification problem built around a business outcome. The goal is not to predict every simulated intermediate value, but to estimate whether a planned promotion is likely to clear a meaningful profitability threshold.

## System Workflow

The current workflow is structured as a simple, production-style ML pipeline:

1. Generate a synthetic promotion dataset with realistic retail planning inputs.
2. Simulate promotion outcomes such as uplift, expected units, profit, and ROI.
3. Create a binary success label from the ROI threshold.
4. Build leakage-safe model features from only the inputs available before launch.
5. Train and evaluate a baseline classification model.
6. Persist the trained artifact to `models/model.pkl`.
7. Prepare for future inference through the FastAPI application layer.

## Data Design

The dataset is intentionally designed around what a retailer could know at planning time versus what only becomes visible after a campaign is executed.

### Approved Raw Inputs

These are the inputs the model is allowed to see:

- `price`
- `discount_pct`
- `baseline_units`
- `cogs`
- `cannibalization_pct`
- `duration_weeks`

### Engineered Model Features

These features are derived from approved pre-promotion inputs and are used by the model:

- `promo_margin_per_unit`
- `discount_to_margin_ratio`
- `baseline_profit`

### Label Creation

Outcome simulation produces ROI, and the label is then created as:

```text
success_label = 1 if roi >= 0.20 else 0
```

This preserves a clean separation between business outcome generation and model training.

## Leakage Prevention

Feature leakage is one of the most important modeling risks in this project. If the model were allowed to train on values that are only known after a promotion runs, it would appear strong in training but fail in a real planning workflow.

This project avoids that issue by restricting training inputs to pre-launch information only.

### Included In The Model

Approved raw inputs:

- `price`
- `discount_pct`
- `baseline_units`
- `cogs`
- `cannibalization_pct`
- `duration_weeks`

Engineered model features:

- `promo_margin_per_unit`
- `discount_to_margin_ratio`
- `baseline_profit`

### Explicitly Excluded From The Model

These fields exist for simulation and label generation, but are not allowed into training features:

- `uplift_pct`
- `expected_units`
- `incremental_units`
- `promo_profit`
- `incremental_profit`
- `roi`
- `baseline_profit_reference`

This design makes the model more credible as a real pre-launch decision-support system.

## Repository Structure

```text
promo-success-prediction-api/
├── data/
│   ├── generate_data.py        # synthetic dataset generation and outcome simulation
│   └── promotions.csv          # generated training dataset
├── models/
│   └── model.pkl               # persisted trained baseline artifact
├── src/app/
│   ├── config.py               # app configuration
│   ├── features.py             # leakage-safe feature engineering
│   ├── inference.py            # inference service placeholder
│   ├── main.py                 # FastAPI app entrypoint
│   ├── model.py                # training, evaluation, and artifact persistence
│   └── schemas.py              # request/response schemas placeholder
├── tests/
│   ├── test_api.py
│   └── test_inference.py
├── pyproject.toml
├── README.md
└── run.sh
```

## Current Progress

Implemented today:

- Project scaffold and package structure are in place.
- Synthetic dataset generation is implemented in `data/generate_data.py`.
- Promotion outcomes are simulated and labels are created from ROI.
- Leakage-safe feature engineering is implemented in `src/app/features.py`.
- A baseline training pipeline is implemented in `src/app/model.py`.
- A Logistic Regression baseline has been trained.
- Standard evaluation metrics have been collected.
- The trained artifact is saved to `models/model.pkl`.
- FastAPI app scaffolding exists with a health endpoint.
- Inference and API schemas are still placeholder implementations and are not yet wired to the saved model artifact.

## Model Results

Baseline model results from the current implementation:

- Dataset size: `10,000` rows — number of simulated promotions used for training
- Class balance: `{0: 0.7857, 1: 0.2143}` — proportion of unsuccessful vs successful promotions
- Accuracy: `0.9685` — overall correctness of predictions
- Precision: `0.9507` — how often predicted “successful” promotions are actually successful
- Recall: `0.8998` — how many truly successful promotions the model correctly identifies
- F1: `0.9246` — balance between precision and recall
- ROC-AUC: `0.9938` — overall ability to distinguish successful vs unsuccessful promotions

From a business perspective, precision helps avoid recommending poor promotions, while recall ensures strong opportunities are not missed. Together, these metrics indicate that the model can support promotion planning decisions with high confidence.

These results are strong for a baseline model on synthetic data and show that the workflow is behaving coherently. While performance is high, this is expected given the synthetic nature of the dataset. Validating the approach on real-world retail data would be a critical next step. The next stage is to harden the training and inference path rather than treat the current score as the final goal.

## How to Run

### Generate The Dataset

If you need to regenerate the synthetic training data:

```bash
python data/generate_data.py
```

This creates `data/promotions.csv` by generating base inputs, simulating promotion outcomes, and assigning `success_label` from ROI.

### Train The Model

Run the baseline training pipeline with:

```bash
python -m src.app.model
```

That command:

- loads `data/promotions.csv`
- builds the approved leakage-safe feature set
- splits the data into train and test sets
- trains a Logistic Regression baseline
- evaluates performance using standard classification metrics
- saves the trained artifact to `models/model.pkl`

If your local environment uses `python3` instead of `python`, use the equivalent command with `python3`.

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- joblib
- FastAPI
- Pydantic
- pytest

## Why This Project Matters

This project is meant to reflect how a product-minded ML system gets built in practice.

It starts with a real business question, defines success in measurable commercial terms, and structures the modeling workflow around a decision that a retailer would actually need to make. For non-technical readers, it shows how predictive models can support earlier and smarter planning. For technical readers, it demonstrates disciplined feature design, explicit leakage prevention, model evaluation, artifact persistence, and a clear path to API-based inference.

In that sense, the project is less about building a flashy model and more about showing sound product thinking, business translation, and production-oriented architecture.

## Next Steps

- Improve model stability by wrapping preprocessing and training in a more robust scaling and pipeline workflow.
- Wire inference to load and use the saved artifact in `models/model.pkl`.
- Replace the current placeholder inference path and placeholder schemas with the real feature contract.
- Expose a prediction endpoint through FastAPI.
- Add tests focused on leakage prevention and feature consistency between training and inference.

## Author

Arad Adler
