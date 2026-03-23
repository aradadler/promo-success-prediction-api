"""Training and loading helpers for the promotion success model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURE_COLUMNS, build_features


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "promotions.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.pkl"
TARGET_COLUMN = "success_label"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_training_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the persisted training dataset."""
    return pd.read_csv(data_path)


def prepare_training_data(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build approved model features and extract the training target."""
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = build_features(dataset)
    y = dataset[TARGET_COLUMN].copy()
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit a scaled logistic regression baseline in one reproducible pipeline."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute standard classification metrics for the held-out split."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def save_model(model: Pipeline, model_path: Path = MODEL_PATH) -> None:
    """Persist the trained artifact for later inference use."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the persisted trained model artifact."""
    return joblib.load(model_path)


def run_training() -> Pipeline:
    """Train, evaluate, and persist the baseline model."""
    dataset = load_training_data()
    X, y = prepare_training_data(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)

    class_balance = y.value_counts(normalize=True).sort_index()

    print(f"Dataset size: {len(dataset)} rows")
    print(f"Class balance: {class_balance.to_dict()}")
    print(f"Feature names used: {MODEL_FEATURE_COLUMNS}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved model to: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    run_training()
