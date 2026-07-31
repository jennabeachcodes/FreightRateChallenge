"""Shared utilities for paths, metrics, and I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train-test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
PREDICTIONS_TEMPLATE = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_FILE = DATA_DIR / "december_chart_inputs.csv"
VALIDATION_PREDICTIONS = PROJECT_ROOT / "validation_predictions.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

INTERNAL_VAL_CUTOFF = "2025-09-15"
MIN_PREDICTED_RATE = 1.0


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / np.maximum(y_true, 1.0))) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def clip_positive(values: np.ndarray | pd.Series, minimum: float = MIN_PREDICTED_RATE) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), minimum)
