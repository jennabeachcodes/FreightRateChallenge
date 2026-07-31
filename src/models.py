"""Model definitions, training helpers, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import regression_metrics


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    estimator: object


def get_model_candidates() -> dict[str, object]:
    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=5.0, random_state=42)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.06,
            max_iter=300,
            l2_regularization=0.1,
            random_state=42,
        ),
    }


def train_and_evaluate(
    name: str,
    estimator: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> ModelResult:
    estimator.fit(x_train, y_train)
    predictions = np.expm1(estimator.predict(x_val))
    metrics = regression_metrics(np.expm1(y_val), predictions)
    return ModelResult(name=name, metrics=metrics, estimator=estimator)


def select_best_model(results: list[ModelResult]) -> ModelResult:
    return min(results, key=lambda result: result.metrics["rmse"])
