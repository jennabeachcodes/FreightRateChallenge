"""Training pipeline with EDA and model selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import build_features, feature_matrix, fit_target_encoders
from src.models import get_model_candidates, select_best_model, train_and_evaluate
from src.preprocessing import (
    fit_preprocessor,
    load_train_data,
    load_validation_data,
    prepare_raw_frame,
    temporal_split,
)
from src.utils import (
    BEST_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    PREPROCESSOR_PATH,
    REPORTS_DIR,
    ensure_directories,
    regression_metrics,
)


@dataclass
class TrainingArtifacts:
    preprocessor_state: object
    encoder_state: object
    best_model_name: str
    best_model: object


def run_eda(train: pd.DataFrame, reports_dir: Path | None = None) -> None:
    reports_dir = reports_dir or REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(train["posted_rate"], bins=60, color="#064A56", alpha=0.85)
    axes[0, 0].set_title("Posted Rate Distribution")
    axes[0, 0].set_xlabel("Posted rate ($)")

    axes[0, 1].scatter(train["distance"], train["posted_rate"], s=8, alpha=0.25, color="#1F7A8C")
    axes[0, 1].set_title("Rate vs Distance")
    axes[0, 1].set_xlabel("Distance (miles)")
    axes[0, 1].set_ylabel("Posted rate ($)")

    equipment_means = train.groupby("equipment")["posted_rate"].mean().sort_values()
    axes[1, 0].bar(equipment_means.index, equipment_means.values, color="#2EC4B6")
    axes[1, 0].set_title("Average Rate by Equipment")
    axes[1, 0].set_ylabel("Posted rate ($)")

    daily = train.groupby(train["date"].dt.date)["posted_rate"].median()
    axes[1, 1].plot(daily.index, daily.values, color="#E76F51", linewidth=2)
    axes[1, 1].set_title("Median Daily Rate Over Time")
    axes[1, 1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(reports_dir / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    missing_weight = train["weight"].isna().sum() + (pd.to_numeric(train["weight"], errors="coerce") <= 0).sum()
    summary = {
        "rows": int(len(train)),
        "date_min": str(train["date"].min().date()),
        "date_max": str(train["date"].max().date()),
        "posted_rate_mean": float(train["posted_rate"].mean()),
        "posted_rate_std": float(train["posted_rate"].std()),
        "missing_or_invalid_weight": int(missing_weight),
        "equipment_counts": train["equipment"].value_counts().to_dict(),
    }
    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def prepare_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, object]:
    raw_train = load_train_data()
    validation_reference = load_validation_data()

    preprocessor_state = fit_preprocessor(raw_train, auxiliary_frames=[validation_reference])
    prepared = prepare_raw_frame(
        raw_train,
        preprocessor_state,
        preprocessor_state.weight_medians,
        preprocessor_state.global_weight_median,
        preprocessor_state.city_coords,
    )
    train_split, val_split = temporal_split(prepared)
    encoder_state = fit_target_encoders(train_split)

    train_features = build_features(train_split, encoder_state)
    val_features = build_features(val_split, encoder_state)
    return train_features, val_features, raw_train, preprocessor_state, encoder_state


def train_models() -> TrainingArtifacts:
    ensure_directories()
    train_features, val_features, raw_train, preprocessor_state, encoder_state = prepare_datasets()
    run_eda(raw_train)

    x_train = feature_matrix(train_features).values
    y_train = np.log1p(train_features["posted_rate"].values)
    x_val = feature_matrix(val_features).values
    y_val = np.log1p(val_features["posted_rate"].values)

    results = []
    for name, estimator in get_model_candidates().items():
        result = train_and_evaluate(name, estimator, x_train, y_train, x_val, y_val)
        results.append(result)
        print(f"{name:>24}  MAE={result.metrics['mae']:,.2f}  RMSE={result.metrics['rmse']:,.2f}  MAPE={result.metrics['mape']:.2f}%")

    best = select_best_model(results)
    comparison = pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values("rmse")
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    joblib.dump(preprocessor_state, PREPROCESSOR_PATH)
    joblib.dump(
        {
            "model": best.estimator,
            "encoder_state": encoder_state,
            "model_name": best.name,
        },
        BEST_MODEL_PATH,
    )

    print(f"\nBest model: {best.name}")
    print(f"Saved model to {BEST_MODEL_PATH}")
    print(f"Saved comparison to {MODEL_COMPARISON_PATH}")

    return TrainingArtifacts(
        preprocessor_state=preprocessor_state,
        encoder_state=encoder_state,
        best_model_name=best.name,
        best_model=best.estimator,
    )


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
