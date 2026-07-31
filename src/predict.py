"""Generate validation and December predictions."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.features import build_features, feature_matrix
from src.preprocessing import (
    load_december_inputs,
    load_train_data,
    load_validation_data,
    prepare_raw_frame,
)
from src.utils import (
    BEST_MODEL_PATH,
    DECEMBER_FILE,
    PREDICTIONS_TEMPLATE,
    VALIDATION_PREDICTIONS,
    clip_positive,
)


def load_artifacts() -> tuple[object, object, object]:
    bundle = joblib.load(BEST_MODEL_PATH)
    preprocessor_state = joblib.load(BEST_MODEL_PATH.parent / "preprocessor.joblib")
    return bundle["model"], bundle["encoder_state"], preprocessor_state


def predict_rates(model: object, frame: pd.DataFrame) -> np.ndarray:
    features = feature_matrix(frame).values
    return clip_positive(np.expm1(model.predict(features)))


def build_inference_frame(
    raw_frame: pd.DataFrame,
    preprocessor_state: object,
    encoder_state: object,
    reference_frames: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    prepared = prepare_raw_frame(
        raw_frame,
        preprocessor_state,
        preprocessor_state.weight_medians,
        preprocessor_state.global_weight_median,
        preprocessor_state.city_coords,
    )
    return build_features(prepared, encoder_state)


def generate_validation_predictions() -> pd.DataFrame:
    model, encoder_state, preprocessor_state = load_artifacts()
    validation_raw = load_validation_data()
    train_raw = load_train_data()

    inference = build_inference_frame(
        validation_raw,
        preprocessor_state,
        encoder_state,
        reference_frames=[train_raw],
    )
    predictions = predict_rates(model, inference)

    output = pd.DataFrame(
        {
            "load_id": validation_raw["load_id"],
            "predicted_rate": np.round(predictions, 2),
        }
    )
    output.to_csv(VALIDATION_PREDICTIONS, index=False)
    return output


def generate_december_predictions() -> pd.DataFrame:
    model, encoder_state, preprocessor_state = load_artifacts()
    december_raw = load_december_inputs()
    validation_raw = load_validation_data()
    train_raw = load_train_data()

    inference = build_inference_frame(
        december_raw,
        preprocessor_state,
        encoder_state,
        reference_frames=[train_raw, validation_raw],
    )
    predictions = predict_rates(model, inference)

    output = december_raw.copy()
    output["predicted_rate"] = np.round(predictions, 2)
    output.to_csv(DECEMBER_FILE, index=False)
    return output


def main() -> None:
    validation_output = generate_validation_predictions()
    december_output = generate_december_predictions()
    print(f"Wrote {len(validation_output):,} rows to {VALIDATION_PREDICTIONS}")
    print(f"Wrote {len(december_output):,} rows to {DECEMBER_FILE}")


if __name__ == "__main__":
    main()
