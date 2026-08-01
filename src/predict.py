"""Generate validation and December predictions."""

from __future__ import annotations

import joblib  # for loading saved model/preprocessor artifacts
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
    # Load the trained model bundle (model + target-encoder state)
    bundle = joblib.load(BEST_MODEL_PATH)
    # Load the fitted preprocessor state saved alongside the model
    preprocessor_state = joblib.load(BEST_MODEL_PATH.parent / "preprocessor.joblib")
    return bundle["model"], bundle["encoder_state"], preprocessor_state


def predict_rates(model: object, frame: pd.DataFrame) -> np.ndarray:
    # Select and cast the feature columns the model expects
    features = feature_matrix(frame).values
    # Predict on log scale, invert with expm1, then clip to positive rates
    return clip_positive(np.expm1(model.predict(features)))


def build_inference_frame(
    raw_frame: pd.DataFrame,
    preprocessor_state: object,
    encoder_state: object,
    reference_frames: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    # Clean and impute raw input using the state fitted on training data
    prepared = prepare_raw_frame(
        raw_frame,
        preprocessor_state,
        preprocessor_state.weight_medians,
        preprocessor_state.global_weight_median,
        preprocessor_state.city_coords,
    )
    # Apply feature engineering and target encoding to the cleaned frame
    return build_features(prepared, encoder_state)


def generate_validation_predictions() -> pd.DataFrame:
    # Load the trained model and fitted preprocessing/encoding state
    model, encoder_state, preprocessor_state = load_artifacts()
    # Load the validation set to score
    validation_raw = load_validation_data()
    # Load training data, kept as reference context for preprocessing
    train_raw = load_train_data()

    # Build model-ready features for the validation set
    inference = build_inference_frame(
        validation_raw,
        preprocessor_state,
        encoder_state,
        reference_frames=[train_raw],
    )
    # Run the model to get predicted rates
    predictions = predict_rates(model, inference)

    # Pair each prediction with its load_id, rounded to cents
    output = pd.DataFrame(
        {
            "load_id": validation_raw["load_id"],
            "predicted_rate": np.round(predictions, 2),
        }
    )
    # Save predictions to disk
    output.to_csv(VALIDATION_PREDICTIONS, index=False)
    return output


def generate_december_predictions() -> pd.DataFrame:
    # Load the trained model and fitted preprocessing/encoding state
    model, encoder_state, preprocessor_state = load_artifacts()
    # Load the December inputs to predict on
    december_raw = load_december_inputs()
    # Load validation and training data as reference context
    validation_raw = load_validation_data()
    train_raw = load_train_data()

    # Build model-ready features for the December set
    inference = build_inference_frame(
        december_raw,
        preprocessor_state,
        encoder_state,
        reference_frames=[train_raw, validation_raw],
    )
    # Run the model to get predicted rates
    predictions = predict_rates(model, inference)

    # Keep all original December columns and append predictions
    output = december_raw.copy()
    output["predicted_rate"] = np.round(predictions, 2)
    # Save predictions to disk
    output.to_csv(DECEMBER_FILE, index=False)
    return output


def main() -> None:
    # Run both prediction pipelines and report how many rows were written
    validation_output = generate_validation_predictions()
    december_output = generate_december_predictions()
    print(f"Wrote {len(validation_output):,} rows to {VALIDATION_PREDICTIONS}")
    print(f"Wrote {len(december_output):,} rows to {DECEMBER_FILE}")


if __name__ == "__main__":
    main()
