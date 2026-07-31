"""Feature engineering and encoding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "distance",
    "weight_clean",
    "weight_missing",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "day_of_week",
    "month",
    "day_of_year",
    "sin_doy",
    "cos_doy",
    "equipment_dry_van",
    "equipment_reefer",
    "equipment_flatbed",
    "lane_encoded",
    "pickup_encoded",
    "delivery_encoded",
    "market_x_quote",
    "log_distance",
    "log_weight",
]


@dataclass
class FeatureEncoderState:
    lane_map: dict[str, float] = field(default_factory=dict)
    pickup_map: dict[str, float] = field(default_factory=dict)
    delivery_map: dict[str, float] = field(default_factory=dict)
    global_mean: float = 0.0
    smoothing: float = 20.0


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["date"])
    result["day_of_week"] = dates.dt.dayofweek
    result["month"] = dates.dt.month
    result["day_of_year"] = dates.dt.dayofyear
    result["sin_doy"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["cos_doy"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    return result


def add_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_x_quote"] = result["market_index"] * result["quote_signal"]
    result["log_distance"] = np.log1p(result["distance"])
    result["log_weight"] = np.log1p(result["weight_clean"])
    return result


def add_equipment_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["equipment_dry_van"] = (result["equipment"] == "Dry Van").astype(int)
    result["equipment_reefer"] = (result["equipment"] == "Reefer").astype(int)
    result["equipment_flatbed"] = (result["equipment"] == "Flatbed").astype(int)
    return result


def _smoothed_target_map(
    categories: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float,
) -> dict[str, float]:
    grouped = pd.DataFrame({"category": categories, "target": target}).groupby("category")["target"]
    stats = grouped.agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    return smoothed.to_dict()


def fit_target_encoders(train: pd.DataFrame, target_col: str = "posted_rate") -> FeatureEncoderState:
    target = np.log1p(train[target_col])
    global_mean = float(target.mean())
    return FeatureEncoderState(
        lane_map=_smoothed_target_map(train["lane"], target, global_mean, smoothing=20.0),
        pickup_map=_smoothed_target_map(train["pickup"], target, global_mean, smoothing=20.0),
        delivery_map=_smoothed_target_map(train["delivery"], target, global_mean, smoothing=20.0),
        global_mean=global_mean,
    )


def apply_target_encoders(frame: pd.DataFrame, state: FeatureEncoderState) -> pd.DataFrame:
    result = frame.copy()
    result["lane_encoded"] = result["lane"].map(state.lane_map).fillna(state.global_mean)
    result["pickup_encoded"] = result["pickup"].map(state.pickup_map).fillna(state.global_mean)
    result["delivery_encoded"] = result["delivery"].map(state.delivery_map).fillna(state.global_mean)
    return result


def build_features(frame: pd.DataFrame, encoder_state: FeatureEncoderState | None = None) -> pd.DataFrame:
    result = add_date_features(frame)
    result = add_numeric_features(result)
    result = add_equipment_features(result)
    if encoder_state is not None:
        result = apply_target_encoders(result, encoder_state)
    return result


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS].astype(float)
