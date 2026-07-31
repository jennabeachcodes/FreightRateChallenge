"""Data loading, cleaning, and temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils import (
    DATA_DIR,
    DECEMBER_FILE,
    INTERNAL_VAL_CUTOFF,
    TRAIN_FILE,
    VALIDATION_FILE,
)


@dataclass
class PreprocessorState:
    """Fitted values used to transform inference data consistently."""

    weight_medians: dict[str, float] = field(default_factory=dict)
    global_weight_median: float = 0.0
    city_coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    daily_market_index: dict[str, float] = field(default_factory=dict)
    daily_quote_signal: dict[str, float] = field(default_factory=dict)
    global_market_index: float = 0.0
    global_quote_signal: float = 0.0


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_FILE, parse_dates=["date"])


def load_validation_data() -> pd.DataFrame:
    return pd.read_csv(VALIDATION_FILE, parse_dates=["date"])


def load_december_inputs() -> pd.DataFrame:
    return pd.read_csv(DECEMBER_FILE, parse_dates=["date"])


def clean_weight(series: pd.Series) -> pd.Series:
    """Treat missing and negative weights as unknown."""
    cleaned = pd.to_numeric(series, errors="coerce")
    return cleaned.where(cleaned > 0)


def fit_weight_imputation(train: pd.DataFrame) -> tuple[dict[str, float], float]:
    cleaned = clean_weight(train["weight"])
    medians = cleaned.groupby(train["equipment"]).median().to_dict()
    global_median = float(cleaned.median())
    return medians, global_median


def impute_weight(
    frame: pd.DataFrame,
    medians: dict[str, float],
    global_median: float,
) -> pd.Series:
    cleaned = clean_weight(frame["weight"])
    filled = cleaned.copy()
    for equipment, median in medians.items():
        mask = filled.isna() & frame["equipment"].eq(equipment)
        filled.loc[mask] = median
    filled = filled.fillna(global_median)
    return filled


def build_city_coordinate_lookup(frames: list[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    lookup: dict[str, tuple[float, float]] = {}
    for frame in frames:
        for role, lat_col, lon_col in (
            ("pickup", "pickup_lat", "pickup_lon"),
            ("delivery", "delivery_lat", "delivery_lon"),
        ):
            cities = frame[[role, lat_col, lon_col]].dropna().drop_duplicates(subset=[role])
            for _, row in cities.iterrows():
                lookup.setdefault(row[role], (float(row[lat_col]), float(row[lon_col])))
    return lookup


def attach_coordinates(frame: pd.DataFrame, lookup: dict[str, tuple[float, float]]) -> pd.DataFrame:
    result = frame.copy()
    for city_col, lat_col, lon_col in (
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ):
        if lat_col not in result.columns:
            result[lat_col] = np.nan
        if lon_col not in result.columns:
            result[lon_col] = np.nan
        for city, (lat, lon) in lookup.items():
            mask = result[city_col].eq(city)
            result.loc[mask, lat_col] = lat
            result.loc[mask, lon_col] = lon
    return result


def fit_daily_market_signals(frames: list[pd.DataFrame]) -> PreprocessorState:
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    daily = combined.groupby(combined["date"].dt.date).agg(
        market_index=("market_index", "median"),
        quote_signal=("quote_signal", "median"),
    )
    state = PreprocessorState(
        daily_market_index={str(idx): val for idx, val in daily["market_index"].items()},
        daily_quote_signal={str(idx): val for idx, val in daily["quote_signal"].items()},
        global_market_index=float(combined["market_index"].median()),
        global_quote_signal=float(combined["quote_signal"].median()),
    )
    return state


def attach_market_signals(frame: pd.DataFrame, state: PreprocessorState) -> pd.DataFrame:
    result = frame.copy()
    if "market_index" not in result.columns:
        result["market_index"] = np.nan
    if "quote_signal" not in result.columns:
        result["quote_signal"] = np.nan

    dates = pd.to_datetime(result["date"]).dt.date.astype(str)
    result["market_index"] = result["market_index"].fillna(dates.map(state.daily_market_index))
    result["quote_signal"] = result["quote_signal"].fillna(dates.map(state.daily_quote_signal))
    result["market_index"] = result["market_index"].fillna(state.global_market_index)
    result["quote_signal"] = result["quote_signal"].fillna(state.global_quote_signal)
    return result


def temporal_split(frame: pd.DataFrame, cutoff: str = INTERNAL_VAL_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_date = pd.Timestamp(cutoff)
    train = frame.loc[frame["date"] < cutoff_date].copy()
    validation = frame.loc[frame["date"] >= cutoff_date].copy()
    return train, validation


def prepare_raw_frame(
    frame: pd.DataFrame,
    state: PreprocessorState,
    weight_medians: dict[str, float],
    global_weight_median: float,
    city_lookup: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    result = attach_coordinates(frame, city_lookup)
    result = attach_market_signals(result, state)
    result["weight_clean"] = impute_weight(result, weight_medians, global_weight_median)
    result["weight_missing"] = clean_weight(frame["weight"]).isna().astype(int)
    result["lane"] = result["pickup"].astype(str) + "->" + result["delivery"].astype(str)
    return result


def fit_preprocessor(train: pd.DataFrame, auxiliary_frames: list[pd.DataFrame] | None = None) -> PreprocessorState:
    auxiliary_frames = auxiliary_frames or []
    weight_medians, global_weight_median = fit_weight_imputation(train)
    city_lookup = build_city_coordinate_lookup([train, *auxiliary_frames])
    market_state = fit_daily_market_signals([train, *auxiliary_frames])

    return PreprocessorState(
        weight_medians=weight_medians,
        global_weight_median=global_weight_median,
        city_coords=city_lookup,
        daily_market_index=market_state.daily_market_index,
        daily_quote_signal=market_state.daily_quote_signal,
        global_market_index=market_state.global_market_index,
        global_quote_signal=market_state.global_quote_signal,
    )
