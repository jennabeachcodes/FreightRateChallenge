# Freight Rate Prediction

A machine learning pipeline for predicting trucking load rates from lane, equipment,
market, and temporal features. Given historical posted rates, it trains and compares
several regression models, then uses the best one to score a held-out validation set
and a fixed December scenario.

## Project structure

```
data/                          Raw CSV inputs
src/
  preprocessing.py             Loading, cleaning, imputation, temporal split
  features.py                  Feature engineering and target encoding
  models.py                    Model definitions and evaluation
  train.py                     EDA, model training, and selection
  predict.py                   Generate submission predictions
  utils.py                     Paths, metrics, helpers
models/                        Saved model artifacts
reports/                       EDA plots and model comparison
score.py                       Submission validator and December chart
validation_predictions.csv     Final validation-set predictions (generated)
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Pipeline

### 1. Train

```bash
python -m src.train
```

Trains four regression models on a temporal split of `train-test.csv`
(train on data before 2025-09-15, hold out Sep–Oct 2025 for internal
evaluation), compares MAE/RMSE/MAPE on that internal holdout, and saves
the best-performing model. This internal split is separate from
`validation.csv` (Nov–Dec 2025), which is only scored in the Predict step
below.

Outputs:

| File | Description |
|---|---|
| `reports/eda_overview.png` | Exploratory plots: rate distribution, rate vs. distance, rate by equipment, daily median rate |
| `reports/eda_summary.json` | Dataset summary stats (row count, date range, rate mean/std, missing weights, equipment counts) |
| `reports/model_comparison.csv` | Validation metrics for every candidate model, sorted by RMSE |
| `models/best_model.joblib` | Best estimator, plus its fitted target-encoder state |
| `models/preprocessor.joblib` | Fitted imputation values and city/market lookup tables |

### 2. Predict

```bash
python -m src.predict
```

Loads the saved model and preprocessing state, rebuilds features the same
way as training, and scores:

- the held-out validation set → `validation_predictions.csv`
- the fixed December scenario → `data/december_chart_inputs.csv` (with `predicted_rate` filled in)

### 3. Validate submission

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

Checks both output files for correct schema, row counts, and value ranges,
then renders `scorer_results/candidate_december.png`.

## Approach

**Split**: Time-based holdout at September 15, 2025, so validation mimics
forecasting into an unseen future period (the same setup used for the
Nov–Dec target).

**Cleaning**: Non-positive or missing weights are treated as unknown and imputed with the equipment-specific median (falling back to a global median
for unseen equipment types). Missing pickup/delivery coordinates and market
signals are filled from lookup tables built across train and validation data.

**Features**
- Distance and weight (raw and log-transformed)
- Market index and quote signal, plus their interaction term
- Cyclical date features (day of week, month, sine/cosine of day-of-year)
- One-hot equipment indicators (Dry Van, Reefer, Flatbed)
- Smoothed target encodings for lane, pickup city, and delivery city —
  fit on the training split only, to prevent leakage into validation metrics

**Target**: Models train on `log1p(posted_rate)`; predictions are
converted back with `expm1` and floored at a small positive minimum.

**Models compared**: Ridge (scaled), Random Forest, Gradient Boosting, and
HistGradientBoosting. The model with the lowest validation RMSE is selected
and saved.

## Known limitations

- **Target encoding is in-fold.** Lane/city smoothed means are fit and
  applied to the same training rows (not via out-of-fold cross-fitting).
  Smoothing (weight=20) limits this for low-count lanes, but it's a known
  simplification rather than a fully leakage-free encoding.
- **No fallback for unseen cities.** If a pickup/delivery city isn't in the
  coordinate lookup built from train/validation data, `feature_matrix()`
  will pass NaN coordinates straight into the model. Not an issue for the
  current validation/December cities, but worth guarding against before
  running on new data.

## Licence

This project is for Spotter AI assessment purposes only.
