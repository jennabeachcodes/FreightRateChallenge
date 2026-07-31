# Freight Rate Prediction

Machine learning pipeline for predicting trucking load rates from lane, equipment, market, and temporal features.

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
validation_predictions.csv     Final validation-set predictions
score.py                       Submission validator and December chart
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train

Trains four regression models on a temporal split (train before 2025-09-15, validate on Sep–Oct 2025), compares MAE/RMSE/MAPE, and saves the best model to `models/`.

```bash
python -m src.train
```

Outputs:

- `reports/eda_overview.png` — exploratory plots
- `reports/eda_summary.json` — dataset summary stats
- `reports/model_comparison.csv` — validation metrics by model
- `models/best_model.joblib` — best estimator + encoder state
- `models/preprocessor.joblib` — imputation and lookup tables

## Predict

Generates scored outputs for the held-out validation set and December chart inputs.

```bash
python -m src.predict
```

Outputs:

- `validation_predictions.csv`
- `data/december_chart_inputs.csv` (with `predicted_rate` filled)

## Validate submission

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

This checks file format and creates `scorer_results/candidate_december.png`.

## Approach

- **Split:** Time-based holdout using September 15, 2025 as cutoff to mimic the Nov–Dec forecast period.
- **Cleaning:** Negative or missing weights are imputed by equipment median.
- **Features:** Distance, market signals, cyclical date features, equipment indicators, smoothed target encodings for lane/cities, and interaction terms.
- **Target:** Models train on `log1p(posted_rate)` and predictions are transformed back with `expm1`.
- **Models compared:** Ridge, Random Forest, Gradient Boosting, HistGradientBoosting (best selected by RMSE).

## Licence
This project is for Spotter AI assessment purposes only. 
