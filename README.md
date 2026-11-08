# Credit Sequence Risk

A machine learning project exploring whether sequence models improve credit-default prediction over gradient-boosted trees.

Inspired by [Affirm's work on credit representations](https://tech.affirm.com/learning-richer-credit-representations-with-transformers-at-affirm-c774e7c1cea6).

## Overview

I worked with the [UCI credit-default dataset](https://doi.org/10.24432/C55S3H): 30,000 customers, six months of payment history, and a binary next-month default target.

The project uses PySpark for feature engineering, LightGBM and PyTorch for modeling, MLflow for experiment tracking, and FastAPI for inference.

## Results

The main question was whether GRU-based sequence representations could improve a LightGBM baseline.

In the original stacking experiment, GRU + LightGBM achieved an ROC-AUC of 0.78044 versus 0.78042 for LightGBM alone. The paired bootstrap confidence interval included zero.

I then compared two LightGBM models with early stopping performed inside the training data:

| Model | Validation ROC-AUC | PR-AUC |
|---|---:|---:|
| LightGBM, 12 engineered features | 0.7801 | 0.5587 |
| LightGBM, 30 features | 0.7803 | 0.5554 |

The extra monthly features did not provide a meaningful improvement, so I kept the simpler model.

**Final held-out test results (4,500 customers):**

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7815 |
| PR-AUC | 0.5548 |
| Brier score | 0.1345 |
| Log loss | 0.4285 |

The test ROC-AUC 95% bootstrap confidence interval was [0.7646, 0.7979].

The raw model was reasonably well calibrated on validation data. Platt scaling gave negligible improvement.

## Underwriting decisions

I also explored a simple approval policy based on predicted default probability.

Under hypothetical assumptions of $100 revenue for a performing loan and $800 loss for a defaulted loan, the break-even default probability is 11.11%.

On policy-validation data, this threshold approved 31.9% of customers, with a 7.5% observed default rate among approved customers.

This corrected economic policy was developed after the original held-out test evaluation, so these policy results are **validation-only**. The original test run used a superseded 12.5% policy with different revenue assumptions.

These calculations illustrate decision economics, not actual lending profitability.

## Engineering

- PySpark feature engineering and customer-level train/validation/test splits
- LightGBM, PyTorch GRU, and out-of-fold stacking
- Probability calibration and bootstrap evaluation
- FastAPI inference with model versioning and prediction IDs
- Feature drift and delayed-outcome monitoring
- Automated testing and GitHub Actions CI

## Running the project

Use Python 3.11, install the dependencies from `pyproject.toml`, and place the UCI Excel dataset in `data/raw/`.

```bash
python -m src.data.prepare_uci
python -m src.features.build_features
python -m src.models.train_lightgbm_fair
python -m src.evaluation.calibration
python -m src.evaluation.decision_economics_v2
```

The final test was evaluated once; its existing saved results should be used rather than rerunning it to tune models or policies.

Run the tests:

```bash
python -m pytest -q
```

Start the API after training the final model:

```bash
python -m uvicorn src.api.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

More details are in [docs/experiments.md](docs/experiments.md).

## Limitations

The dataset contains only six months of single-account history. Random customer splits do not measure future distribution shift, and the project does not solve reject inference or real-world lending profitability. Demographic attributes were excluded from model inputs, but fairness was not established.

This is an independent educational project, not affiliated with Affirm.
