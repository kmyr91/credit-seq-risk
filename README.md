# Credit Sequence Risk

An end-to-end machine learning engineering project for credit-default prediction, inspired by Affirm's work on learned credit representations.

## Research question

Do learned sequence representations improve credit-default prediction beyond a strong gradient-boosted tree baseline?

This project compares engineered financial features, recurrent neural networks, and leakage-controlled GRU prediction stacking.

**Current finding:** On the UCI credit-default dataset, GRU stacking did not provide a statistically detectable improvement over LightGBM.

## Dataset

UCI Default of Credit Card Clients  
https://doi.org/10.24432/C55S3H

- 30,000 customers
- Six monthly observations per customer
- 180,000 records in long format
- Binary next-month default target
- Customer-disjoint stratified split: 70% training, 15% validation, 15% test

The final test set has not yet been evaluated.

Dataset limitations: one account per customer, six fixed monthly observations, and no time-to-event or censoring information.

## Models

| Model | Validation ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| LightGBM baseline | 0.77808 | 0.55283 | 0.13608 |
| Improved LightGBM | 0.78042 | 0.55748 | 0.13561 |
| Standalone GRU | 0.76530 | 0.53840 | 0.13760 |
| GRU + LightGBM stacking | 0.78044 | 0.55656 | 0.13548 |

The final two LightGBM models in the stacking comparison use the same 21,000 training customers.

See [Experimental results](docs/experiments.md) for methodology, uncertainty estimates, and limitations.

## Project structure

```text
src/
  data/          Data validation and preparation
  features/      PySpark customer feature engineering
  models/        LightGBM, GRU, OOF stacking
  evaluation/    Statistical model comparison
tests/           Automated tests
docs/            Experimental and architectural documentation
experiments/     Archived exploratory implementations
data/            Local datasets (not tracked by Git)
outputs/         Local models and predictions (not tracked by Git)
```

## Training workflow

Run commands from the project root in the configured Python environment.

```bash
python -m src.data.prepare_uci
python -m src.features.build_features
python -m src.models.train_lightgbm
python -m src.data.prepare_sequences
python -m src.models.train_gru
python -m src.models.train_gru_oof
python -m src.models.train_stacked_lightgbm
python -m src.evaluation.compare_models
```

The raw UCI Excel file must first be placed in `data/raw/` as expected by the data preparation script.

PySpark requires a compatible Java runtime. The local setup used Java 17.

The GRU encoder and LightGBM stacking stages run as separate processes.

## Experiment tracking

MLflow stores local experiments in `mlflow.db`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://127.0.0.1:5000.

## Testing

```bash
python -m pytest -q
```

## Design and methodological notes

- Fit supervised model components on training data only.
- Each GRU OOF prediction comes from an encoder that did not train on that customer's target.
- GRU preprocessing scalers are fitted separately inside each training fold.
- Validation metrics are not final test metrics.
- Paired bootstrap intervals quantify sampling uncertainty for the current validation predictions, not variation across training runs.
- GRU logit stacking is not a reproduction of Affirm's hierarchical Transformer embedding architecture.

## Next engineering milestones

- Expand automated testing and reproducibility.
- Implement a production-style prediction API.
- Add model validation, monitoring, and CI.
- Simulate approval-rate and default-risk trade-offs.
- Investigate richer, appropriately licensed longitudinal credit datasets.

## Reference

Affirm Engineering, *Learning Richer Credit Representations with Transformers at Affirm*:

https://tech.affirm.com/learning-richer-credit-representations-with-transformers-at-affirm-c774e7c1cea6

This is an independent educational project and is not affiliated with Affirm.
