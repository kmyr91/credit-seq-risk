# Experimental Results

## 1. Objective

Investigate whether sequence-based neural models provide useful additional information for next-month credit-default prediction.

This is an Affirm-inspired research direction, not a reproduction of Affirm's production modeling system.

## 2. Dataset and experimental setup

UCI Default of Credit Card Clients, DOI: 10.24432/C55S3H.

- 30,000 customers
- Six observations per customer, April through September 2005
- 21,000 train / 4,500 validation / 4,500 test
- Stratified, customer-disjoint random splits with seed 42
- Final test set remains untouched

The GRU inputs are repayment status, bill amount, and payment amount for each of the six months.

The improved tree baseline uses 12 engineered financial features.

## 3. Models and validation results

| Experiment | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| Initial LightGBM | 0.778077 | 0.552829 | 0.136080 |
| Improved LightGBM | 0.780420 | 0.557476 | 0.135606 |
| Standalone GRU | 0.7653 | 0.5384 | 0.1376 |
| Disjoint-stage GRU embedding hybrid | 0.7755 | 0.5414 | 0.1402 |
| Matched 5,250-customer LightGBM control | 0.7752 | 0.5445 | 0.1368 |
| Full-data LightGBM control | 0.780420 | 0.557476 | 0.135606 |
| Full-data GRU logit stack | 0.780442 | 0.556563 | 0.135480 |

The first embedding hybrid trained its encoder on 15,750 customers and its second-stage tree on 5,250 different customers.

The matched control used the same 5,250 second-stage customers, without embeddings.

The final stack used five-fold GRU out-of-fold logits, plus the 12 engineered features, allowing LightGBM to train on all 21,000 customers.

The five GRU fold ROC-AUC values were approximately 0.7617, 0.7710, 0.7808, 0.7721, and 0.7542. Overall training OOF ROC-AUC was 0.7667.

## 4. Paired bootstrap uncertainty

Comparison: full-data GRU logit stack minus full-data LightGBM control.

2,000 paired, class-stratified bootstrap resamples of the same 4,500 validation customers; percentile 95% confidence intervals.

| Metric | Difference | 95% CI |
|---|---:|---|
| ROC-AUC | +0.000022 | [-0.002513, +0.002520] |
| PR-AUC | -0.000913 | [-0.007377, +0.005846] |
| Brier | -0.000127 | [-0.000698, +0.000443] |

Every interval includes zero. Therefore, these results do not show a statistically detectable improvement for the current validation predictions.

The correlation between the *final LightGBM control and stacked model* validation probabilities was 0.99402.

This is NOT the correlation between GRU OOF logits and LightGBM OOF logits, which has not been calculated.

## 5. Interpretation

On this short, single-account dataset, GRU logit stacking did not measurably improve the matched LightGBM baseline.

Possible contributing factors include the limited history length, information already captured by engineered features, and the GRU's predictive quality.

These explanations are hypotheses, not established causal findings.

The result does not imply that neural embeddings cannot help on longer, multi-account credit histories.

## 6. Important methodological limits

- The validation set has been used for model comparison and early stopping; the bootstrap is not an independent confirmatory test.
- Bootstrap intervals condition on the trained models and this validation sample. They do not capture randomness from training or repeated splitting.
- The current tabular baseline has not yet been expanded to include all appropriate raw monthly variables.
- Stacking a scalar GRU logit is different from using learned multidimensional embeddings.
- UCI does not support the hierarchical multi-account representation or AFT survival objective used by Affirm.
- The final held-out test set is reserved for the final project evaluation.

## 7. Follow-up work

1. Strengthen the raw-feature tabular baseline.
2. Investigate licensed longitudinal multi-account data.
3. Build reusable inference, testing, monitoring, and decision-policy components.

Historical disjoint-stage hybrid code is retained under `experiments/archive/`.
