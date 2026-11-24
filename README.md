# Temporal Deep Learning for Credit Card Fraud Detection with Explainable AI

## Project Overview

This project explores temporal deep learning approaches for credit card fraud detection using the Credit Card Fraud Detection dataset.

The main objective is to investigate whether modeling transaction sequences can help detect fraudulent behavior in highly imbalanced financial data. Two temporal architectures, Temporal Transformer and Hybrid LSTM-Transformer, are evaluated with different sequence window sizes.

The experiments also compare these models with Logistic Regression and XGBoost baselines. Explainability methods are applied to the Temporal Transformer model to investigate feature importance, explanation fidelity, and stability.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

- Total transactions: 284,807
- Normal transactions: 284,315
- Fraudulent transactions: 492
- Fraud ratio: approximately 0.173%
- Input features: 30
- Target variable: `Class`

The dataset is ordered by transaction time to preserve temporal structure.

A chronological split is used:
- Training: 70%
- Validation: 15%
- Testing: 15%

The `Amount` feature is standardized using statistics fitted on the training set.

## Methodology

### Temporal Sequence Modeling

Transactions are transformed into sequences using two window sizes:

- Window size 5
- Window size 10

The following architectures are evaluated:

1. Temporal Transformer
2. Hybrid LSTM-Transformer

Both architectures are tested with different sequence lengths to examine the effect of temporal context on fraud detection performance.

### Baseline Models

To provide a comparative evaluation, two additional models are implemented:

- Logistic Regression
- XGBoost

### Model Evaluation

Each model's classification threshold is selected using the validation set. Final performance is then evaluated on the held-out test set.

The evaluation metrics include:

- Average Precision (PR-AUC)
- ROC-AUC
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

## Experimental Results

### Test Set Performance

| Model | Window | PR-AUC | ROC-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| Temporal Transformer | 5 | 0.7386 | 0.9465 | 1.0000 | 0.6923 | 0.8182 |
| Hybrid LSTM-Transformer | 5 | 0.7451 | 0.9761 | 0.9487 | 0.7115 | 0.8132 |
| Temporal Transformer | 10 | 0.7551 | 0.9797 | 0.9487 | 0.7115 | 0.8132 |
| Hybrid LSTM-Transformer | 10 | 0.7289 | 0.9756 | 0.8974 | 0.6731 | 0.6731 |
| Logistic Regression | — | 0.7021 | 0.9750 | 0.9429 | 0.6346 | 0.7586 |
| XGBoost | — | 0.7647 | 0.9704 | 0.8864 | 0.7500 | 0.8125 |

**Key observations:**

- XGBoost achieved the highest test PR-AUC (0.7647).
- Temporal Transformer with a window size of 10 achieved the highest ROC-AUC (0.9797) among the evaluated models.
- Temporal models provided competitive performance compared with the baseline approaches.
- Model performance varied across metrics, highlighting the importance of evaluating fraud detection systems using more than one metric.

## Explainable AI (XAI)

Explainability experiments focus primarily on the Temporal Transformer model with a window size of 10.

The following methods are investigated:

- Integrated Gradients (IG)
- SHAP

### Feature Importance

The most influential features identified by the two methods include:

- Integrated Gradients: V14, V4, V12, V10, V2
- SHAP: V14, V4, V12, V10, V11

### Fidelity Evaluation

Feature deletion experiments are used to investigate how model predictions change when highly ranked features are removed.

The evaluation also compares top-ranked feature deletion against random feature deletion.

### Stability Evaluation

Explanation stability is examined by introducing Gaussian noise at different levels and measuring changes in feature attribution rankings.

Metrics include:

- Cosine Similarity
- Spearman Rank Correlation

The experiments also include reproducibility checks and model randomization tests as additional XAI evaluation procedures.

## Technologies

- Python
- PyTorch
- Scikit-learn
- XGBoost
- SHAP
- NumPy
- Pandas
- Matplotlib
- Google Colab

## Repository Structure

```text
ccfd-temporal-transformer-xai/
│
├── temporal_transformer_xai.ipynb
└── README.md
```

## How to Run

1. Open the notebook in Google Colab.
2. Upload or connect the Credit Card Fraud Detection dataset.
3. Run the notebook cells in order.
4. Review the model comparison and explainability results.

## Notes

- The dataset is highly imbalanced, making PR-AUC an important evaluation metric.
- Classification thresholds are selected using validation data rather than the test set.
- Explainability experiments are primarily conducted on the Temporal Transformer model with a window size of 10.
- The results represent experimental findings on this dataset and should not be interpreted as evidence of performance on every real-world fraud detection system.
