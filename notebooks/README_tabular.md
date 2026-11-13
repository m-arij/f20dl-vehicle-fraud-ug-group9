# Tabular Vehicle Insurance Fraud Detection

## Overview

This component of the F20DL group project investigates vehicle insurance fraud detection using machine learning on tabular claim records.

Dataset: [fraud_oracle](https://www.kaggle.com/datasets/tajamulkhan/fraud-oracle)

- Records: 15,420
- Features: 33 columns, including the target
- Target: FraudFound_P
- Fraud prevalence: 5.99%

## Notebooks

| Notebook | Purpose |

| tabular_eda.ipynb | Data exploration and visualisations |
| tabular_preprocessing.ipynb | Encoding, imputation and data splitting |
| tabular_models.ipynb | Traditional ML, Random Forest and evaluation |
| tabular_mlp.ipynb | Neural network experiments |
| tabular_clustering.ipynb | K-Means clustering and feature selection |

## Setup

Python 3.12 was used for development.

Install dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter ipykernel
```

Download fraud_oracle.csv and place it in:

`data/raw/fraud_oracle.csv`

Run the notebooks using the project's Python environment.

## Preprocessing

An 80/20 stratified train/test split was used.

- Training: 12,336 records
- Testing: 3,084 records
- Zero ages treated as missing
- Median imputation and standard scaling
- One-hot encoding for categorical variables
- Identifier-like columns excluded

Preprocessing was done using training data to reduce data leakage.

## Models and Evaluation

Models included Decision Tree, Logistic Regression, Gaussian Naive Bayes, Random Forest and MLP.

Five-fold stratified cross-validation was used during model development.

K-Means clustering and Random Forest feature-selection experiments were also conducted.

### Final Random Forest Results

The final experiment excluded Fault, PoliceReportFiled and WitnessPresent as a precaution for initial claim screening.

| Metric | Test Result |

| Accuracy | 0.8888 |
| Fraud Precision | 0.1681 |
| Fraud Recall | 0.2162 |
| Fraud F1 | 0.1891 |
| Average Precision | 0.1588 |

Classification threshold: 0.60

The model detected 40 out of 185 fraudulent test claims.

Evaluation figures are saved in `results/tabular/`.

## Limitations

The dataset is imbalanced. The MLP showed overfitting, while K-Means produced weak cluster separation. Some predictors may contain information unavailable at initial claim submission.

## Contribution

Syed Taha Rahman — Tabular EDA, preprocessing, traditional ML, Random Forest tuning, MLP, clustering, feature selection and evaluation.
