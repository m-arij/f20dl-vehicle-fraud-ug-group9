# Tabular Vehicle Insurance Fraud Detection

## Overview

This is the tabular component of our F20DL group project. We compare machine learning models for detecting fraudulent vehicle insurance claims.

Dataset: [fraud_oracle](https://www.kaggle.com/datasets/tajamulkhan/fraud-oracle)

- Records: 15,420
- Columns: 33, including the target
- Target: FraudFound_P
- Classes: 0 = non-fraud, 1 = fraud
- Fraud prevalence: approximately 5.99%

The tabular records and the group's image dataset are separate. They are not matched or merged.

## Notebooks

| Notebook | Purpose |
|---|---|
| tabular_eda.ipynb | Explore the data, class balance and unusual values |
| tabular_preprocessing.ipynb | Check splitting, imputation and encoding |
| tabular_models.ipynb | Train traditional models, tune Random Forest and evaluate results |
| tabular_mlp.ipynb | Compare two MLP configurations using cross-validation |
| tabular_clustering.ipynb | Explore clusters and compare feature-selection approaches |

## Setup

Python 3.12 was used during development.

Install the required packages:

```bash
python -m pip install pandas numpy matplotlib seaborn "scikit-learn>=1.7" jupyter ipykernel
```

Scikit-learn 1.7 or newer is needed for the MLP sample-weight support used here.

Place the dataset at:

```text
data/raw/fraud_oracle.csv
```

Open the notebooks from the project's notebooks folder.

Suggested order:

1. tabular_eda.ipynb
2. tabular_preprocessing.ipynb
3. tabular_mlp.ipynb
4. tabular_models.ipynb
5. tabular_clustering.ipynb

Run the MLP notebook before the final model-comparison cell in tabular_models.ipynb. That cell reads the saved MLP scores.

Each modelling notebook creates its own preprocessing pipeline. Fitted objects are not shared between notebook kernels.

## Data Splitting and Preprocessing

An 80/20 stratified train/test split is used with random_state=42.

- Training records: 12,336
- Test records: 3,084
- PolicyNumber and RepNumber are excluded as identifier-like features.
- Categorical features are one-hot encoded.
- Year and DriverRating are treated as categorical features.
- Missing numeric values are filled using the training median.
- Numeric features are standardised in the shared preprocessing pipeline.
- Gaussian Naive Bayes and the MLP receive dense encoded inputs.

Scaling is useful for logistic regression, the MLP and K-Means. Trees do not require scaling, although the shared pipeline also applies it to the tree models.

Preprocessing stays inside the classification pipelines. Each cross-validation fold learns its imputation and encoding from its training records only.

### Missing Ages

There are 320 records with Age recorded as zero:

- 251 in the training set
- 69 in the test set

Zero ages are treated as unknown. An AgeMissing flag records which ages were missing before imputation.

The training median supplies a numeric placeholder, not the person's actual age. The full training set's median recorded age is 39, while individual cross-validation folds calculate their own medians.

All zero-age records have the policyholder age band "16 to 17". Other age bands also disagree with the numeric ages. We therefore avoid assuming an exact replacement age without further documentation.

## Models and Validation

The traditional models are:

- Decision Tree
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest

The neural network experiments compare:

- Original MLP: hidden layers of 64 and 32 neurons, alpha=0.001
- Regularised MLP: hidden layers of 32 and 16 neurons, alpha=0.01

Both networks use scikit-learn's MLPClassifier.

Five-fold stratified cross-validation is used for model development. Class weights or sample weights are used where configured to give fraud records more influence.

### Hyperparameter Tuning

Gaussian Naive Bayes uses GridSearchCV to compare variance-smoothing values.

The original Random Forest uses RandomizedSearchCV with 12 candidate configurations.

A separate reduced-feature Random Forest uses RandomizedSearchCV with 8 candidate configurations. It is tuned separately because its input features differ from the original forest.

Average precision is the search metric. These limited searches do not guarantee globally optimal settings.

Decision Tree and Logistic Regression remain baseline models. The MLP experiment compares two predefined configurations.

## Evaluation

We report:

- Fraud precision
- Fraud recall
- Fraud F1
- Average precision
- Accuracy for context

Average precision summarises precision–recall performance across thresholds. F1 depends on the classification threshold, so the two metrics can change differently.

Development confusion matrices and a comparison chart show differences between models.

The comparison table uses current validation scores rather than manually entered numbers. MLP scores are loaded from:

```text
results/tabular/mlp_cv_comparison.csv
```

Individual MLP fold scores are saved to:

```text
results/tabular/mlp_cv_folds.csv
```

The complete model comparison is saved to:

```text
results/tabular/model_cv_comparison.csv
```

## Final Random Forest Experiment

The final classifier excludes:

- Fault
- PoliceReportFiled
- WitnessPresent

These features may depend on the stage of claim processing. Their availability at initial submission is not confirmed by the supplied CSV.

Excluding them represents a conservative early-screening scenario, rather than proof that they leak the target.

The final classifier uses settings selected by the reduced-feature forest's own search.

Its threshold is selected using the highest F1 among the tested thresholds on out-of-fold training predictions. Test labels are not used for threshold selection.

### Held-Out Test Results

| Metric | Result |
|---|---:|
| Classification threshold | 0.50 |
| Accuracy | 0.8048 |
| Fraud precision | 0.1460 |
| Fraud recall | 0.4649 |
| Fraud F1 | 0.2222 |
| Average precision | 0.1663 |

The model:

- Correctly identified 86 fraudulent claims.
- Missed 99 fraudulent claims.
- Incorrectly flagged 503 non-fraudulent claims.
- Correctly identified 2,396 non-fraudulent claims.

It detects more fraud than the earlier version but also produces more false alerts.

The final evaluation figures are saved to:

```text
results/tabular/rf_confusion_matrix.png
results/tabular/rf_precision_recall.png
```

## MLP Findings

Both MLP configurations were evaluated using the same five stratified folds.

| Configuration | Mean CV F1 | Mean CV Average Precision |
|---|---:|---:|
| Original MLP | 0.1911 | 0.1724 |
| Regularised MLP | 0.1967 | 0.1752 |

The regularised configuration produced slightly higher scores. The differences are small and do not establish a statistically significant advantage.

These are development cross-validation results, not held-out test results. They should not be compared with the final forest's test scores as if they were the same evaluation.

## Clustering

MiniBatchKMeans explores claim structure without using fraud labels during fitting.

Cluster counts from 2 to 6 are compared using sampled silhouette scores. Four clusters produced the highest score in the current experiment:

- Selected K: 4
- Silhouette score: 0.0814

The low score suggests weak separation under the chosen encoding and distance measure.

A two-dimensional TruncatedSVD projection shows a sample of claims and the projected cluster centres. Clustering uses the full encoded feature space, so the plot may hide or distort separation.

Bar charts separately describe cluster sizes and observed fraud rates. Fraud labels are used after clustering to describe the groups, not to create them.

## Feature Selection

The Random Forest in the clustering notebook supports a separate supervised feature-selection experiment. It does not replace the final classifier in tabular_models.ipynb.

A model-based selector is fitted inside each cross-validation fold. The classifier is compared with and without selection using the same folds and fixed classifier settings.

| Approach | Mean CV F1 | Mean CV Average Precision |
|---|---:|---:|
| With feature selection | 0.2444 | 0.2028 |
| Without feature selection | 0.2422 | 0.2149 |

Feature selection slightly increased F1 but reduced average precision. This does not show a clear overall benefit.

Feature-importance scores describe how this forest used the encoded inputs. They do not establish causes of fraud.

This experiment includes Fault, whereas the final reduced-feature classifier excludes it. Their results represent different feature sets.

## Limitations

- Fraud cases are uncommon, and accuracy alone can be misleading.
- The final classifier still misses fraud and produces many false alerts.
- Numeric ages and policyholder age bands contain inconsistencies.
- Gaussian Naive Bayes has distributional limitations with one-hot inputs.
- Hyperparameter searches cover limited settings.
- Tuned-model development scores reuse folds that guided parameter selection.
- The test set was examined during earlier development, so it is not a completely untouched future evaluation.
- Different feature sets limit direct comparisons between some experiments.
- K-Means separation depends on the encoding and distance measure.
- The small MLP score differences should not be overstated.
- These experiments do not establish readiness for deployment.

## Contribution

Syed Taha Rahman — Tabular data exploration, preprocessing, traditional models, Random Forest tuning, MLP experiments, clustering, feature selection and evaluation.