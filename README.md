# Vehicle Insurance Fraud Detection

**Group:** Dubai UG Group 9

**Project title:** Detecting fraudulent vehicle insurance claims using machine learning on tabular claim records and claim images

## Group Members

- Fatima Zubair
- Muhammad Arij
- Ahlam Rafati
- Syed Taha Rahman
- Maimuna Sultana

---

## Initial Project Proposal

### Project Topic

Insurance companies get thousands of vehicle claims, and a small number of them are fraud (like staged or exaggerated accidents). In this project we try to predict whether a claim is fraudulent using two separate datasets: a tabular dataset of claim records and an image dataset of labelled fraud and non-fraud claim images.

### Why We Picked It

Fraud makes premiums go up for honest drivers, and companies can only check a few claims by hand, so a model that ranks claims by risk could really help investigators. It is also a good fit for this module:

- The fraud data is imbalanced.
- Most of its features are nominal (good for decision trees, Naïve Bayes and Random Forest).
- The images let us try a CNN.

### Aim

Our aim is to build and fairly compare machine learning models for fraud detection on two kinds of data: a Random Forest and other models on the tabular claim records, and a CNN on the claim images.

### Objectives

- Build a pipeline that downloads and prepares both datasets, so anyone in the group can rerun it and get the same results.
- Explore the data with EDA, visualisations, feature selection and clustering.
- Train and compare at least 3 baseline models, with Random Forest [5] as our main model for the tabular data.
- Build and tune neural networks (an MLP for the claims data and a CNN [7] for the images) and compare them with the baselines.
- Evaluate the models properly, using stratified splits and precision, recall, F1 and PR-AUC [8] instead of just accuracy.

---

## Datasets

| | Dataset 1: fraud_oracle | Dataset 2: Car Insurance Fraud Detection |
|---|---|---|
| **Source and link** | [Kaggle](https://www.kaggle.com/datasets/tajamulkhan/fraud-oracle) [1] | [Kaggle](https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection) [4] |
| **Licence** | MIT | CC0: Public Domain |
| **Type** | Tabular | Image |
| **Size** | About 15,400 claims and 33 features | 6500+ images |
| **Classes** | 2 (fraud / not fraud) | 2 (fraud / non-fraud), provided as train and test folders |
| **Feature types** | Mostly nominal, with a few numeric ones | Raw pixels (colour) |
| **Why we chose it** | Nominal data works well with trees and Bayes, and it is imbalanced like real fraud data | Lets us build and tune a CNN |
| **Prior use** | People have compared lots of classifiers on it in public notebooks. One reports AdaBoost as the best at 95.9% accuracy [3]. | A labelled image dataset for detecting fraudulent car insurance claims, suited to CNN image classification [4]. |

We are treating the two datasets as separate tasks and not merging them, because the images are not linked to the claim records.

---

## Data Summary and Preprocessing

Fraud is only a small part of the tabular claims (about 6%), so the data is really imbalanced.

### Preprocessing Plan

- Check for missing values and duplicates.
- Encode the nominal features and scale the numeric ones.
- Do a stratified train/test split before any resampling.
- Use class weights or resampling (for example SMOTE [6]) on the training data only.
- For the images, use the provided train and test folders, create a validation set from the training images, then resize, normalise and augment the training set only.

> **Note:** It is not fully documented how the tabular dataset was collected, so we will be careful with our results and mention this in the report.

---

## Work Plan (MoSCoW)

| Priority | Tasks |
|---|---|
| **Must (MVP)** | Pipeline and README (R1); EDA and feature selection (R2); 3+ baselines on the claims data: Random Forest (main model), decision tree, naive Bayes, and logistic regression or kNN (R3); MLP on the claims data (R4); CNN on the claim images (R4); evaluation with precision, recall, F1 and PR-AUC |
| **Should** | k-means clustering with evaluation (R2); hyperparameter tuning of Random Forest with cross-validation (R3); compare ways of handling imbalance (class weights vs resampling); augmentation and early stopping for the CNN (R4) |
| **Could** | Boosting (for example AdaBoost); transfer learning for the CNN; feature importance; a fairness check on age and sex |
| **Won't** | Text or generative models; merging the two datasets; making synthetic data |

### Main Risks and How We Will Deal With Them

| Risk | Plan |
|---|---|
| Imbalance | Class weights, and judge models with precision, recall, F1 and PR-AUC |
| Data leakage | Split before preprocessing and fit scalers on training data only |
| Overfitting on images | Augmentation, early stopping and a validation set |
| Bias | Fairness check |
| Reproducibility | One pipeline, one requirements file and fixed random seeds |

---

## Milestones

| Week | Milestone |
|---|---|
| Week 4 | Project pitch |
| Week 5 | Update this README and submit a copy |
| Weeks 5 to 6 | Data pipeline and EDA (R2) |
| Weeks 7 to 8 | Baseline models, including Random Forest (R3) |
| Weeks 8 to 10 | Neural networks: MLP and CNN (R4) |
| Week 10 | Code freeze and report draft |
| Week 11 | Report and code submitted (25 Nov, 15:30) |
| Week 12 | Mini-viva and peer assessment |

---

## Image Dataset (CNN)

This part uses the car image/photos dataset to check if a CNN can tell fraud and non-fraud claim images apart. It is a separate task from the table dataset. Full details (setup, preprocessing, models, parameters and results): [`notebooks/README.md`]

## References

1. Khan, T. *fraud_oracle* [Dataset]. Kaggle. Available at: https://www.kaggle.com/datasets/tajamulkhan/fraud-oracle.
2. Kaggle. Car Insurance Fraud Detection. Available at: https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection?resource=download.
3. Phua, C., Lee, V., Smith, K. and Gayler, R. (2010) *A Comprehensive Survey of Data Mining-based Fraud Detection Research*. arXiv:1009.6119. Available at: https://arxiv.org/abs/1009.6119.
4. Nugroho, T. H. (2022) *Vehicle Insurance Claim Fraud Detection (Comparison of 12 Models)*. BISA AI Academy. Available at: https://bisa.ai/portofolio/detail/Nzg3.
5. pacificrm. *Car Insurance Fraud Detection* [Dataset]. Kaggle. Available at: https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection.
6. Breiman, L. (2001) 'Random Forests', *Machine Learning*, 45(1), pp. 5-32.
7. Chawla, N. V., Bowyer, K. W., Hall, L. O. and Kegelmeyer, W. P. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', *Journal of Artificial Intelligence Research*, 16, pp. 321-357.
8. LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998) 'Gradient-based learning applied to document recognition', *Proceedings of the IEEE*, 86(11), pp. 2278-2324.
9. Saito, T. and Rehmsmeier, M. (2015) 'The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets', *PLoS ONE*, 10(3), e0118432.
10. Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.
