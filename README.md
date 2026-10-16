Vehicle Insurance Fraud Detection

Group: Dubai Group 9 UG

Project title: Detecting fraudulent vehicle insurance claims using machine learning on claim records and damage images

Group members:
Fatima Zubair
Muhammad Arij
Ahlam Rafati
Syed Taha Rahman
Maimuna Sultana

Initial Project Proposal:
1. Project topic
Insurance companies get thousands of vehicle claims, and a small number of them are fraud (like staged or exaggerated accidents). In this project we try to predict whether a claim is fraudulent and we do a separate task where we classify how badly a car is damaged from photos.

Why we picked it: Fraud makes premiums go up for honest drivers, and companies can only check a few claims by hand, so a model that ranks claims by risk could really help investigators. It's also a good fit for this module: the fraud data is imbalanced, most of its features are nominal (good for decision trees and Naïve Bayes), and the images let us try a CNN.

Our aim is to build and fairly compare machine learning models for fraud detection and damage classification.

2. Objectives:
-Build a pipeline that downloads and prepares both datasets, so anyone in the group can rerun it and get the same results.
-Explore the data with EDA, visualisations, feature selection and clustering.
-Train and compare at least 3 baseline models.
-Build and tune neural networks (an MLP for the claims data and a CNN for the images) and compare them with the baselines.
-Evaluate the models properly, using stratified splits and precision, recall, F1 and PR-AUC instead of just accuracy.

3. Datasets
Dataset 1 : Vehicle Claims Fraud Detection	
Source and link	Kaggle: https://www.kaggle.com/datasets/tajamulkhan/fraud-oracle
Licence : MIT 
Size: About 15,400 claims and 33 features	
Classes: 	2 (fraud / not fraud)

Dataset 2 : Car Severity Image
Source and Link Kaggle:https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection?resource=download
Licence : CC0: Public Domain
Size :2,300+ images
Classes:3 (minor / moderate / severe)

Feature types	Mostly nominal, with a few numeric ones	Raw pixels (colour)
Why we chose it	: Nominal data works well with trees and Bayes, and it's imbalanced like real fraud data	Lets us build and tune a CNN
Prior use : People have compared lots of classifiers on it in public notebooks. One reports AdaBoost as the best at 95.9% accuracy [3].	Damage severity classification is a common public task, e.g. on Roboflow Universe .

We're treating the two datasets as separate tasks and not merging them, because the images aren't linked to the claim records.

4. Data summary and preprocessing:
Fraud is only a small part of the claims (about 6%, so the data is really imbalanced. 
Our preprocessing plan:
-Check for missing values and duplicates.
-Encode the nominal features and scale the numeric ones.
-Do a stratified train/test split before any resampling.
-Use class weights or resampling on the training data only.
-For the images, resize, normalise and augment the training set.
It isn't fully documented how the tabular dataset was collected, so we'll be careful with our results and mention this in the report.

6. Work plan (MoSCoW):
Priority	Tasks:
Must (MVP): Pipeline and README (R1); EDA and feature selection (R2); 3+ baselines: decision tree, naive Bayes, and logistic regression or kNN (R3); MLP on the claims data (R4); evaluation with precision, recall, F1 and PR-AUC
Should: k-means clustering with evaluation (R2); CNN on the damage images (R4); compare ways of handling imbalance (class weights vs resampling)
Could: Random forest or boosting; hyperparameter tuning; feature importance; a fairness check on age and sex
Won't: Text or generative models; merging the two datasets; making synthetic data

Main risks and how we'll deal with them: imbalance (class weights), data leakage (split before preprocessing), bias (fairness check) and reproducibility (one pipeline, one requirements file and fixed random seeds).

6. Milestones:
Week 4:	Project pitch
Week 5:	Update this README and submit a copy
Week 5 to 6: Data pipeline and EDA (R2)
Week 7 to 8:	Baseline models (R3)
Week 8 to 10:	Neural networks (R4)
Week 10	:Code freeze and report draft
Week 11	:Report and code submitted (25 Nov, 15:30)
Week 12	:Mini-viva and peer assessment


7. References
Kaggle. Vehicle Claim Fraud Detection (shivamb). https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
Phua, Lee, Smith and Gayler (2010). A Comprehensive Survey of Data Mining-based Fraud Detection Research. arXiv:1009.6119
Nugroho, T. H. Vehicle Insurance Claim Fraud Detection (comparison of 12 models). https://bisa.ai/portofolio/detail/Nzg3
Roboflow Universe. car-damage-severity. https://universe.roboflow.com/sinfo/car-damage-severity
Kaggle. Car damage severity images. TODO add exact link
