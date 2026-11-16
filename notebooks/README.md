# Image Dataset: Fraud Detection with a CNN

This folder contains the image part of our project. The goal is to see if a Convolutional Neural Network (CNN) can tell fraudulent and non-fraudulent car insurance claim photos apart, and how it compares with classical machine learning models.

- `image_eda.ipynb`: exploring and checking the image dataset
- `image_cnn.ipynb`: preprocessing, training the CNNs, evaluation and comparison with other models

## 1. Setup

### Why Python 3.12
We first used Python 3.14, but TensorFlow does not have a stable release for it yet, so pip installed a test version (2.22.0rc0). We switched to Python 3.12 so everyone in the group (and the markers) gets the same stable TensorFlow version (2.21.0). All library versions are pinned in `requirements.txt` so the results can be repeated.

### How to run
1. Create a virtual environment with Python 3.12 and install the libraries:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Download the dataset (see `data/README.md`) and change `DATA_DIR` in the first cell of each notebook.
3. Run `image_eda.ipynb` from top to bottom.
4. Run `image_cnn.ipynb` from top to bottom. Training both CNNs takes about 40 minutes on a CPU but The trained models are included in models/. To skip training, run Cells 1–4, skip the two training cells, and run from the load cell onwards. The trained models are saved to `models/`

All random seeds are fixed (`SEED = 42`) so results stay the same between runs.

## 2. The dataset

[Car Insurance Fraud Detection](https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection) on Kaggle: 6,616 photos of damaged cars labelled Fraud or Non-Fraud, already split into train and test folders.

| Split | Fraud | Non-Fraud | Fraud % |
|---|---|---|---|
| Train | 200 | 5,000 | 3.8% |
| Test | 93 | 1,323 | 6.6% |

What we found in EDA (`image_eda.ipynb`):
- The data is **very imbalanced**, so accuracy is not a useful metric.
- Fraud and non-fraud photos **look very similar** (both show damaged cars), so the task is hard.
- **0 broken files** and **0 duplicate images** (checked with MD5 hashes), so there is no data leakage between train and test.
- There are **502 different image sizes** (800–1500 × 600–1125 pixels), so all images need resizing.
- Fraud and non-fraud images have almost the **same average size**, so the model cannot use image size as a shortcut.

## 3. Preprocessing

| Step | What we did | Why |
|---|---|---|
| Validation split | 15% of train, stratified (train 4,420 / validation 780 / test 1,416) | The dataset has no validation set. We need one to make decisions without touching the test set. Stratified keeps the same fraud ratio in both parts. |
| Labels | Fraud = 1, Non-Fraud = 0 | Fraud is the class we want to catch, so precision and recall are calculated for it. |
| Resize | 128 × 128 pixels | A CNN needs all images the same size. 128 is small enough to train on a laptop CPU. |
| Normalise | Pixel values from 0–255 to 0–1 | Neural networks train more stably with small input values. |
| Augmentation | Random horizontal flip, rotation up to ~18° (0.05), brightness ±10% | Only 170 fraud images in training, so small random changes help stop the CNN memorising them. Applied to training images only. |
| Caching and batching | Images cached in memory, batches of 32 | Faster training after the first epoch. |

## 4. Handling the imbalance: CNN v1 and CNN v2

Fraud is only about 4% of the training data, so a model could just say "non-fraud" for everything and still be 96% accurate. To deal with this we trained **two versions of the same CNN** and compared them. The architecture is identical; only the way the imbalance is handled changes, so the comparison is fair.

| | CNN v1 | CNN v2 |
|---|---|---|
| Method | **Class weights** | **Oversampling** |
| How it works | Fraud mistakes count 13× in the loss, non-fraud 0.52× (`total / (2 × class count)`) | Training batches are built to be about 50% fraud, by repeating the fraud images |
| Class weights used | Yes | No |

## 5. CNN architecture and parameters

| Layer / setting | Value | Why |
|---|---|---|
| Input | 128 × 128 × 3 (RGB) | Matches the resized images |
| Augmentation | Inside the model as the first layer | Keras only applies it during training, not on validation or test |
| Convolution blocks | 3 blocks with 32, 64 and 128 filters | Early layers find simple patterns (edges), later layers find bigger ones (dents, cracks). Filters grow as the image gets smaller. |
| Filter size | 3 × 3, `padding="same"` | Standard small filter; padding keeps the image size the same inside each block |
| Activation | ReLU | Lets the network learn non-linear patterns |
| Batch normalisation | After each convolution | Keeps values in a stable range so training is smoother |
| Max pooling | 2 × 2 after each block | Halves the image size (128 → 64 → 32 → 16), keeps the strongest features |
| Global average pooling | 1 value per filter (128 values) | Far fewer parameters than `Flatten`, which helps avoid overfitting with a small dataset |
| Dense hidden layer | 64 units, ReLU | Combines the 128 features before the final decision |
| Dropout | 0.3 | Randomly switches off 30% of units during training to reduce overfitting |
| Output | 1 unit, sigmoid | Gives the probability that an image is fraud (0–1) |
| Total parameters | 102,465 | Small model that suits a laptop CPU and 4,420 training images |

### Training settings

| Setting | Value | Why |
|---|---|---|
| Optimiser | Adam, learning rate 0.001 | Common default that works well without much tuning |
| Loss | Binary cross-entropy | Standard loss for a two-class problem with a sigmoid output |
| Metrics during training | Precision, recall, PR-AUC | Suitable for imbalanced data (accuracy would be misleading) |
| Max epochs | 30 | Upper limit; early stopping usually ends training sooner |
| Early stopping | Watches validation PR-AUC, patience 5, restores best weights | Stops when the model stops improving on validation, to avoid overfitting |
| Steps per epoch (v2 only) | 138 | v2's training data repeats forever, so we set an epoch to the same size as v1's |

## 6. Choosing the threshold

The CNN outputs a probability. By default anything above 0.5 counts as fraud, but with imbalanced data 0.5 is usually not the best cut-off. For each model we tried every threshold on the **validation set** and picked the one with the **best F1**:

- CNN v1: threshold 0.86
- CNN v2: threshold 0.98 (high because v2 was trained on 50% fraud batches, so its probabilities are pushed up)

The test set was **only used once**, at the end, so the final scores are not biased by any of our choices.

## 7. Comparison with classical models

To compare the CNN with other techniques, we turned each image into **54 colour features** (a 16-bin histogram plus the mean and standard deviation for each of the red, green and blue channels) and trained:

| Model | Settings |
|---|---|
| Logistic Regression | Features scaled with StandardScaler, `class_weight="balanced"`, `max_iter=2000` |
| Random Forest | 300 trees, `class_weight="balanced"` |
| Decision Tree | `max_depth=8` (so it does not memorise the data), `class_weight="balanced"` |
| Naive Bayes | Gaussian Naive Bayes |
| Baseline | Always predicts non-fraud |

They followed the same rules as the CNN: trained on training data, threshold chosen on validation, tested once.

## 8. Results (test set)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| **CNN v2 (oversampling)** | **73.7%** | 30.1% | **42.7%** | **55.3%** |
| CNN v1 (class weights) | 23.2% | **41.9%** | 29.9% | 25.2% |
| Random Forest | 28.2% | 25.8% | 27.0% | 25.5% |
| Logistic Regression | 8.8% | 24.7% | 13.0% | 8.6% |
| Decision Tree | 9.3% | 15.1% | 11.5% | 8.9% |
| Naive Bayes | 10.5% | 8.6% | 9.5% | 10.1% |
| Always non-fraud (baseline) | 0.0% | 0.0% | 0.0% | 6.6% |

Main points:
- **CNN v2 was the best model**, with a PR-AUC more than twice as high as any other model.
- **Oversampling helped much more than class weights.** CNN v1 was only about as good as Random Forest.
- CNN v2 is **precise but misses a lot of fraud**: it caught 28 of 93 fraud images and only wrongly flagged 10 of 1,323 honest claims. CNN v1 catches more fraud but with many more false alarms.
- **Accuracy is misleading:** the baseline gets 93.4% accuracy without catching any fraud.

## 9. Limitations

- CNN v2 **overfits**: its training PR-AUC reached about 97% while validation was much lower, because the same 170 fraud images are seen many times.
- The validation set has **only 30 fraud images**, so validation scores were noisy between epochs.
- Each model was trained **once** (no repeated runs with different seeds) because training on a CPU is slow.
- Resizing to 128 × 128 **loses detail** in the photos.
- Possible next step: **transfer learning** (e.g. MobileNetV2), which often works better on small image datasets.

## 10. Output files (`results/images/`)

| File | What it shows |
|---|---|
| `class_counts.png` | Number of images per class |
| `sample_images.png` | Example fraud and non-fraud images |
| `augmentation_examples.png` | One image with random augmentation |
| `cnn_v1_training_curves.png`, `cnn_v2_training_curves.png` | Training vs validation curves |
| `cnn_validation_pr_curves.png` | Precision-recall curves on validation |
| `cnn_v2_test_confusion_matrix.png` | Test confusion matrix for CNN v2 |
| `image_model_comparison.png`, `image_model_comparison.csv` | All models compared on the test set |
| `history_v1.json`, `history_v2.json` | Saved training histories |