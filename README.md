# ML Model Comparator — Universal Supervised Learning App

**Author:** Abhijit Lalasaheb Zende

**Jio Institute ID:** 26PGPAI0009

**Program:** PGP in Artificial Intelligence and Data Science

**Institution:** Jio Institute

---

## Overview

A universal Streamlit web application that trains and compares **6 machine learning models** on **any CSV dataset** — entirely in the browser, with no pre-training required.

Upload your dataset, choose a task type, pick a target column and input features, and instantly get side-by-side metrics and visualisations for all 6 models.

The app was originally built and validated on the **UCI Bank Marketing Dataset** (predicting whether a customer will subscribe to a term deposit), and has since been extended into a full supervised learning comparator.

---

## What the App Does

1. **Upload** any CSV dataset
2. **Preview** the data — shape, first 10 rows, column type summary
3. **Select** task type — auto-detected, but always overridable
4. **Pick** a target column and input features
5. **Train** all 6 models on-the-fly with an 80/20 split — a live progress bar tracks each model
6. **Compare** all 6 models in a single highlighted table (best value per metric shown in green)
7. **Drill down** into each model — metric cards, plots, and detailed reports

---

## Models

### Classification (Binary & Multiclass) — 6 models

| Model | Key Hyperparameters | Scaling |
|-------|--------------------|----|
| Logistic Regression | `max_iter=1000`, `solver=lbfgs` | StandardScaler |
| Decision Tree | `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, `criterion=gini` | None |
| K-Nearest Neighbors | `n_neighbors=11`, `weights=distance`, `metric=minkowski` | StandardScaler |
| Naive Bayes | `var_smoothing=1e-9` (GaussianNB) | None |
| Random Forest | `n_estimators=100`, `max_depth=15`, `min_samples_split=10`, `max_features=sqrt` | None |
| XGBoost | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8` | None |

**Metrics:** Accuracy · AUC · Precision · Recall · F1 Score · MCC
**Visuals:** Confusion Matrix · Radar Chart · Bar Chart · Classification Report

### Regression — 6 models

| Model | Key Hyperparameters | Scaling |
|-------|--------------------|----|
| Linear Regression | default | StandardScaler |
| Ridge Regression | `alpha=1.0` | StandardScaler |
| Lasso Regression | `alpha=0.01`, `max_iter=5000` | StandardScaler |
| Decision Tree | `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10` | None |
| Random Forest | `n_estimators=100`, `max_depth=15`, `max_features=sqrt` | None |
| XGBoost | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8` | None |

**Metrics:** R² · Adjusted R² · RMSE · MAE · MAPE
**Visuals:** Actual vs Predicted · Residuals Plot · R² Bar Chart

### Sample Results — Bank Marketing Dataset (Binary Classification)

Using `bank.csv` with `deposit` as the target and all 16 columns as features.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8470 | 0.9000 | 0.8150 | 0.8500 | 0.8320 | 0.6850 |
| Decision Tree | 0.8400 | 0.8750 | 0.8000 | 0.8450 | 0.8220 | 0.6700 |
| K-Nearest Neighbors | 0.8510 | 0.8900 | 0.8250 | 0.8400 | 0.8320 | 0.6950 |
| Naive Bayes | 0.8470 | 0.8034 | 0.8100 | 0.8600 | 0.8340 | 0.6850 |
| Random Forest | 0.8421 | 0.9100 | 0.8200 | 0.8500 | 0.8350 | 0.6800 |
| XGBoost | 0.8550 | 0.9200 | 0.8350 | 0.8600 | 0.8470 | 0.7050 |

---

## Dataset Compatibility

The app works with **any supervised learning CSV** where:
- One column is the target (at least 2 unique non-null values)
- At least one other column exists as a feature

**Auto-preprocessing applied automatically:**
- Categorical columns (`object` / `category` dtype) → Label Encoded
- Numeric columns → NaN filled with median
- Regression target → kept as float (must be numeric)

**Task type auto-detection logic:**
- 2 unique target values → Binary Classification
- Categorical target or ≤15 unique values → Multiclass Classification
- Numeric target with >15 unique values → Regression
- Always overridable with the radio button

**Tested datasets:**
- UCI Bank Marketing (`bank.csv`) — included in this repo
- Titanic survival prediction
- Breast cancer diagnosis (Wisconsin)
- Any other supervised learning CSV

---

## Roadmap

| Version | Task | Status |
|---------|------|--------|
| v1 | Binary Classification | ✅ Done |
| v2 | Multiclass Classification + Regression | ✅ Done |
| v3 | Hyperparameter tuning UI, feature importance plots | 🔜 Planned |

---

## Project Structure

```
ml-model-comparator/
│
├── README.md                     # This file
├── bank.csv                      # Sample dataset (UCI Bank Marketing)
├── requirements.txt              # Python dependencies
├── app.py                        # Streamlit web application
│
└── models/                       # Standalone model scripts (bank dataset reference)
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    ├── random_forest.py
    └── xgboost_model.py
```

> **Note:** The `models/` scripts are standalone reference pipelines for the bank dataset showing each model's hyperparameters and training logic. The Streamlit app (`app.py`) trains all models on-the-fly from any uploaded CSV and does not depend on any pre-trained `.pkl` files.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Abhiz2411/ml-benchmark-app.git
cd ml-benchmark-app
```

### Step 2: Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Quick test with the included dataset
1. Upload `bank.csv`
2. Select `deposit` as the target column
3. Keep all 16 columns selected as features
4. Click **🚀 Train & Evaluate All Models**

---

## Running Standalone Model Scripts (Bank Dataset Reference)

Each script in `models/` is a self-contained training pipeline for the bank dataset.
Run any of them individually to see per-model training output and metrics:

```bash
python models/logistic_regression.py
python models/decision_tree.py
python models/knn.py
python models/naive_bayes.py
python models/random_forest.py
python models/xgboost_model.py
```

---

## Troubleshooting

**ModuleNotFoundError**
```
Activate your virtual environment, then: pip install -r requirements.txt
```

**Streamlit port already in use**
```bash
streamlit run app.py --server.port 8502
```

**Target column validation error**
```
Binary: target must have exactly 2 unique non-null values.
Multiclass: target must have 3 or more unique values.
Regression: target must be numeric.
```

**Training is slow**
```
KNN and Random Forest can be slower on large datasets (>100k rows).
Consider sampling the dataset before uploading.
```

---

## References

1. UCI Bank Marketing Dataset — https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
2. Scikit-learn Documentation — https://scikit-learn.org/stable/
3. XGBoost Documentation — https://xgboost.readthedocs.io/
4. Streamlit Documentation — https://docs.streamlit.io/

---

**Built by Abhijit Lalasaheb Zende**
PGP in Artificial Intelligence and Data Science · Jio Institute · 2026
