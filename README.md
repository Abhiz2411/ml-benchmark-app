# ML Model Comparator — Universal Supervised Learning App

**Author:** Abhijit Lalasaheb Zende

**Jio Institute ID:** 26PGPAI0009

**Program:** PGP in Artificial Intelligence and Data Science

**Institution:** Jio Institute

---

## Overview

A universal Streamlit web application that trains and compares **6 machine learning models** on **any CSV dataset** — entirely in the browser, with no pre-training required.

Upload your dataset, choose a target column and input features, and instantly get side-by-side metrics, confusion matrices, radar charts, and detailed reports for all 6 models.

> **Current version: Binary Classification**
> Multiclass classification and regression support are planned for upcoming versions (see [Roadmap](#roadmap)).

The app was originally built and validated on the **UCI Bank Marketing Dataset** (predicting whether a customer will subscribe to a term deposit), and is being progressively generalized into a full supervised learning comparator.

---

## What the App Does

1. **Upload** any binary classification CSV dataset
2. **Preview** the data — shape, first 10 rows, column type summary
3. **Select** a target column (must have exactly 2 unique values) and input features
4. **Train** all 6 models on-the-fly with an 80/20 stratified split — a live progress bar tracks each model
5. **Compare** all 6 models in a single highlighted table (best value per metric shown in green)
6. **Drill down** into each model — metric cards, confusion matrix, radar chart, bar chart, and classification report

---

## Models

All 6 models are trained simultaneously with the hyperparameters below.

| Model | Key Hyperparameters | Scaling |
|-------|--------------------|----|
| Logistic Regression | `max_iter=1000`, `solver=lbfgs` | StandardScaler |
| Decision Tree | `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, `criterion=gini` | None |
| K-Nearest Neighbors | `n_neighbors=11`, `weights=distance`, `metric=minkowski` | StandardScaler |
| Naive Bayes | `var_smoothing=1e-9` (GaussianNB) | None |
| Random Forest | `n_estimators=100`, `max_depth=15`, `min_samples_split=10`, `max_features=sqrt` | None |
| XGBoost | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8` | None |

### Evaluation Metrics (Binary Classification)

**Accuracy · AUC · Precision · Recall · F1 Score · MCC**

### Sample Results — Bank Marketing Dataset

Results when using `bank.csv` with `deposit` as the target and all 16 columns as features.

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

The app currently works with **any binary classification CSV** where:
- One column has exactly **2 unique non-null values** (the target)
- At least one other column exists as a feature

**Auto-preprocessing applied automatically:**
- Categorical columns (`object` / `category` dtype) → Label Encoded
- Numeric columns → used as-is
- Missing values → filled with mode (categorical) or median (numeric)

**Tested datasets:**
- UCI Bank Marketing (`bank.csv`) — included in this repo
- Titanic survival prediction
- Breast cancer diagnosis (Wisconsin)
- Any other binary classification CSV

---

## Roadmap

The goal is to evolve this into a **complete supervised learning comparator** covering all three task types.

| Version | Task | Status |
|---------|------|--------|
| v1 — Current | Binary Classification | ✅ Done |
| v2 | Multiclass Classification | 🔜 Planned |
| v3 | Regression | 🔜 Planned |

**Planned additions for Multiclass (v2):**
- Auto-detect number of target classes and switch to multiclass mode
- Metrics: macro/weighted Accuracy, Precision, Recall, F1, Cohen's Kappa
- Per-class breakdown in classification report
- One-vs-Rest AUC

**Planned additions for Regression (v3):**
- Regression model set: Linear Regression, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor
- Metrics: RMSE, MAE, R², Adjusted R², MAPE
- Actual vs Predicted scatter plot, residual plot

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
The selected target column must have exactly 2 unique non-null values (binary classification).
Multiclass support is coming in v2.
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
