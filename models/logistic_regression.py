"""
Logistic Regression Model for Bank Marketing Dataset
Author: BITS Pilani Student
Dataset: UCI Bank Marketing Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import pickle
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df):
    """
    Preprocess the bank marketing dataset
    """
    # Create a copy
    data = df.copy()

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'month', 'poutcome']

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Encode target variable
    target_encoder = LabelEncoder()
    data['deposit'] = target_encoder.fit_transform(data['deposit'])

    # Separate features and target
    X = data.drop('deposit', axis=1)
    y = data['deposit']

    return X, y, label_encoders, target_encoder


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    return model, scaler


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate all required metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    return metrics


def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate the trained model
    """
    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification Report
    cr = classification_report(y_test, y_pred)

    return metrics, cm, cr, y_pred, y_pred_proba


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("Logistic Regression - Bank Marketing Dataset")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv('bank.csv')
    print(f"Dataset shape: {df.shape}")

    # Preprocess
    print("\n[2/5] Preprocessing data...")
    X, y, label_encoders, target_encoder = preprocess_data(df)

    # Split data
    print("\n[3/5] Splitting data (80-20 train-test split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train model
    print("\n[4/5] Training Logistic Regression model...")
    model, scaler = train_logistic_regression(X_train, y_train)
    print("Model training completed!")

    # Evaluate
    print("\n[5/5] Evaluating model...")
    metrics, cm, cr, y_pred, y_pred_proba = evaluate_model(model, scaler, X_test, y_test)

    # Display results
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:.<30} {metric_value:.4f}")

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(cm)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(cr)

    # Save model artifacts
    print("\n[*] Saving model artifacts...")
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_names': X.columns.tolist()
    }

    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    print("[✓] Model saved successfully: models/logistic_regression_model.pkl")
    print("\n" + "=" * 60)

    return metrics


if __name__ == "__main__":
    main()
