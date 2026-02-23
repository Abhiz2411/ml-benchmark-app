"""
Universal Binary Classification App - Interactive Streamlit Application
Supports any binary classification CSV dataset with on-the-fly model training.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Universal Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def auto_preprocess(df, target_col, feature_cols):
    """
    Auto-detect column types and encode accordingly.
    Returns X (features), y (label-encoded target), label_encoders dict.
    """
    data = df[feature_cols + [target_col]].copy()

    label_encoders = {}

    # Process features
    for col in feature_cols:
        if data[col].dtype in ['object', 'category']:
            # Fill missing with mode
            mode_val = data[col].mode()
            if not mode_val.empty:
                data[col] = data[col].fillna(mode_val[0])
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
        else:
            # Fill missing with median
            data[col] = data[col].fillna(data[col].median())

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(data[target_col].astype(str))
    label_encoders['__target__'] = le_target

    X = data[feature_cols]

    return X, y, label_encoders


# ---------------------------------------------------------------------------
# Model training functions
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        criterion='gini'
    )
    model.fit(X_train, y_train)
    return model, None


def train_knn(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = KNeighborsClassifier(
        n_neighbors=11,
        weights='distance',
        algorithm='auto',
        metric='minkowski',
        p=2
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_naive_bayes(X_train, y_train):
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train, y_train)
    return model, None


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, None


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model, None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Train all 6 models and return results dict.
    Shows a progress bar while training.
    """
    trainers = [
        ('Logistic Regression', train_logistic_regression),
        ('Decision Tree', train_decision_tree),
        ('K-Nearest Neighbors', train_knn),
        ('Naive Bayes', train_naive_bayes),
        ('Random Forest', train_random_forest),
        ('XGBoost', train_xgboost),
    ]

    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, trainer_fn) in enumerate(trainers):
        status_text.text(f"Training {name}... ({i + 1}/{len(trainers)})")

        model, scaler = trainer_fn(X_train, y_train)

        if scaler is not None:
            X_test_input = scaler.transform(X_test)
        else:
            X_test_input = X_test

        y_pred = model.predict(X_test_input)
        y_pred_proba = model.predict_proba(X_test_input)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'metrics': metrics,
            'cm': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }

        progress_bar.progress((i + 1) / len(trainers))

    status_text.text("All models trained successfully!")
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_comparison_table(all_results):
    """
    Render a styled DataFrame comparing all 6 models.
    Best value per column highlighted green.
    """
    metric_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    rows = {}
    for model_name, res in all_results.items():
        rows[model_name] = {k: res['metrics'][k] for k in metric_keys}

    comparison_df = pd.DataFrame(rows).T
    comparison_df = comparison_df[metric_keys]

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #90EE90' if v else '' for v in is_max]

    styled = comparison_df.style.apply(highlight_max, axis=0).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)


def plot_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    return fig


def plot_metrics_radar(metrics):
    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Model Performance Radar Chart",
        width=600,
        height=500
    )
    return fig


def plot_metrics_bar(metrics):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker=dict(
            color=list(metrics.values()),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        text=[f'{v:.4f}' for v in metrics.values()],
        textposition='outside'
    ))
    fig.update_layout(
        title="Performance Metrics Overview",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        height=500,
        showlegend=False
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.title("🤖 Universal Binary Classification App")
    st.markdown("Upload any binary classification CSV, pick your target and features, and compare all 6 models.")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    st.sidebar.subheader("1️⃣ Upload CSV Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload any binary classification dataset in CSV format"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
**Instructions:**
1. Upload a CSV dataset
2. Select the target (label) column
3. Select input feature columns
4. Click **Train & Evaluate All Models**
5. Explore per-model results

**Requirements:**
- Target column must have exactly 2 unique non-null values
- At least 1 feature column must be selected
    """)

    # -----------------------------------------------------------------------
    # Step 1 — Upload & Preview
    # -----------------------------------------------------------------------
    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return

    st.subheader("Step 1 — Dataset Preview")
    st.success(f"File uploaded: **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    with st.expander("View first 10 rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # Column type summary
    col_summary = pd.DataFrame({
        'Column': df.columns,
        'Dtype': df.dtypes.astype(str).values,
        'Non-Null': df.notna().sum().values,
        'Null': df.isna().sum().values,
        'Unique Values': df.nunique().values,
    })
    with st.expander("Column type summary"):
        st.dataframe(col_summary, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Step 2 — Select Variables
    # -----------------------------------------------------------------------
    st.subheader("Step 2 — Select Variables")

    all_columns = df.columns.tolist()

    target_col = st.selectbox(
        "Select the **target** (label) column:",
        options=all_columns,
        help="Must have exactly 2 unique non-null values"
    )

    # Validate target
    unique_vals = df[target_col].dropna().unique()
    if len(unique_vals) != 2:
        st.error(
            f"Target column **{target_col}** has {len(unique_vals)} unique values "
            f"({list(unique_vals)[:10]}). Please select a binary column (exactly 2 unique values)."
        )
        return
    else:
        st.success(f"Target classes: **{unique_vals[0]}** and **{unique_vals[1]}**")

    default_features = [c for c in all_columns if c != target_col]
    feature_cols = st.multiselect(
        "Select **input feature** columns:",
        options=default_features,
        default=default_features,
        help="Select at least 1 feature column"
    )

    if len(feature_cols) == 0:
        st.warning("Please select at least 1 feature column.")
        return

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Step 3 — Train & Evaluate
    # -----------------------------------------------------------------------
    st.subheader("Step 3 — Train & Evaluate All Models")

    if st.button("🚀 Train & Evaluate All Models", type="primary"):
        with st.spinner("Preprocessing data..."):
            try:
                X, y, label_encoders = auto_preprocess(df, target_col, feature_cols)
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        st.write(f"Train set: **{X_train.shape[0]}** rows | Test set: **{X_test.shape[0]}** rows")

        try:
            all_results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return

        st.session_state['all_results'] = all_results
        st.session_state['y_test'] = y_test
        st.session_state['target_classes'] = label_encoders.get('__target__').classes_.tolist()

    # -----------------------------------------------------------------------
    # Step 4 — Results
    # -----------------------------------------------------------------------
    if 'all_results' not in st.session_state:
        return

    all_results = st.session_state['all_results']
    y_test = st.session_state['y_test']
    class_names = st.session_state.get('target_classes', ['0', '1'])

    st.markdown("---")
    st.subheader("Step 4 — Results")

    # 4a — Comparison table
    st.markdown("#### Model Comparison (best per metric highlighted)")
    plot_comparison_table(all_results)

    st.markdown("---")

    # 4b — Per-model expandable sections
    st.markdown("#### Per-Model Details")

    for model_name, res in all_results.items():
        with st.expander(f"📊 {model_name}", expanded=False):
            metrics = res['metrics']
            y_pred = res['y_pred']
            y_pred_proba = res['y_pred_proba']
            cm = res['cm']

            # Metric cards
            cols = st.columns(6)
            metric_labels = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
            for col, label in zip(cols, metric_labels):
                col.metric(label, f"{metrics[label]:.4f}")

            # Confusion matrix + radar side by side
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                fig_cm = plot_confusion_matrix(cm, class_names=class_names)
                st.plotly_chart(fig_cm, use_container_width=True)
            with viz_col2:
                fig_radar = plot_metrics_radar(metrics)
                st.plotly_chart(fig_radar, use_container_width=True)

            # Bar chart
            fig_bar = plot_metrics_bar(metrics)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Classification report
            st.markdown("**Classification Report**")
            cr = classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True
            )
            cr_df = pd.DataFrame(cr).transpose()
            st.dataframe(
                cr_df.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p><strong>Universal Binary Classification App</strong></p>
        <p>Supports any binary classification CSV &mdash; 6 models trained on-the-fly</p>
        <p>Built by <strong>Abhijit Zende</strong></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
