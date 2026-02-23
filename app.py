"""
Universal ML Comparator - Interactive Streamlit Application
Supports: Binary Classification, Multiclass Classification, Regression
Built by Abhijit Lalasaheb Zende
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error
)
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Model Comparator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    h1 { color: #1f77b4; }
    h2 { color: #2ca02c; }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINARY      = "Binary Classification"
MULTICLASS  = "Multiclass Classification"
REGRESSION  = "Regression"


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def detect_task_type(series):
    """Suggest a task type based on the target column's properties."""
    n_unique = series.nunique()
    dtype    = series.dtype

    if n_unique == 2:
        return BINARY
    elif dtype in ['object', 'category'] or n_unique <= 15:
        return MULTICLASS
    else:
        return REGRESSION


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def auto_preprocess(df, target_col, feature_cols, task_type):
    """
    Encode features and prepare target.
    - Categorical features  → LabelEncoder
    - Numeric features      → fill NaN with median
    - Classification target → LabelEncoder
    - Regression target     → float, fill NaN with median
    Returns: X (DataFrame), y (ndarray), label_encoders (dict)
    """
    data = df[feature_cols].copy()
    label_encoders = {}

    for col in feature_cols:
        if data[col].dtype in ['object', 'category']:
            mode_val = data[col].mode()
            if not mode_val.empty:
                data[col] = data[col].fillna(mode_val[0])
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
        else:
            data[col] = data[col].fillna(data[col].median())

    X = data

    if task_type == REGRESSION:
        target_series = df[target_col].copy()
        if target_series.dtype in ['object', 'category']:
            raise ValueError(
                f"Regression requires a numeric target. '{target_col}' is categorical."
            )
        y = target_series.fillna(target_series.median()).values.astype(float)
    else:
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target_col].astype(str))
        label_encoders['__target__'] = le_target

    return X, y, label_encoders


# ---------------------------------------------------------------------------
# Classification training functions  (binary + multiclass)
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_scaled, y_train)
    return model, scaler


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, min_samples_leaf=10,
        random_state=42, criterion='gini'
    )
    model.fit(X_train, y_train)
    return model, None


def train_knn(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = KNeighborsClassifier(
        n_neighbors=11, weights='distance', algorithm='auto',
        metric='minkowski', p=2
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_naive_bayes(X_train, y_train):
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train, y_train)
    return model, None


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, None


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, gamma=0,
        reg_alpha=0, reg_lambda=1, random_state=42, n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model, None


# ---------------------------------------------------------------------------
# Regression training functions
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    return model, scaler


def train_ridge(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_lasso(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Lasso(alpha=0.01, max_iter=5000)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_decision_tree_regressor(X_train, y_train):
    model = DecisionTreeRegressor(
        max_depth=10, min_samples_split=20, min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, None


def train_random_forest_regressor(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, None


def train_xgboost_regressor(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, eval_metric='rmse'
    )
    model.fit(X_train, y_train)
    return model, None


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def calculate_classification_metrics(y_true, y_pred, y_pred_proba, task_type):
    avg = 'binary' if task_type == BINARY else 'macro'

    try:
        if task_type == BINARY:
            auc = roc_auc_score(y_true, y_pred_proba)
        else:
            auc = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovr', average='macro'
            )
    except Exception:
        auc = float('nan')

    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'AUC':       auc,
        'Precision': precision_score(y_true, y_pred, average=avg, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, average=avg, zero_division=0),
        'F1 Score':  f1_score(y_true, y_pred, average=avg, zero_division=0),
        'MCC':       matthews_corrcoef(y_true, y_pred),
    }


def calculate_regression_metrics(y_true, y_pred, n_features):
    r2   = r2_score(y_true, y_pred)
    n    = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else float('nan')
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
           if mask.any() else float('nan')

    return {
        'R²':       r2,
        'Adj. R²':  adj_r2,
        'RMSE':     rmse,
        'MAE':      mae,
        'MAPE (%)': mape,
    }


# ---------------------------------------------------------------------------
# Train & evaluate — classification
# ---------------------------------------------------------------------------

def train_and_evaluate_classification(X_train, X_test, y_train, y_test, task_type):
    trainers = [
        ('Logistic Regression',  train_logistic_regression),
        ('Decision Tree',        train_decision_tree),
        ('K-Nearest Neighbors',  train_knn),
        ('Naive Bayes',          train_naive_bayes),
        ('Random Forest',        train_random_forest),
        ('XGBoost',              train_xgboost),
    ]

    results     = {}
    progress    = st.progress(0)
    status_text = st.empty()

    for i, (name, fn) in enumerate(trainers):
        status_text.text(f"Training {name}... ({i + 1}/{len(trainers)})")
        model, scaler = fn(X_train, y_train)
        X_in = scaler.transform(X_test) if scaler is not None else X_test

        y_pred          = model.predict(X_in)
        y_proba_full    = model.predict_proba(X_in)
        y_pred_proba    = y_proba_full[:, 1] if task_type == BINARY else y_proba_full

        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba, task_type)
        cm      = confusion_matrix(y_test, y_pred)

        results[name] = {
            'metrics':      metrics,
            'cm':           cm,
            'y_pred':       y_pred,
            'y_pred_proba': y_pred_proba,
        }
        progress.progress((i + 1) / len(trainers))

    status_text.text("All models trained successfully!")
    return results


# ---------------------------------------------------------------------------
# Train & evaluate — regression
# ---------------------------------------------------------------------------

def train_and_evaluate_regression(X_train, X_test, y_train, y_test):
    trainers = [
        ('Linear Regression',    train_linear_regression),
        ('Ridge Regression',     train_ridge),
        ('Lasso Regression',     train_lasso),
        ('Decision Tree',        train_decision_tree_regressor),
        ('Random Forest',        train_random_forest_regressor),
        ('XGBoost',              train_xgboost_regressor),
    ]

    n_features  = X_train.shape[1]
    results     = {}
    progress    = st.progress(0)
    status_text = st.empty()

    for i, (name, fn) in enumerate(trainers):
        status_text.text(f"Training {name}... ({i + 1}/{len(trainers)})")
        model, scaler = fn(X_train, y_train)
        X_in  = scaler.transform(X_test) if scaler is not None else X_test
        y_pred = model.predict(X_in)

        metrics = calculate_regression_metrics(y_test, y_pred, n_features)

        results[name] = {
            'metrics': metrics,
            'y_pred':  y_pred,
        }
        progress.progress((i + 1) / len(trainers))

    status_text.text("All models trained successfully!")
    return results


# ---------------------------------------------------------------------------
# Plots — classification
# ---------------------------------------------------------------------------

def plot_comparison_table_classification(all_results):
    metric_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    rows = {name: {k: res['metrics'][k] for k in metric_keys}
            for name, res in all_results.items()}
    df = pd.DataFrame(rows).T

    def highlight_max(s):
        return ['background-color: #90EE90' if v == s.max() else '' for v in s]

    st.dataframe(
        df.style.apply(highlight_max, axis=0).format("{:.4f}"),
        use_container_width=True
    )


def plot_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale='Blues', text=cm,
        texttemplate='%{text}', textfont={"size": 16}, showscale=True
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted', yaxis_title='Actual',
        width=500, height=500,
        xaxis=dict(side='bottom'), yaxis=dict(autorange='reversed')
    )
    return fig


def plot_metrics_radar(metrics):
    categories = list(metrics.keys())
    values     = [v if not np.isnan(v) else 0 for v in metrics.values()]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title="Performance Radar", height=450
    )
    return fig


def plot_metrics_bar(metrics):
    values = [v if not np.isnan(v) else 0 for v in metrics.values()]
    fig = go.Figure(go.Bar(
        x=list(metrics.keys()), y=values,
        marker=dict(color=values, colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Score")),
        text=[f'{v:.4f}' for v in values], textposition='outside'
    ))
    fig.update_layout(
        title="Metrics Overview", xaxis_title="Metric", yaxis_title="Score",
        yaxis=dict(range=[0, 1.15]), height=450, showlegend=False
    )
    return fig


# ---------------------------------------------------------------------------
# Plots — regression
# ---------------------------------------------------------------------------

def plot_comparison_table_regression(all_results):
    metric_keys   = ['R²', 'Adj. R²', 'RMSE', 'MAE', 'MAPE (%)']
    higher_better = {'R²', 'Adj. R²'}
    lower_better  = {'RMSE', 'MAE', 'MAPE (%)'}

    rows = {name: res['metrics'] for name, res in all_results.items()}
    df   = pd.DataFrame(rows).T[metric_keys]

    def highlight_best(frame):
        styles = pd.DataFrame('', index=frame.index, columns=frame.columns)
        for col in frame.columns:
            if col in higher_better:
                best = frame[col].max()
            else:
                best = frame[col].min()
            styles[col] = [
                'background-color: #90EE90' if v == best else ''
                for v in frame[col]
            ]
        return styles

    st.dataframe(
        df.style.apply(highlight_best, axis=None).format("{:.4f}"),
        use_container_width=True
    )


def plot_actual_vs_predicted(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.tolist(), y=y_pred.tolist(), mode='markers',
        marker=dict(color='#1f77b4', opacity=0.55, size=6),
        name='Predictions'
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Fit'
    ))
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Actual', yaxis_title='Predicted',
        height=430, showlegend=True
    )
    return fig


def plot_residuals(y_test, y_pred):
    y_test    = np.array(y_test)
    y_pred    = np.array(y_pred)
    residuals = y_test - y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred.tolist(), y=residuals.tolist(), mode='markers',
        marker=dict(color='#ff7f0e', opacity=0.55, size=6)
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2)
    fig.update_layout(
        title='Residuals vs Predicted',
        xaxis_title='Predicted', yaxis_title='Residual (Actual − Predicted)',
        height=430, showlegend=False
    )
    return fig


def plot_regression_bar(metrics):
    """Bar chart for R² and Adj. R² only (same 0-1 scale)."""
    keys   = ['R²', 'Adj. R²']
    values = [metrics[k] for k in keys]
    fig = go.Figure(go.Bar(
        x=keys, y=values,
        marker=dict(color=['#1f77b4', '#2ca02c']),
        text=[f'{v:.4f}' for v in values], textposition='outside'
    ))
    fig.update_layout(
        title='R² Scores', yaxis=dict(range=[0, 1.15]),
        height=350, showlegend=False
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.title("🤖 ML Model Comparator")
    st.markdown(
        "Upload any CSV, pick a task type, select your target and features, "
        "and compare all models side by side."
    )
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    st.sidebar.subheader("1️⃣ Upload CSV Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    st.sidebar.markdown("---")
    st.sidebar.info("""
**Supported tasks**
- 🔵 Binary Classification
- 🟣 Multiclass Classification
- 🟢 Regression

**Auto-preprocessing**
- Categorical → Label Encoded
- Numeric → NaN filled with median
- 80 / 20 stratified split (classification) or random split (regression)
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
        st.error(f"Could not read CSV: {e}")
        return

    st.subheader("Step 1 — Dataset Preview")
    st.success(
        f"**{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    with st.expander("View first 10 rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    col_summary = pd.DataFrame({
        'Column':       df.columns,
        'Dtype':        df.dtypes.astype(str).values,
        'Non-Null':     df.notna().sum().values,
        'Null':         df.isna().sum().values,
        'Unique Values': df.nunique().values,
    })
    with st.expander("Column type summary"):
        st.dataframe(col_summary, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Step 2 — Select variables + task type
    # -----------------------------------------------------------------------
    st.subheader("Step 2 — Select Variables & Task Type")

    all_columns = df.columns.tolist()

    target_col = st.selectbox(
        "Select the **target** column:",
        options=all_columns,
        help="The column you want to predict"
    )

    # Auto-detect and let user override
    suggested = detect_task_type(df[target_col].dropna())
    task_type = st.radio(
        "Task type:",
        [BINARY, MULTICLASS, REGRESSION],
        index=[BINARY, MULTICLASS, REGRESSION].index(suggested),
        horizontal=True,
        help="Auto-detected from your target column. Override if needed."
    )

    # Validate
    n_unique    = df[target_col].dropna().nunique()
    unique_vals = df[target_col].dropna().unique()

    if task_type == BINARY:
        if n_unique != 2:
            st.error(
                f"Binary Classification needs exactly 2 unique target values. "
                f"**'{target_col}'** has {n_unique}."
            )
            return
        st.success(f"Target classes: **{unique_vals[0]}** and **{unique_vals[1]}**")

    elif task_type == MULTICLASS:
        if n_unique < 3:
            st.warning(
                f"Multiclass selected but target has only {n_unique} unique value(s). "
                f"Consider switching to Binary Classification."
            )
        preview = list(unique_vals[:6])
        more    = f" … (+{n_unique - 6} more)" if n_unique > 6 else ""
        st.success(f"**{n_unique}** target classes detected: {preview}{more}")

    else:  # REGRESSION
        if df[target_col].dtype in ['object', 'category']:
            st.error(
                f"Regression requires a numeric target. "
                f"**'{target_col}'** is categorical."
            )
            return
        tgt = df[target_col].dropna()
        st.success(
            f"Numeric target — range: [{tgt.min():.3g}, {tgt.max():.3g}], "
            f"mean: {tgt.mean():.3g}, std: {tgt.std():.3g}"
        )

    default_features = [c for c in all_columns if c != target_col]
    feature_cols = st.multiselect(
        "Select **input feature** columns:",
        options=default_features,
        default=default_features,
    )

    if not feature_cols:
        st.warning("Select at least 1 feature column.")
        return

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Step 3 — Train
    # -----------------------------------------------------------------------
    st.subheader("Step 3 — Train & Evaluate All Models")

    model_names = {
        BINARY:     ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
                     'Naive Bayes', 'Random Forest', 'XGBoost'],
        MULTICLASS: ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
                     'Naive Bayes', 'Random Forest', 'XGBoost'],
        REGRESSION: ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
                     'Decision Tree', 'Random Forest', 'XGBoost'],
    }
    st.caption(f"Models that will run: {' · '.join(model_names[task_type])}")

    if st.button("🚀 Train & Evaluate All Models", type="primary"):
        with st.spinner("Preprocessing data..."):
            try:
                X, y, label_encoders = auto_preprocess(
                    df, target_col, feature_cols, task_type
                )
            except ValueError as e:
                st.error(str(e))
                return

        if task_type == REGRESSION:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

        st.write(
            f"Train: **{X_train.shape[0]:,}** rows | "
            f"Test: **{X_test.shape[0]:,}** rows | "
            f"Features: **{X_train.shape[1]}**"
        )

        try:
            if task_type == REGRESSION:
                all_results = train_and_evaluate_regression(
                    X_train, X_test, y_train, y_test
                )
            else:
                all_results = train_and_evaluate_classification(
                    X_train, X_test, y_train, y_test, task_type
                )
        except Exception as e:
            st.error(f"Training failed: {e}")
            return

        st.session_state['all_results']    = all_results
        st.session_state['y_test']         = y_test
        st.session_state['task_type']      = task_type
        st.session_state['n_features']     = len(feature_cols)
        if task_type != REGRESSION:
            st.session_state['target_classes'] = \
                label_encoders['__target__'].classes_.tolist()

    # -----------------------------------------------------------------------
    # Step 4 — Results
    # -----------------------------------------------------------------------
    if 'all_results' not in st.session_state:
        return

    all_results  = st.session_state['all_results']
    y_test       = st.session_state['y_test']
    saved_task   = st.session_state['task_type']
    n_features   = st.session_state.get('n_features', 1)

    st.markdown("---")
    st.subheader("Step 4 — Results")
    st.caption(f"Showing results for: **{saved_task}**")

    # ── Regression results ─────────────────────────────────────────────────
    if saved_task == REGRESSION:
        st.markdown("#### Model Comparison (green = best per metric)")
        plot_comparison_table_regression(all_results)
        st.markdown("---")
        st.markdown("#### Per-Model Details")

        for model_name, res in all_results.items():
            with st.expander(f"📊 {model_name}", expanded=False):
                metrics = res['metrics']
                y_pred  = np.array(res['y_pred'])

                # Metric cards
                cols = st.columns(5)
                for col, (label, val) in zip(cols, metrics.items()):
                    col.metric(label, f"{val:.4f}" if not np.isnan(val) else "N/A")

                # Actual vs Predicted + Residuals
                v1, v2 = st.columns(2)
                with v1:
                    st.plotly_chart(
                        plot_actual_vs_predicted(y_test, y_pred),
                        use_container_width=True
                    )
                with v2:
                    st.plotly_chart(
                        plot_residuals(y_test, y_pred),
                        use_container_width=True
                    )

                # R² bar chart
                st.plotly_chart(
                    plot_regression_bar(metrics), use_container_width=True
                )

    # ── Classification results ─────────────────────────────────────────────
    else:
        class_names = st.session_state.get('target_classes', ['0', '1'])

        st.markdown("#### Model Comparison (green = best per metric)")
        plot_comparison_table_classification(all_results)
        st.markdown("---")
        st.markdown("#### Per-Model Details")

        for model_name, res in all_results.items():
            with st.expander(f"📊 {model_name}", expanded=False):
                metrics      = res['metrics']
                y_pred       = res['y_pred']
                y_pred_proba = res['y_pred_proba']
                cm           = res['cm']

                # Metric cards
                cols = st.columns(6)
                for col, (label, val) in zip(cols, metrics.items()):
                    col.metric(label, f"{val:.4f}" if not np.isnan(val) else "N/A")

                # Confusion matrix + Radar
                v1, v2 = st.columns(2)
                with v1:
                    st.plotly_chart(
                        plot_confusion_matrix(cm, class_names=class_names),
                        use_container_width=True
                    )
                with v2:
                    st.plotly_chart(
                        plot_metrics_radar(metrics), use_container_width=True
                    )

                # Bar chart
                st.plotly_chart(
                    plot_metrics_bar(metrics), use_container_width=True
                )

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
        <p><strong>ML Model Comparator</strong> &mdash;
        Binary Classification &middot; Multiclass Classification &middot; Regression</p>
        <p>Built by <strong>Abhijit Lalasaheb Zende</strong></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
