"""
Streamlit application for evaluating pre-trained breast cancer classifiers.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# 1. App configuration and model locations
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# These filenames match the model-saving names used by the individual scripts.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "K-Nearest Neighbors (KNN)": "knn_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "Support Vector Machine (SVM)": "svm_model.joblib",
}

st.set_page_config(
    page_title="Breast Cancer Model Evaluation",
    page_icon="🩺",
    layout="wide",
)

st.title("Breast Cancer Classification – Model Evaluation")
st.write(
    "Upload a **test CSV** and evaluate it using one of the pre-trained models. "
    "The app does not train models or upload the full training dataset."
)

# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    """Load and cache a trained scikit-learn pipeline from disk."""
    return joblib.load(model_path)


def get_expected_features(model):
    """
    Get the feature names and order used when the model pipeline was trained.

    The StandardScaler is the first pipeline step and stores feature_names_in_
    when the model was trained with a pandas DataFrame.
    """
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scaler")
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            return list(scaler.feature_names_in_)

    return None


def prepare_test_data(uploaded_df, model):
    """
    Validate uploaded test data and arrange its columns in training order.

    The CSV must contain Diagnosis and all predictor columns used during
    training. ID is optional and is not passed to the model.
    """
    df = uploaded_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Accept common capitalization variations for the target column.
    diagnosis_column = next(
        (column for column in df.columns if column.lower() == "diagnosis"),
        None,
    )
    if diagnosis_column is None:
        raise ValueError(
            "The uploaded CSV must contain a 'Diagnosis' target column."
        )

    # Separate the true labels from the predictor columns.
    y_raw = df[diagnosis_column]
    X = df.drop(columns=[diagnosis_column])
    X = X.drop(columns=["ID", "id"], errors="ignore")

    # Convert diagnosis labels to the same numeric encoding used for training.
    if pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y = (
            y_raw.astype(str)
            .str.strip()
            .str.upper()
            .map({"B": 0, "M": 1, "BENIGN": 0, "MALIGNANT": 1})
        )

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Diagnosis must contain B/M, Benign/Malignant, or numeric 0/1 labels."
        )

    # Retrieve the exact training feature order, when available.
    expected_features = get_expected_features(model)

    if expected_features is not None:
        missing = [feature for feature in expected_features if feature not in X.columns]
        if missing:
            raise ValueError(
                "The uploaded CSV is missing required feature columns: "
                + ", ".join(missing)
            )
        X = X.loc[:, expected_features]
    else:
        # Fallback for a model that was not fitted with named DataFrame columns.
        expected_count = getattr(model, "n_features_in_", None)
        if expected_count is not None and X.shape[1] != expected_count:
            raise ValueError(
                f"This model expects {expected_count} features, "
                f"but the uploaded CSV contains {X.shape[1]}."
            )

    # Convert all predictors to numeric and reject incomplete/invalid rows.
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values found in feature columns: "
            + ", ".join(bad_columns)
        )

    if len(X) == 0:
        raise ValueError("The uploaded test CSV contains no rows.")

    return X, y.astype(int)


def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics and return predictions and reports."""
    predictions = model.predict(X_test)

    # Use probability estimates for ROC AUC when the classifier provides them.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = predictions

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
        "Confusion Matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "Classification Report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0,
        ),
    }

    # AUC requires both classes to be present in the uploaded test labels.
    if len(pd.Series(y_test).unique()) == 2:
        metrics["AUC"] = roc_auc_score(y_test, scores)
    else:
        metrics["AUC"] = None

    return metrics, predictions


# ---------------------------------------------------------------------------
# 3. Sidebar: model selection and test CSV upload
# ---------------------------------------------------------------------------
st.sidebar.header("Evaluation settings")

available_models = {
    display_name: MODEL_DIR / filename
    for display_name, filename in MODEL_FILES.items()
    if (MODEL_DIR / filename).exists()
}

if not available_models:
    st.error(
        "No trained model files were found. Create a 'model' folder beside "
        "app.py and place the five .joblib model files inside it."
    )
    st.code(
        "model/\n"
        "├── logistic_regression_model.joblib\n"
        "├── decision_tree_model.joblib\n"
        "├── knn_model.joblib\n"
        "├── naive_bayes_model.joblib\n"
        "└── svm_model.joblib"
    )
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Select classification model",
    options=list(available_models.keys()),
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"],
    help=(
        "Upload only held-out test rows. Include the Diagnosis column and "
        "the same feature columns used during model training."
    ),
)

st.sidebar.caption(
    "Training data stays outside the app. Only the test CSV is uploaded for evaluation."
)

# ---------------------------------------------------------------------------
# 4. Load the selected model and evaluate the uploaded test data
# ---------------------------------------------------------------------------
model_path = available_models[selected_model_name]
try:
    model = load_model(model_path)
except Exception as error:
    st.error(f"Could not load {selected_model_name}: {error}")
    st.stop()

st.subheader(f"Selected model: {selected_model_name}")

if uploaded_file is None:
    st.info("Upload a test CSV from the sidebar to display evaluation results.")
    st.markdown(
        """
        **Test CSV requirements**
        - Must contain a `Diagnosis` column with `B` / `M` labels (or 0 / 1).
        - Must contain all predictor columns used during training.
        - May contain an `ID` column; it will be ignored.
        - Do not upload the full training dataset.
        """
    )
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
    X_test, y_test = prepare_test_data(uploaded_df, model)
    metrics, predictions = evaluate_model(model, X_test, y_test)
except Exception as error:
    st.error(f"Unable to evaluate the uploaded CSV: {error}")
    st.stop()

# ---------------------------------------------------------------------------
# 5. Display evaluation metrics
# ---------------------------------------------------------------------------
st.success("Test dataset evaluated successfully.")

st.write(
    f"**Test samples:** {len(X_test)}  |  "
    f"**Features supplied:** {X_test.shape[1]}  |  "
    f"**Model:** {selected_model_name}"
)

metric_columns = st.columns(6)
metric_values = [
    ("Accuracy", metrics["Accuracy"]),
    ("AUC", metrics["AUC"]),
    ("Precision", metrics["Precision"]),
    ("Recall", metrics["Recall"]),
    ("F1 Score", metrics["F1 Score"]),
    ("MCC", metrics["MCC"]),
]

for column, (label, value) in zip(metric_columns, metric_values):
    column.metric(label, "N/A" if value is None else f"{value:.4f}")

# ---------------------------------------------------------------------------
# 6. Display confusion matrix and classification report
# ---------------------------------------------------------------------------
st.subheader("Confusion Matrix")

cm = metrics["Confusion Matrix"]
confusion_df = pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"],
)
st.dataframe(confusion_df, use_container_width=True)

st.subheader("Classification Report")
st.code(metrics["Classification Report"], language="text")

# ---------------------------------------------------------------------------
# 7. Preview predictions and allow the evaluation results to be downloaded
# ---------------------------------------------------------------------------
st.subheader("Test Predictions")

prediction_df = pd.DataFrame(
    {
        "Actual Diagnosis": y_test.map({0: "Benign", 1: "Malignant"}).values,
        "Predicted Diagnosis": pd.Series(predictions).map(
            {0: "Benign", 1: "Malignant"}
        ).values,
    }
)
st.dataframe(prediction_df, use_container_width=True)

download_df = uploaded_df.copy()
download_df["Predicted_Diagnosis"] = pd.Series(predictions).map(
    {0: "B", 1: "M"}
).values

st.download_button(
    label="Download predictions CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="test_predictions.csv",
    mime="text/csv",
)
