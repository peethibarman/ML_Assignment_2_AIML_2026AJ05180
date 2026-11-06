"""
K-Nearest Neighbors (KNN)
"""

# Imports
import joblib
from pathlib import Path
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# File path and experiment settings
# Resolving paths relative to this script so it can be run from any directory
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
EVAL_PATH = BASE_DIR / "eval_knn.txt"
MODEL_DIR = BASE_DIR / "model"

# Keeping the split
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Selecting the 14 features 
N_FEATURES = 14
N_NEIGHBORS = 5


def load_and_split_data():
    """
    Read data.csv, validate and clean the dataset, then create a stratified
    80/20 training and testing split.
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Checking that the required input file exists before attempting to read it
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Place data.csv beside this script."
        )

    # Loading the dataset and remove accidental spaces from column names.
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Confirming that the expected target column is available.
    if "Diagnosis" not in df.columns:
        raise ValueError("data.csv must contain a 'Diagnosis' target column.")

    df = df.drop(columns=["ID"], errors="ignore")
    df = df.dropna()

    # Converting the diagnosis labels into numeric classes:
    # Benign (B) = 0 and Malignant (M) = 1.
    y = (
        df["Diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"B": 0, "M": 1})
    )

    # Stoping if the target contains labels other than B or M
    if y.isna().any():
        raise ValueError("Diagnosis must contain only B (Benign) or M (Malignant).")

    # Converting all predictor columns to numeric value
    X = df.drop(columns=["Diagnosis"]).apply(pd.to_numeric, errors="coerce")

    # Keeping only rows where every feature has a valid numeric value
    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Validating the dataset size and feature count before splitting
    if X.shape[1] < N_FEATURES:
        raise ValueError(
            f"Expected at least {N_FEATURES} numeric features; found {X.shape[1]}."
        )
    if len(X) < 20:
        raise ValueError("Not enough valid rows to create a train/test split.")
    if y.nunique() != 2:
        raise ValueError("The cleaned dataset must contain both diagnosis classes.")

    # Stratification preserving the class proportions in both data subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, list(X.columns)


def build_model():
    """
    Building a KNN pipeline using the selected 14 features.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("feature_selection", SelectKBest(score_func=f_classif, k=N_FEATURES)),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="uniform", metric="minkowski"),
            ),
        ]
    )


def evaluate_model(model, X_test, y_test):
    """
    Generating predictions and calculate classification evaluation metrics.
    Returns:
        A dictionary containing the metrics, confusion matrix and report.
    """
    # Predict the class labels for the held-out test data.
    predictions = model.predict(X_test)

    # Use probability scores for AUC when the model supports them.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = predictions

    # Calculating the requested metrics. Zero-division is handled explicitly
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "auc": roc_auc_score(y_test, scores),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0,
        ),
    }
    return metrics


def save_evaluation(metrics, train_count, test_count, feature_names):
    """
    Write the model configuration and evaluation results to a text file.

    The output file can be used later when preparing the final assignment
    comparison and observations.
    """
    # Format the evaluation results in a readable assignment-friendly layout.
    report = f"""MODEL: K-Nearest Neighbors (KNN)
            DATASET: data.csv
            SPLIT: 80% training / 20% testing
            RANDOM STATE: {RANDOM_STATE}
            FEATURE COUNT: {len(feature_names)}
            SELECTED FEATURES: {", ".join(feature_names)}
            TRAINING SAMPLES: {train_count}
            TESTING SAMPLES: {test_count}

            EVALUATION METRICS
            Accuracy  : {metrics['accuracy']:.4f}
            AUC Score : {metrics['auc']:.4f}
            Precision : {metrics['precision']:.4f}
            Recall    : {metrics['recall']:.4f}
            F1 Score  : {metrics['f1']:.4f}
            MCC Score : {metrics['mcc']:.4f}

            CONFUSION MATRIX
            {metrics['confusion_matrix'].tolist()}

            CLASSIFICATION REPORT
            {metrics['classification_report']}
            """

    # Saving the evaluation file beside this script
    EVAL_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Evaluation saved to: {EVAL_PATH}")


def main():
    """
    Run the complete KNN workflow:
    load data, build the model, train, evaluate and save the evaluation file.
    """
    # Preparing the training and testing data from data.csv
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()

    # Creating the model and fit it using only the training subset
    model = build_model()
    model.fit(X_train, y_train)

    # Evaluating performance on the separate test subset
    metrics = evaluate_model(model, X_test, y_test)

    # Retrieving the names of the 14 features selected using training data only
    selector = model.named_steps["feature_selection"]
    selected_features = [
        name for name, selected in zip(feature_names, selector.get_support()) if selected
    ]

    # Saving the trained model in the model folder
    MODEL_DIR = BASE_DIR / "model"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "knn_model.joblib")

    print("Model saved successfully in model/knn_model.joblib")

    # Saving the results for use in the final assignment summary.
    save_evaluation(metrics, len(X_train), len(X_test), selected_features)


# Run the workflow only when this file is executed directly.
if __name__ == "__main__":
    main()
