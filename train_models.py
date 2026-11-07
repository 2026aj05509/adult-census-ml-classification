"""Train five classification pipelines on the Adult Census Income dataset."""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "adult.csv"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
TARGET = "income"
POSITIVE_CLASS = ">50K"


def build_preprocessor(numeric_cols, categorical_cols, dense=False):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=not dense)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], sparse_threshold=0 if dense else 1.0)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    # '?' is used by the source dataset to represent missing categorical values.
    df = df.replace("?", np.nan)
    df = df.drop(columns=["fnlwgt"], errors="ignore")
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str).str.strip().str.replace(".", "", regex=False)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": LinearSVC(class_weight="balanced", random_state=42, max_iter=5000),
    }

    rows = []
    for name, estimator in model_specs.items():
        is_nb = name == "Naive Bayes"
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols, dense=is_nb)),
            ("classifier", estimator),
        ])
        print(f"Training {name} ...", flush=True)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index(POSITIVE_CLASS)
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            score = pipeline.predict_proba(X_test)[:, positive_index]
        else:
            raw = pipeline.decision_function(X_test)
            score = raw if positive_index == 1 else -raw
        row = {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score((y_test == POSITIVE_CLASS).astype(int), score),
            "Precision": precision_score(y_test, pred, pos_label=POSITIVE_CLASS, zero_division=0),
            "Recall": recall_score(y_test, pred, pos_label=POSITIVE_CLASS, zero_division=0),
            "F1": f1_score(y_test, pred, pos_label=POSITIVE_CLASS, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        }
        rows.append(row)
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)
        print(f"  Metrics: {row}", flush=True)

    metrics = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    metrics.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    print("\nSaved metrics, five model pipelines, and test_data.csv")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
