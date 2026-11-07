"""Streamlit interface for evaluating saved Adult Census Income classifiers."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "income"
POSITIVE_CLASS = ">50K"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Adult Income Classifier", page_icon="📊", layout="wide")
st.title("Adult Census Income — Classification Model Evaluation")
st.write("Upload a held-out test CSV containing the model input features and the `income` target. Saved models are loaded; this app does not retrain them.")

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

models = load_models()
upload = st.file_uploader("Upload test data (CSV)", type=["csv"])
if upload is not None:
    try:
        data = pd.read_csv(upload)
        if TARGET not in data.columns:
            st.error(f"The uploaded CSV must include the target column '{TARGET}'.")
            st.stop()
        y_true = data[TARGET].astype(str).str.strip().str.replace(".", "", regex=False)
        X_test = data.drop(columns=[TARGET, "fnlwgt"], errors="ignore")
        expected = list(models["Logistic Regression"].feature_names_in_)
        missing = sorted(set(expected) - set(X_test.columns))
        extra = sorted(set(X_test.columns) - set(expected))
        if missing:
            st.error("Missing required feature columns: " + ", ".join(missing))
            st.stop()
        # Match training feature order; harmless extra columns are not used.
        X_test = X_test[expected]
        st.caption(f"Loaded {len(data):,} test rows. Extra columns ignored: {', '.join(extra) if extra else 'none'}.")

        selected = st.selectbox("Select classifier", list(MODEL_FILES.keys()))
        run_all = st.checkbox("Show results for all five models", value=True)
        model_names = list(MODEL_FILES.keys()) if run_all else [selected]
        results = []
        details = {}
        for name in model_names:
            estimator = models[name]
            pred = estimator.predict(X_test)
            classes = list(estimator.named_steps["classifier"].classes_)
            positive_index = classes.index(POSITIVE_CLASS)
            classifier = estimator.named_steps["classifier"]
            if hasattr(classifier, "predict_proba"):
                scores = estimator.predict_proba(X_test)[:, positive_index]
            else:
                raw = estimator.decision_function(X_test)
                scores = raw if positive_index == 1 else -raw
            y_binary = (y_true == POSITIVE_CLASS).astype(int)
            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_true, pred),
                "AUC": roc_auc_score(y_binary, scores),
                "Precision": precision_score(y_true, pred, pos_label=POSITIVE_CLASS, zero_division=0),
                "Recall": recall_score(y_true, pred, pos_label=POSITIVE_CLASS, zero_division=0),
                "F1": f1_score(y_true, pred, pos_label=POSITIVE_CLASS, zero_division=0),
                "MCC": matthews_corrcoef(y_true, pred),
            }
            results.append(row)
            details[name] = (y_true, pred)

        st.subheader("Evaluation metrics")
        metrics_df = pd.DataFrame(results).set_index("Model")
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        st.download_button("Download metrics CSV", metrics_df.to_csv().encode("utf-8"), "evaluation_metrics.csv", "text/csv")

        st.subheader("Confusion matrix and classification report")
        detail_model = st.selectbox("Model for detailed report", model_names, key="detail_model")
        truth, prediction = details[detail_model]
        labels = ["<=50K", ">50K"]
        cm = confusion_matrix(truth, prediction, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
        st.dataframe(cm_df, use_container_width=True)
        st.text(classification_report(truth, prediction, labels=labels, zero_division=0))
    except Exception as exc:
        st.error(f"Unable to evaluate the uploaded file: {exc}")
else:
    st.info("Upload test_data.csv (included in the repository) to view model metrics and the confusion matrix.")
