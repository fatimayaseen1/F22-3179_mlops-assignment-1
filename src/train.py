import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops-assi-1")

# -----------------------------
# Load dataset and split
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Ensure directories exist
# -----------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------
# Training and logging function
# -----------------------------
def train_and_log(model, model_name, dataset_name="Iris"):
    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n📊 {model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # Log hyperparameters
    # -----------------------------
    mlflow.log_params(model.get_params())
    mlflow.set_tag("model_type", type(model).__name__)
    mlflow.set_tag("dataset", dataset_name)

    # Log metrics
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # -----------------------------
    # Save & log model
    # -----------------------------
    model_path = f"models/{model_name}_model.pkl"
    joblib.dump(model, model_path)

    mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,           # ✅ avoids deprecation
        input_example=X_test[:1]   # ✅ fixes signature warning
    )
    mlflow.log_artifact(model_path, artifact_path="saved_models")

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - Confusion Matrix")
    cm_path = f"results/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # -----------------------------
    # Evaluation CSV
    # -----------------------------
    eval_data = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    eval_path = f"results/{model_name}_evaluation.csv"
    eval_data.to_csv(eval_path, index=False)
    mlflow.log_artifact(eval_path, artifact_path="evaluation")

    print(f"✅ {model_name} training and logging complete.")

    return {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


# -----------------------------
# Models to train
# -----------------------------
models = [
    LogisticRegression(max_iter=200),
    RandomForestClassifier(n_estimators=100),
    SVC(kernel="linear", probability=True)
]

# -----------------------------
# Train & collect metrics + run_ids
# -----------------------------
all_results = []
model_runs = {}   # store run_ids for each model

for model in models:
    with mlflow.start_run(run_name=type(model).__name__) as run:
        result = train_and_log(model, model_name=type(model).__name__)
        all_results.append(result)
        model_runs[type(model).__name__] = run.info.run_id   # ✅ save run_id


# -----------------------------
# Comparison Run
# -----------------------------
with mlflow.start_run(run_name="Comparison_All_Models"):
    results_df = pd.DataFrame(all_results)
    print("\n📊 Model Comparison Table:\n", results_df)

    # Save CSV
    comp_csv_path = "results/model_comparison.csv"
    results_df.to_csv(comp_csv_path, index=False)
    mlflow.log_artifact(comp_csv_path, artifact_path="comparison")

    # Bar plot
    results_df.set_index("model")[["accuracy", "precision", "recall", "f1_score"]].plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    comp_plot_path = "results/model_comparison.png"
    plt.savefig(comp_plot_path)
    plt.close()
    mlflow.log_artifact(comp_plot_path, artifact_path="comparison")

    print("✅ Comparison run logged successfully.")


# -----------------------------
# Register Models in Registry
# -----------------------------
client = MlflowClient()
for model_name, run_id in model_runs.items():
    model_uri = f"runs:/{run_id}/{model_name}"
    mv = mlflow.register_model(model_uri, model_name)
    print(f"✅ Registered {model_name} as version {mv.version} in MLflow Model Registry.")
