from comet_ml import Experiment
import pandas as pd
import time
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# COMET CONFIG (NO .env FILE, NO HARDCODED API KEY)
# =========================================================
# IMPORTANT:
# Set your API key in terminal before running:
# Windows (CMD):   set COMET_API_KEY=your_api_key
# PowerShell:      $env:COMET_API_KEY="your_api_key"
# Mac/Linux:       export COMET_API_KEY=your_api_key

COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_PROJECT_NAME = "breast-cancer-benchmark"
COMET_WORKSPACE = "saurav-kumavat"

if not COMET_API_KEY:
    raise ValueError(
        "❌ COMET_API_KEY not found.\n"
        "Set it in terminal before running.\n"
        "Windows CMD: set COMET_API_KEY=your_api_key\n"
        "PowerShell:  $env:COMET_API_KEY='your_api_key'\n"
        "Mac/Linux:   export COMET_API_KEY=your_api_key"
    )


# =========================================================
# COMET EXPERIMENT
# =========================================================
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORKSPACE
)

experiment.set_name("Breast Cancer Multi-Model Benchmark")

# Log general project info
experiment.log_parameter("dataset", "sklearn_breast_cancer")
experiment.log_parameter("test_size", 0.2)
experiment.log_parameter("random_state", 42)


# =========================================================
# LOAD DATASET
# =========================================================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Log dataset info
experiment.log_dataset_hash(X)
experiment.log_parameter("num_samples", X.shape[0])
experiment.log_parameter("num_features", X.shape[1])


# =========================================================
# DATA SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# MODELS
# - Scaling for Logistic Regression and SVM
# - random_state for reproducibility
# =========================================================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),

    "DecisionTree": DecisionTreeClassifier(
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
}


# =========================================================
# RESULTS STORAGE
# =========================================================
results = []
best_model_name = None
best_f1 = -1


# =========================================================
# TRAIN & EVALUATE MODELS
# =========================================================
for name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print(f"{'=' * 60}")

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    train_time = end - start
    pred = model.predict(X_test)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # ROC / AUC
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # Store results
    results.append([name, acc, prec, rec, f1, roc_auc, train_time])

    # Best model tracking based on F1 Score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

    # -----------------------------------------------------
    # PRINT METRICS
    # -----------------------------------------------------
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print(f"Training Time : {train_time:.4f} sec")

    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=data.target_names))

    # -----------------------------------------------------
    # LOG METRICS TO COMET
    # -----------------------------------------------------
    experiment.log_metric(f"{name}_accuracy", acc)
    experiment.log_metric(f"{name}_precision", prec)
    experiment.log_metric(f"{name}_recall", rec)
    experiment.log_metric(f"{name}_f1", f1)
    experiment.log_metric(f"{name}_roc_auc", roc_auc)
    experiment.log_metric(f"{name}_train_time", train_time)

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    experiment.log_figure(figure_name=f"{name}_confusion_matrix")
    plt.close()

    # -----------------------------------------------------
    # ROC CURVE
    # -----------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    experiment.log_figure(figure_name=f"{name}_roc_curve")
    plt.close()

    # -----------------------------------------------------
    # FEATURE IMPORTANCE (TREE-BASED MODELS ONLY)
    # -----------------------------------------------------
    actual_model = model
    if isinstance(model, Pipeline):
        actual_model = model.named_steps["model"]

    if hasattr(actual_model, "feature_importances_"):
        importance = actual_model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feature_importance_df.head(15),
            x="Importance",
            y="Feature"
        )
        plt.title(f"{name} - Top 15 Feature Importance")
        plt.tight_layout()

        experiment.log_figure(figure_name=f"{name}_feature_importance")
        plt.close()

    # -----------------------------------------------------
    # LOG MODEL (OPTIONAL)
    # Some Comet versions can error on sklearn Pipeline objects.
    # If you get an error here, comment this line.
    # -----------------------------------------------------
    try:
        experiment.log_model(name, model)
    except Exception as e:
        print(f"⚠️ Could not log model '{name}' to Comet: {e}")


# =========================================================
# RESULTS DATAFRAME
# =========================================================
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Training Time (sec)"
    ]
)

# Sort by F1 Score descending
results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

print(f"\n{'=' * 60}")
print("MODEL COMPARISON")
print(f"{'=' * 60}")
print(results_df)

# Log results table
experiment.log_table(
    filename="model_comparison.csv",
    tabular_data=results_df
)

# =========================================================
# MODEL COMPARISON PLOTS
# =========================================================

# Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
experiment.log_figure(figure_name="model_accuracy_comparison")
plt.close()

# F1 Score Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title("Model F1 Score Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
experiment.log_figure(figure_name="model_f1_comparison")
plt.close()

# ROC-AUC Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="ROC-AUC", data=results_df)
plt.title("Model ROC-AUC Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
experiment.log_figure(figure_name="model_roc_auc_comparison")
plt.close()


# =========================================================
# SAVE RESULTS LOCALLY
# =========================================================
results_df.to_csv("model_results.csv", index=False)
experiment.log_asset("model_results.csv")


# =========================================================
# BEST MODEL SUMMARY
# =========================================================
print(f"\n🏆 Best Model based on F1 Score: {best_model_name} ({best_f1:.4f})")

experiment.log_parameter("best_model", best_model_name)
experiment.log_metric("best_model_f1", best_f1)


# =========================================================
# END EXPERIMENT
# =========================================================
experiment.end()

print("\n✅ Experiment completed successfully!")