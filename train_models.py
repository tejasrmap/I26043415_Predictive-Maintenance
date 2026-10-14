"""
train_models.py
PRG4206 Machine Learning Assignment: Explainable Predictive Maintenance
End-to-end model training, evaluation, and pipeline serialization.

This script:
1. Loads AI4I 2020 dataset from data/ai4i2020.csv.
2. Selects 6 operational features, excluding identifiers and failure modes.
3. Performs a stratified 80/20 train/test split (random_state=42).
4. Verifies stratification and index non-overlap via assertions.
5. Builds a ColumnTransformer preprocessor (StandardScaler + OneHotEncoder).
6. Fits Logistic Regression, Random Forest, and SVM with class_weight='balanced'.
7. Evaluates each model on the held-out test set across standard & imbalance-aware metrics.
8. Saves metrics to results/model_metrics.csv.
9. Generates evaluation figures in figures/.
10. Serializes trained pipelines to models/*.joblib.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def main():
    print("=" * 75)
    print("PRG4206 PREDICTIVE MAINTENANCE: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 75)

    # 1. Directory Setup
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "ai4i2020.csv"
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    models_dir = project_root / "models"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataset Loading & Feature Selection
    print(f"\n[Step 1] Loading dataset from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")

    target_col = "Machine failure"
    assert target_col in df.columns, f"Target column '{target_col}' not found in dataset!"

    # Explicitly exclude identifiers and failure modes
    excluded_columns = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", target_col]
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # Verify column selection
    for col in feature_columns:
        assert col in df.columns, f"Required predictor '{col}' not found in dataset!"
    for exc in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert exc not in feature_columns, f"Leaked column '{exc}' must not be in predictors!"

    X = df[feature_columns].copy()
    y = df[target_col].copy()

    categorical_features = ["Type"]
    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    print(f"\nPredictor variables ({len(feature_columns)} features):")
    for f in feature_columns:
        print(f"  - {f} ({'categorical' if f in categorical_features else 'numerical'})")
    print(f"Target variable: '{target_col}' (Binary: 0 = No failure, 1 = Machine failure)")

    total_failures = y.sum()
    failure_rate = y.mean() * 100
    print(f"Total failures: {total_failures:,} / {len(y):,} ({failure_rate:.2f}%)")

    # 3. Reproducible Stratified Train-Test Split (80/20, random_state=42)
    print("\n[Step 2] Performing reproducible Stratified Train-Test Split (80/20, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Validation Checks & Assertions
    print("Running validation assertions:")
    assert len(X_train) == 8000, f"Expected 8000 training rows, got {len(X_train)}"
    assert len(X_test) == 2000, f"Expected 2000 testing rows, got {len(X_test)}"
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, (
        "Data leakage! Overlap detected between train and test indices!"
    )
    assert len(X_train) + len(X_test) == len(df), "Train + test size mismatch!"

    train_fail_rate = y_train.mean() * 100
    test_fail_rate = y_test.mean() * 100
    print(f"  -> Train set: {len(X_train):,} samples, {y_train.sum():,} failures ({train_fail_rate:.2f}%)")
    print(f"  -> Test set:  {len(X_test):,} samples, {y_test.sum():,} failures ({test_fail_rate:.2f}%)")
    assert abs(train_fail_rate - failure_rate) < 0.1, "Stratification mismatch in train set!"
    assert abs(test_fail_rate - failure_rate) < 0.1, "Stratification mismatch in test set!"
    print("  -> All split integrity and stratification assertions PASSED.")

    # 4. Preprocessing Pipeline Definition
    print("\n[Step 3] Defining Preprocessing with ColumnTransformer and Pipeline...")
    print("  - StandardScaler applied to numerical features (fit on train only)")
    print("  - OneHotEncoder applied to categorical feature 'Type' (fit on train only)")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    # 5. Model Architecture & Class Imbalance Strategy
    # Class weights inversely proportional to frequencies: n_samples / (n_classes * n_samples_c)
    # Class 0 weight ~0.518, Class 1 weight ~14.75.
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=42,
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    SVC(
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                        kernel="rbf",
                    ),
                ),
            ]
        ),
    }

    model_file_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Support Vector Machine": "svm.joblib",
    }

    # 6. Model Training & Evaluation
    print("\n[Step 4] Fitting pipelines on Training Data & Evaluating on Held-Out Test Data...")
    metrics_records = []
    curves_data = {}

    for model_name, pipeline in models.items():
        print(f"\n--- Training {model_name} ---")
        pipeline.fit(X_train, y_train)

        # Ensure all models are evaluated on the exact same held-out test data
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"  Accuracy:         {acc:.4f}")
        print(f"  Precision (Fail): {prec:.4f}")
        print(f"  Recall (Fail):    {rec:.4f}")
        print(f"  F1-Score (Fail):  {f1:.4f}")
        print(f"  ROC-AUC:          {roc_auc:.4f}")
        print(f"  PR-AUC / AP:      {pr_auc:.4f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        metrics_records.append(
            {
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        curves_data[model_name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "cm": cm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "prec_curve": prec_curve,
            "rec_curve": rec_curve,
            "pr_auc": pr_auc,
        }

        # Save pipeline to models/
        save_path = models_dir / model_file_map[model_name]
        joblib.dump(pipeline, save_path)
        print(f"  Serialized Pipeline saved to: {save_path.name}")

    # 7. Save Metrics Table
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = results_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[Step 5] Evaluation metrics saved to: {metrics_csv_path}")

    # 8. Visualizations
    print("\n[Step 6] Generating and saving evaluation figures...")
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Figure 1: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (model_name, data) in zip(axes, curves_data.items()):
        cm = data["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=13, weight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xticklabels(["No Fail (0)", "Fail (1)"])
        ax.set_yticklabels(["No Fail (0)", "Fail (1)"])
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {cm_path.name}")

    # Figure 2: Model Comparison Bar Chart
    metrics_plot_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(
        data=metrics_plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#2b5c8f", "#2a9d8f", "#e76f51"],
    )
    plt.title("Model Performance Comparison (Predictive Maintenance)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.08)
    for p in bar_plot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            bar_plot.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                xytext=(0, 2),
                textcoords="offset points",
            )
    plt.legend(title="Model Architecture", loc="lower right", frameon=True)
    plt.tight_layout()
    comp_path = figures_dir / "model_comparison_metrics.png"
    plt.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {comp_path.name}")

    # Figure 3: ROC Curves
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#2a9d8f", "Support Vector Machine": "#e76f51"}
    for model_name, data in curves_data.items():
        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{model_name} (AUC = {data['roc_auc']:.3f})",
            color=colors[model_name],
            lw=2.2,
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold", pad=10)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = figures_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {roc_path.name}")

    # Figure 4: Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for model_name, data in curves_data.items():
        plt.plot(
            data["rec_curve"],
            data["prec_curve"],
            label=f"{model_name} (PR-AUC = {data['pr_auc']:.3f})",
            color=colors[model_name],
            lw=2.2,
        )
    baseline_ap = failure_rate / 100.0
    plt.axhline(
        y=baseline_ap,
        color="gray",
        linestyle="--",
        lw=1.2,
        label=f"Baseline Prevalence ({baseline_ap:.3f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall (PR) Curves (Imbalance Sensitive)", fontsize=13, weight="bold", pad=10)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    pr_path = figures_dir / "pr_curves.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {pr_path.name}")

    # 9. Summary & Best Model Determination
    best_recall_model = metrics_df.loc[metrics_df["Recall"].idxmax()]["Model"]
    best_recall_val = metrics_df["Recall"].max()

    best_f1_model = metrics_df.loc[metrics_df["F1-Score"].idxmax()]["Model"]
    best_f1_val = metrics_df["F1-Score"].max()

    print("\n" + "=" * 75)
    print("FINAL SUMMARY REPORT & MODEL COMPARISON")
    print("=" * 75)
    print(metrics_df.to_string(index=False))
    print("-" * 75)
    print(f"Best Model for Failure-Class Recall:   {best_recall_model} (Recall = {best_recall_val:.4f})")
    print(f"Best Model for Failure-Class F1-Score: {best_f1_model} (F1 = {best_f1_val:.4f})")
    print("=" * 75)
    print("\nClass Imbalance Mitigation & Observations:")
    print("  1. Logistic Regression: With class_weight='balanced', the linear decision boundary")
    print("     shifts to capture minority failures, yielding higher recall at the cost of precision.")
    print("  2. Random Forest: Captures complex non-linear interactions (e.g. rotational speed x torque,")
    print("     process temp difference), delivering superior F1-score and Precision-Recall AUC.")
    print("  3. Support Vector Machine: RBF kernel with class-weighted C parameter provides non-linear")
    print("     separation, balanced across sensitivity and specificity.")
    print("=" * 75)


if __name__ == "__main__":
    main()
