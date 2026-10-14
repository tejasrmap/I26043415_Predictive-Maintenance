"""
compute_svm_permutation_importance.py
PRG4206 Explainable Predictive Maintenance
Calculates genuine Permutation Feature Importance for the fitted SVM pipeline
on held-out validation data (not the final test set).

Saves:
- results/svm_permutation_importance.csv
- figures/svm_permutation_importance.png
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("COMPUTING SVM PERMUTATION FEATURE IMPORTANCE (VALIDATION FOLD)")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "ai4i2020.csv"
    svm_path = project_root / "models" / "svm.joblib"
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    df = pd.read_csv(data_path)
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    target_col = "Machine failure"

    X = df[feature_columns].copy()
    y = df[target_col].copy()

    # 2. Replicate training split: 80% train, 20% held-out test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Create held-out validation fold from training data (20% of train = 1,600 samples)
    # The final test set (X_test) is NEVER used for permutation importance calculation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
    )

    print(f"Validation fold size: {len(X_val):,} samples ({y_val.sum():,} failure cases)")

    # 4. Load fitted SVM pipeline
    svm_pipeline = joblib.load(svm_path)

    # 5. Compute Permutation Importance on validation fold
    # Scoring metric: average_precision (PR-AUC), sensitive to imbalanced positive class
    print("Calculating permutation importance (scoring='average_precision', n_repeats=10, random_state=42)...")
    perm_result = permutation_importance(
        svm_pipeline,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    # 6. Format and save results table
    perm_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance_Mean": perm_result.importances_mean,
            "Importance_Std": perm_result.importances_std,
        }
    ).sort_values("Importance_Mean", ascending=False)

    csv_out = results_dir / "svm_permutation_importance.csv"
    perm_df.to_csv(csv_out, index=False)
    print(f"\n[SAVED] Table saved to: {csv_out}")
    print(perm_df.to_string(index=False))

    # 7. Generate visualization
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid", font_scale=1.0)
    
    # Plot horizontal bar with error bars
    y_pos = np.arange(len(perm_df))
    plt.barh(
        y_pos,
        perm_df["Importance_Mean"],
        xerr=perm_df["Importance_Std"],
        align="center",
        color="#e76f51",
        ecolor="#4a4e69",
        capsize=4,
        alpha=0.88,
        height=0.55,
    )
    plt.yticks(y_pos, perm_df["Feature"])
    plt.gca().invert_yaxis()  # Highest on top
    plt.xlabel("Mean Score Drop in Average Precision (PR-AUC) [± 1 Std]", fontsize=11)
    plt.title(
        "SVM Permutation Feature Importance (Global Model Reliance)\nEvaluated on Held-Out Validation Fold (n_repeats=10)",
        fontsize=12,
        weight="bold",
        pad=10,
    )
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
    plt.tight_layout()

    fig_out = figures_dir / "svm_permutation_importance.png"
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Figure saved to: {fig_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
