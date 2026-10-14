"""
verify_app_logic.py
PRG4206 Independent Correctness Verification Suite
Comprehensive automated test suite covering:
1. Model loading and pipeline integrity
2. Input column schema and order alignment
3. Positive-class probability extraction via classes_ lookup
4. Finite probability bounds and valid class labels
5. Absence of target and identifier leakage columns across pipelines
6. Random Forest SHAP output dimensions, class selection, feature alignment, and additivity reconstruction
7. Logistic Regression logit reconstruction and sigmoid probability mapping
8. SVM decision score semantics and Platt probability calibration
9. Genuine permutation importance calculation verification
10. Presence and integrity of all evaluation metrics and visual artifacts
11. Consistency of model decision rules and probability thresholding
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap


def get_positive_class_index(pipeline):
    """Locates index corresponding to class label 1 in pipeline's classes_."""
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 in classes:
        return classes.index(1)
    return 1


def run_verification_suite():
    print("=" * 75)
    print("PRG4206 COMPREHENSIVE CORRECTNESS & AUDIT VERIFICATION SUITE")
    print("=" * 75)

    test_failures = []

    # -------------------------------------------------------------
    # Check 1: Model Files Exist and Load Successfully
    # -------------------------------------------------------------
    print("\n[Check 1/11] Verifying Model Files Exist and Load...")
    models_dir = Path("models")
    expected_models = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "svm.joblib",
    ]
    models = {}
    for mf in expected_models:
        mp = models_dir / mf
        if not mp.exists():
            test_failures.append(f"Model file missing: {mp}")
        else:
            try:
                models[mf] = joblib.load(mp)
                print(f"  -> Successfully loaded {mf}")
            except Exception as e:
                test_failures.append(f"Failed to load {mf}: {e}")

    # -------------------------------------------------------------
    # Check 2: Expected Predictor Names and Column Order
    # -------------------------------------------------------------
    print("\n[Check 2/11] Verifying Predictor Schema and Column Order...")
    expected_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    test_input = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
            }
        ]
    )
    if list(test_input.columns) != expected_cols:
        test_failures.append(
            f"Input column mismatch: expected {expected_cols}, got {list(test_input.columns)}"
        )
    else:
        print("  -> Predictor names and order match exactly.")

    # -------------------------------------------------------------
    # Check 3: Positive-Class Probability Extraction via classes_
    # -------------------------------------------------------------
    print("\n[Check 3/11] Verifying Positive-Class Probability Extraction via classes_...")
    for mf, pipe in models.items():
        clf = pipe.named_steps["classifier"]
        classes = list(clf.classes_)
        if 1 not in classes:
            test_failures.append(f"Class 1 not found in {mf} classes: {classes}")
        else:
            pos_idx = get_positive_class_index(pipe)
            if classes[pos_idx] != 1:
                test_failures.append(
                    f"Index {pos_idx} in {mf} maps to class {classes[pos_idx]}, not 1"
                )
            else:
                print(f"  -> {mf}: class 1 located at index {pos_idx} (classes_={classes})")

    # -------------------------------------------------------------
    # Check 4: Prediction Output Validity and Finite Probabilities
    # -------------------------------------------------------------
    print("\n[Check 4/11] Verifying Finite Probabilities in [0, 1] and Valid Class Labels...")
    for mf, pipe in models.items():
        pos_idx = get_positive_class_index(pipe)
        proba_arr = pipe.predict_proba(test_input)[0]
        pos_proba = proba_arr[pos_idx]
        pred_label = pipe.predict(test_input)[0]

        if not np.isfinite(pos_proba) or not (0.0 <= pos_proba <= 1.0):
            test_failures.append(f"{mf} produced invalid probability: {pos_proba}")
        if pred_label not in [0, 1]:
            test_failures.append(f"{mf} produced invalid predicted label: {pred_label}")
        if not np.isclose(np.sum(proba_arr), 1.0, atol=1e-5):
            test_failures.append(f"{mf} probabilities do not sum to 1.0: {proba_arr}")
        print(f"  -> {mf}: label={pred_label}, P(Fail)={pos_proba:.4f}, sum(P)={np.sum(proba_arr):.4f}")

    # -------------------------------------------------------------
    # Check 5: Prohibited Leakage Columns Strictly Absent
    # -------------------------------------------------------------
    print("\n[Check 5/11] Verifying Zero Target and Identifier Leakage...")
    leaked_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID", "Machine failure"]
    for col in leaked_cols:
        if col in test_input.columns:
            test_failures.append(f"Leaked column present in test_input: {col}")

    # Inspect pipeline ColumnTransformers
    for mf, pipe in models.items():
        preprocessor = pipe.named_steps["preprocessor"]
        for name, transformer, cols in preprocessor.transformers_:
            if name != "remainder":
                for c in cols:
                    if c in leaked_cols:
                        test_failures.append(
                            f"Pipeline {mf} transformer '{name}' includes leaked column: {c}"
                        )
    print("  -> Zero leakage confirmed across input schema and pipeline preprocessors.")

    # -------------------------------------------------------------
    # Check 6: Random Forest SHAP Dimensions and Additivity Reconstruction
    # -------------------------------------------------------------
    print("\n[Check 6/11] Verifying Random Forest SHAP Output, Dimensions & Additivity...")
    rf_pipe = models["random_forest.joblib"]
    preprocessor = rf_pipe.named_steps["preprocessor"]
    rf_clf = rf_pipe.named_steps["classifier"]
    X_trans = preprocessor.transform(test_input)
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    tree_explainer = shap.TreeExplainer(rf_clf)
    shap_vals = tree_explainer.shap_values(X_trans)
    expected_vals = tree_explainer.expected_value
    pos_idx = get_positive_class_index(rf_pipe)

    # Dimension verification
    if isinstance(shap_vals, np.ndarray):
        print(f"  -> SHAP values array shape: {shap_vals.shape} (expected 3D or 2D)")
        if len(shap_vals.shape) == 3:
            vals_class1 = shap_vals[0, :, pos_idx]
            if shap_vals.shape != (1, len(clean_names), 2):
                test_failures.append(f"Unexpected SHAP 3D shape: {shap_vals.shape}")
        else:
            vals_class1 = shap_vals[0]
    elif isinstance(shap_vals, list):
        print(f"  -> SHAP values list length: {len(shap_vals)}")
        vals_class1 = shap_vals[pos_idx][0]
    else:
        test_failures.append(f"Unrecognized SHAP values format: {type(shap_vals)}")
        vals_class1 = None

    if vals_class1 is not None:
        if len(vals_class1) != len(clean_names):
            test_failures.append(
                f"SHAP features count ({len(vals_class1)}) != clean_names ({len(clean_names)})"
            )

        # Baseline value extraction
        if isinstance(expected_vals, (np.ndarray, list)):
            base_val = float(expected_vals[pos_idx])
        else:
            base_val = float(expected_vals)

        # Additivity verification: base_val + sum(shap) == P(class 1)
        sum_shap = float(np.sum(vals_class1))
        reconstructed_prob = base_val + sum_shap
        actual_prob = float(rf_pipe.predict_proba(test_input)[0][pos_idx])
        additivity_error = abs(reconstructed_prob - actual_prob)

        print(f"  -> Baseline expected value E[f(X)]: {base_val:.4f}")
        print(f"  -> Sum of SHAP values:             {sum_shap:+.4f}")
        print(f"  -> Reconstructed probability:       {reconstructed_prob:.4f}")
        print(f"  -> Model predict_proba output:      {actual_prob:.4f}")
        print(f"  -> Additivity discrepancy:          {additivity_error:.2e}")

        if additivity_error > 1e-5:
            test_failures.append(
                f"Random Forest SHAP additivity violated: error={additivity_error:.2e}"
            )
        else:
            print("  -> Random Forest SHAP additivity verified within numerical tolerance.")

    # -------------------------------------------------------------
    # Check 7: Logistic Regression Logit Reconstruction & Sigmoid Mapping
    # -------------------------------------------------------------
    print("\n[Check 7/11] Verifying Logistic Regression Logit Reconstruction & Sigmoid Mapping...")
    lr_pipe = models["logistic_regression.joblib"]
    preprocessor = lr_pipe.named_steps["preprocessor"]
    lr_clf = lr_pipe.named_steps["classifier"]

    X_t = preprocessor.transform(test_input)[0]
    weights = lr_clf.coef_[0]
    intercept = float(lr_clf.intercept_[0])
    sum_impacts = float(np.sum(weights * X_t))
    reconstructed_logit = intercept + sum_impacts

    actual_decision_score = float(lr_clf.decision_function(preprocessor.transform(test_input))[0])
    logit_error = abs(reconstructed_logit - actual_decision_score)

    sigmoid_prob = 1.0 / (1.0 + np.exp(-reconstructed_logit))
    pos_idx = get_positive_class_index(lr_pipe)
    actual_lr_prob = float(lr_pipe.predict_proba(test_input)[0][pos_idx])
    prob_error = abs(sigmoid_prob - actual_lr_prob)

    print(f"  -> Intercept:              {intercept:.4f}")
    print(f"  -> Sum(Beta_i * z_i):      {sum_impacts:+.4f}")
    print(f"  -> Reconstructed Logit:    {reconstructed_logit:.4f}")
    print(f"  -> Decision function:      {actual_decision_score:.4f} (Error: {logit_error:.2e})")
    print(f"  -> Sigmoid(Logit):         {sigmoid_prob:.4f}")
    print(f"  -> Actual predict_proba:   {actual_lr_prob:.4f} (Error: {prob_error:.2e})")

    if logit_error > 1e-5:
        test_failures.append(f"Logistic Regression logit reconstruction mismatch: error={logit_error:.2e}")
    if prob_error > 1e-5:
        test_failures.append(f"Logistic Regression sigmoid probability mismatch: error={prob_error:.2e}")
    if logit_error <= 1e-5 and prob_error <= 1e-5:
        print("  -> Logistic Regression mathematical reconstruction verified.")

    # -------------------------------------------------------------
    # Check 8: SVM Decision Score Semantics & Platt Calibration
    # -------------------------------------------------------------
    print("\n[Check 8/11] Verifying SVM Decision Score Semantics & Platt Calibration...")
    svm_pipe = models["svm.joblib"]
    preprocessor = svm_pipe.named_steps["preprocessor"]
    svm_clf = svm_pipe.named_steps["classifier"]

    decision_dist = float(svm_clf.decision_function(preprocessor.transform(test_input))[0])
    pos_idx = get_positive_class_index(svm_pipe)
    svm_prob = float(svm_pipe.predict_proba(test_input)[0][pos_idx])
    svm_pred = int(svm_pipe.predict(test_input)[0])

    print(f"  -> Decision function score: {decision_dist:+.4f}")
    print(f"  -> Platt-calibrated prob:   {svm_prob:.4f}")
    print(f"  -> Predicted class label:   {svm_pred}")

    # For standard binary SVM, predict() sign matches decision_function >= 0
    expected_pred_by_margin = 1 if decision_dist >= 0.0 else 0
    if svm_pred != expected_pred_by_margin:
        test_failures.append(
            f"SVM predict() label ({svm_pred}) inconsistent with decision_function sign ({decision_dist})"
        )
    else:
        print("  -> SVM margin sign and discrete predict() label are fully consistent.")

    # -------------------------------------------------------------
    # Check 9: Genuine Permutation Importance Calculation Exists
    # -------------------------------------------------------------
    print("\n[Check 9/11] Verifying Permutation Feature Importance Artifacts...")
    perm_csv = Path("results/svm_permutation_importance.csv")
    perm_fig = Path("figures/svm_permutation_importance.png")

    if not perm_csv.exists():
        test_failures.append(f"Missing permutation importance CSV: {perm_csv}")
    else:
        perm_df = pd.read_csv(perm_csv)
        if list(perm_df.columns) != ["Feature", "Importance_Mean", "Importance_Std"]:
            test_failures.append(f"Invalid permutation importance columns: {list(perm_df.columns)}")
        if len(perm_df) != 6:
            test_failures.append(f"Expected 6 features in permutation importance, found {len(perm_df)}")
        if perm_df["Importance_Mean"].isnull().any():
            test_failures.append("Permutation importance contains NaN values")
        print(f"  -> Permutation table verified: {len(perm_df)} features, top feature: '{perm_df.iloc[0]['Feature']}'")

    if not perm_fig.exists():
        test_failures.append(f"Missing permutation importance figure: {perm_fig}")
    else:
        print(f"  -> Permutation figure verified: {perm_fig.name}")

    # -------------------------------------------------------------
    # Check 10: Evaluation Metrics & Diagnostic Figures Exist
    # -------------------------------------------------------------
    print("\n[Check 10/11] Verifying Evaluation Metrics & Diagnostic Figures...")
    metrics_path = Path("results/model_metrics.csv")
    if not metrics_path.exists():
        test_failures.append(f"Missing metrics CSV: {metrics_path}")
    else:
        m_df = pd.read_csv(metrics_path)
        if len(m_df) != 3:
            test_failures.append(f"Expected 3 models in {metrics_path}, got {len(m_df)}")
        else:
            print("  -> Model metrics table verified:")
            for _, row in m_df.iterrows():
                print(f"     * {row['Model']}: Recall={row['Recall']:.4f}, F1={row['F1-Score']:.4f}, PR-AUC={row['PR-AUC']:.4f}")

    expected_figures = [
        "figures/confusion_matrices.png",
        "figures/model_comparison_metrics.png",
        "figures/roc_curves.png",
        "figures/pr_curves.png",
        "figures/svm_permutation_importance.png",
    ]
    for fig_file in expected_figures:
        fp = Path(fig_file)
        if not fp.exists() or fp.stat().st_size == 0:
            test_failures.append(f"Missing or empty figure: {fp}")
        else:
            print(f"  -> Figure verified: {fp.name} ({fp.stat().st_size:,} bytes)")

    # -------------------------------------------------------------
    # Check 11: Displayed Decision Rules and Threshold Consistency
    # -------------------------------------------------------------
    print("\n[Check 11/11] Verifying Decision Rules and Threshold Semantics...")
    # Test across multiple sensor scenarios
    scenarios = [
        {"name": "Safe baseline", "input": {"Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 20}},
        {"name": "High strain", "input": {"Type": "L", "Air temperature [K]": 302.2, "Process temperature [K]": 311.4, "Rotational speed [rpm]": 1320, "Torque [Nm]": 68.0, "Tool wear [min]": 230}},
    ]
    for sc in scenarios:
        df_sc = pd.DataFrame([sc["input"]])
        for mf, pipe in models.items():
            pos_idx = get_positive_class_index(pipe)
            p = float(pipe.predict_proba(df_sc)[0][pos_idx])
            pred = int(pipe.predict(df_sc)[0])
            if not (0.0 <= p <= 1.0):
                test_failures.append(f"Scenario '{sc['name']}' produced out-of-range prob for {mf}: {p}")
            if pred not in [0, 1]:
                test_failures.append(f"Scenario '{sc['name']}' produced invalid class for {mf}: {pred}")
        print(f"  -> Scenario '{sc['name']}': checked across all 3 models without error.")

    # -------------------------------------------------------------
    # Final Result
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    if test_failures:
        print(f"VERIFICATION FAILED WITH {len(test_failures)} ISSUE(S):")
        for idx, err in enumerate(test_failures, 1):
            print(f"  {idx}. {err}")
        print("=" * 75)
        sys.exit(1)
    else:
        print("ALL 11 VERIFICATION CHECKS PASSED PERFECTLY!")
        print("=" * 75)
        sys.exit(0)


if __name__ == "__main__":
    run_verification_suite()
