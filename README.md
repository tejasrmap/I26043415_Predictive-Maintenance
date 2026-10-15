# Predictive Maintenance Dashboard

## Overview

This repository contains a **Streamlit** dashboard for predictive maintenance of industrial equipment. The application demonstrates explainable AI (XAI) techniques for three distinct machine learning models:

- **Logistic Regression**: Linear coefficients visualized as log-odds contributions and odds ratios.
- **Random Forest**: SHAP waterfall and ranked-impact bar charts showing exact probability additivity and feature contributions to predicted failure.
- **Support Vector Machine (SVM)**: Permutation importance scores with validation-fold confidence intervals and decision function analysis.

The dashboard also features:
- **Real-Time Inference**: Interactive scenario simulation with real-time risk scores and status alerts.
- **Model Evaluation & Benchmarks**: Confusion matrices, ROC curves, calibration curves, and cross-model performance comparisons.
- **Methodology & Dataset Limitations**: Data lineage, preprocessing documentation, zero-leakage validation constraints, and operational caveats.

---

## Repository Structure

```text
.
├── app.py                                # Main Streamlit application
├── models/                               # Trained scikit-learn model pipelines
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── svm.joblib
├── figures/                              # Diagnostic and benchmark visualizations
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── calibration_curves.png
│   └── feature_importance_comparison.png
├── verify_app_logic.py                   # Automated 11-point verification test suite
├── compute_svm_permutation_importance.py # Script for offline SVM permutation scoring
├── check_data.py                         # Dataset sanity check and validation script
├── .gitignore                            # Cache and environment ignore rules
└── README.md                             # Project documentation
```

---

## Prerequisites

- **Python 3.10+**
- Virtual environment manager (`venv`, `conda`, or `uv`)

---

## Installation & Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd "I26043415_Predictive Maintenance"
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate.bat
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install streamlit scikit-learn pandas numpy matplotlib seaborn shap joblib
   ```

---

## Running the Dashboard

Launch the application with Streamlit:

```bash
streamlit run app.py
```

The app will start a local web server (usually at `http://localhost:8501`). Open this address in any modern web browser.

---

## Application Navigation

The dashboard is structured into four main tabs:

1. **Real-Time Inference**:
   - Select operational presets (e.g., Nominal Operation, High Torque Overload, Extreme Heat Dissipation Strain) or adjust sensor sliders manually.
   - Inspect instant failure risk probabilities, operational health indicators, and safety thresholds.

2. **Explainable AI (Attribution & SHAP)**:
   - **Random Forest (SHAP)**: Waterfall and ranked-impact bar charts showing exact probability additivity ($\sum \phi_i = P(\text{failure}) - E[P]$) with explicit reference baselines.
   - **Logistic Regression**: Log-odds component breakdown displaying directional push toward or away from failure.
   - **Support Vector Machine**: Out-of-fold permutation importance metrics with confidence intervals for non-linear RBF decision surfaces.

3. **Model Evaluation & Benchmarks**:
   - Review hold-out test metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
   - Side-by-side diagnostic figures for confusion matrices, calibration curves, and ROC curves.

4. **Methodology & Limitations**:
   - Review preprocessing steps, class imbalance handling, and leakage prevention rules (exclusion of `UDI`, `Product ID`, and post-failure flags `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).

---

## Verification & Testing

An automated 11-point test suite is included to ensure end-to-end mathematical consistency, widget stability, and absence of data leakage:

```bash
python verify_app_logic.py
```

---

## License

This project is licensed under the MIT License.
