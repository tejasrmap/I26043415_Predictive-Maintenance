# PRG4206 Explainable Machine Learning for Predictive Maintenance

A comprehensive, leakage-safe machine learning and Explainable AI (XAI) framework for predicting industrial equipment failure using the **AI4I 2020 Predictive Maintenance Dataset**.

---

## 📋 Submission Checklist & Deliverables

This repository contains all components required for coursework submission and end-to-end evaluation:

- [x] **All Source Code**:
  - `app.py`: Interactive Streamlit dashboard for real-time inference, scenario simulation, and XAI breakdowns.
  - `train_models.py`: Complete pipeline script for stratified data splitting, preprocessing, model fitting, metric logging, and diagnostic plot generation.
  - `verify_app_logic.py`: Independent 11-point correctness verification suite validating data leakage absence, SHAP additivity, logit reconstruction, decision rules, and file integrity.
  - `compute_svm_permutation_importance.py`: Offline permutation feature importance scoring for SVM with cross-validation confidence intervals.
  - `check_data.py`: Dataset integrity, column schema, target balance, and summary statistics inspection.
  - `test.py`: Standalone figure generation for dataset target distribution and architectural workflow diagrams.
- [x] **README.md**: Comprehensive setup, replication, verification, and navigation instructions.
- [x] **requirements.txt**: Fully pinned dependencies with tested minimum version constraints.
- [x] **Sample Data & Download Instructions**: Complete benchmark dataset included at `data/ai4i2020.csv` with official UCI repository source citation and DOI.
- [x] **Saved Model Files**: Serialized scikit-learn pipelines pre-trained and included in `models/` (total size ~3.8 MB, well within git/submission limits).

---

## 📂 Repository Structure

```text
.
├── data/
│   └── ai4i2020.csv                      # AI4I 2020 benchmark dataset (10,000 rows × 14 columns)
├── models/                               # Serialized scikit-learn fitted pipelines
│   ├── logistic_regression.joblib        # Scaled L-BFGS Logistic Regression (3.5 KB)
│   ├── random_forest.joblib              # 100-tree balanced Random Forest (3.65 MB)
│   └── svm.joblib                        # Platt-calibrated RBF Kernel SVM (147 KB)
├── figures/                              # Diagnostic benchmark figures & ROC/PR curves
│   ├── confusion_matrices.png            # Normalized and raw confusion matrices
│   ├── model_comparison_metrics.png      # Comparative bar charts across 6 evaluation metrics
│   ├── roc_curves.png                    # Receiver Operating Characteristic curves with AUC
│   ├── pr_curves.png                     # Precision-Recall curves with PR-AUC / Average Precision
│   └── svm_permutation_importance.png    # SVM out-of-fold permutation importance rankings
├── results/                              # Quantitative audit results & evaluation CSVs
│   ├── dataset_summary.txt               # Statistical distributions and missing value audit
│   ├── model_metrics.csv                 # Exact held-out test evaluation metrics
│   └── svm_permutation_importance.csv    # Computed permutation importance scores
├── app.py                                # Interactive Streamlit dashboard application
├── train_models.py                       # End-to-end training, evaluation & artifact generation script
├── verify_app_logic.py                   # Automated 11-point correctness verification test suite
├── compute_svm_permutation_importance.py # Script for computing SVM permutation feature importance
├── check_data.py                         # Dataset sanity check and validation script
├── test.py                               # Script for rendering target distribution & architecture figures
├── system_architecture.png               # High-level system architecture diagram
├── target_distribution.png               # Target class distribution chart (96.61% vs 3.39%)
├── requirements.txt                      # Project dependency specification
├── requirements-lock.txt                 # Exact environment lockfile
├── .gitignore                            # Python, cache, and OS ignore rules
└── README.md                             # Project documentation and replication guide
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.10.x and 3.11.x)
- `pip` package manager

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Lineage & Download Instructions

The dataset used in this project is the **AI4I 2020 Predictive Maintenance Dataset**:
- **Source**: UCI Machine Learning Repository
- **DOI**: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- **URL**: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Characteristics**: 10,000 observations representing industrial milling machine telemetry.
- **Target**: `Machine failure` (binary: 0 = No failure [9,661 / 96.61%], 1 = Failure [339 / 3.39%]).
- **Leakage Controls**:
  - `UDI` and `Product ID` are dropped to prevent identity/ordering leakage.
  - Failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are dropped because they represent post-failure diagnostic labels that leak the target.
  - The model uses strictly 6 operational predictors: `Type` (categorical: L, M, H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`.

*Note: The complete dataset is already included in this repository under `data/ai4i2020.csv` for immediate replication.*

---

## 🚀 Execution & Replication Guide

### Step 1: Run Independent Verification Suite (11 Checks)
Run the automated test suite to confirm end-to-end correctness, schema validation, zero leakage, mathematical reconstruction of SHAP additivity, Logistic Regression logits, and SVM calibration:

```bash
python verify_app_logic.py
```
*Expected Output: `ALL 11 VERIFICATION CHECKS PASSED PERFECTLY!`*

### Step 2: Retrain Models & Regenerate Artifacts (Optional)
To retrain all three model pipelines from scratch and regenerate all figures and metrics:

```bash
python train_models.py
```
This executes:
1. Stratified 80/20 train/test split (`random_state=42`).
2. Training-partition-only feature scaling (`StandardScaler`) and encoding (`OneHotEncoder`).
3. Model fitting with balanced class weighting:
   - Logistic Regression (`L-BFGS`, `max_iter=1000`)
   - Random Forest (`n_estimators=100`, `random_state=42`)
   - Support Vector Machine (`RBF kernel`, `probability=True`, `random_state=42`)
4. Metrics logging to `results/model_metrics.csv`.
5. Saving evaluation figures to `figures/`.
6. Serializing pipeline models to `models/*.joblib`.

### Step 3: Launch the Streamlit Dashboard
Start the interactive application locally:

```bash
streamlit run app.py
```
Once launched, open your web browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Features

1. **Real-Time Predictive Inference**:
   - Preset telemetry scenarios (Nominal Operation, Overstrain Overload, Extreme Thermal Dissipation Strain).
   - Interactive numeric sliders with live probability outputs, health status badges, and threshold-sensitive alert indicators.
2. **Explainable AI (XAI) Attribution**:
   - **Random Forest**: TreeExplainer SHAP waterfall plots and ranked impact charts verifying exact additivity $\sum \phi_i = P(\text{failure}) - E[P]$.
   - **Logistic Regression**: Linear log-odds decomposition showing directional feature push ($\beta_i \cdot z_i$) converted via the sigmoid function.
   - **Support Vector Machine**: Out-of-fold permutation feature importance with confidence intervals.
3. **Model Evaluation & Benchmarks**:
   - Side-by-side performance table highlighting trade-offs between precision and recall under extreme class imbalance (3.39% failure prevalence).
   - High-resolution ROC curves, Precision-Recall curves, and confusion matrices.
4. **Methodology & Governance**:
   - Complete documentation of leakage prevention, class weighting rationale, feature schema, and operational safety limitations.

---

## 📦 Model Artifact Sizes & Integrity

All trained models are serialized as standard scikit-learn `Pipeline` objects preserving both preprocessing transformers and estimators:
- `models/logistic_regression.joblib`: 3.5 KB
- `models/random_forest.joblib`: 3.65 MB
- `models/svm.joblib`: 147 KB

Because all models are compact (< 4 MB total), they are committed directly in the repository and included in the distribution package without requiring external cloud downloads.

---

## 📄 Academic Integrity & License

This project was developed for coursework evaluation under the MIT License.
