# An Explainable Machine Learning Framework for Industrial Predictive Maintenance: Comparative Evaluation of Failure Prediction Models

> **Academic Research & Benchmark Demonstration** | PRG4206 Machine Learning  
> **Dataset**: AI4I 2020 Predictive Maintenance (10,000 Records, 28.5:1 Imbalance)  
> **Disclaimer**: This is an educational research benchmark using synthetic industrial data, not a validated real-world industrial safety system.

---

## 📌 Executive Summary & Abstract

Unplanned downtime in automated manufacturing causes severe financial losses, secondary component degradation, and critical worker safety hazards. This paper and software repository present an end-to-end Explainable Machine Learning (XAI) framework for predictive maintenance on computer numerical control (CNC) milling machinery.

### Research Contributions:
1. **Target & Identifier Leakage Prevention**: Strictly isolates six operational telemetry features (`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`), completely excluding five failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and row indices (`UDI`, `Product ID`).
2. **Comparative Empirical Evaluation**: Benchmarks three diverse classifier families (**Logistic Regression**, **Random Forest**, and **Support Vector Machine with RBF Kernel**) under extreme class imbalance (3.39% failures, 28.5:1 ratio) on an 80/20 stratified split.
3. **Dual-Scale Explainability**: Audits local individual machine risk using **SHAP TreeExplainer** with machine-precision probability additivity ($\Delta = 1.92 \times 10^{-14}$) alongside linear **Logit Decomposition** and non-linear **Validation Permutation Feature Importance**.
4. **Interactive Demonstration System**: An interactive Streamlit dashboard enabling real-time sensor simulation, multi-model consensus checks, and transparent feature attribution.

---

## 📂 Repository Structure

```
I26043415_Predictive Maintenance/
├── data/
│   └── ai4i2020.csv                    # AI4I 2020 dataset (10,000 rows × 14 columns)
├── models/
│   ├── logistic_regression.joblib      # Serialized Logistic Regression pipeline
│   ├── random_forest.joblib            # Serialized Random Forest pipeline
│   └── svm.joblib                      # Serialized Support Vector Machine (RBF) pipeline
├── results/
│   ├── dataset_summary.txt             # Dataset verification & distribution report
│   ├── model_metrics.csv               # Empirical benchmark results on held-out test set
│   └── svm_permutation_importance.csv  # Validation-fold permutation reliance table
├── figures/
│   ├── confusion_matrices.png          # Test set confusion matrices (all 3 models)
│   ├── model_comparison_metrics.png    # Comparative bar chart across 6 metrics
│   ├── roc_curves.png                  # Receiver Operating Characteristic (ROC-AUC)
│   ├── pr_curves.png                   # Precision-Recall Curves (Class Imbalance Metric)
│   └── svm_permutation_importance.png  # SVM validation-fold permutation importance
├── check_data.py                       # Initial dataset verification & validation script
├── train_models.py                     # Pipeline training, evaluation & artifact generation
├── compute_svm_permutation_importance.py # Validation-fold permutation reliance calculation
├── app.py                              # Interactive Streamlit demonstration dashboard
├── verify_app_logic.py                 # Comprehensive 11-point correctness test suite
├── requirements.txt                    # Project minimum dependencies
├── requirements-lock.txt               # Exact tested environment package lockfile
└── README.md                           # Project research paper documentation
```

---

## 📊 Empirical Benchmark Results (Held-Out Test Set: 2,000 Samples)

All models were evaluated on the identical 20% stratified test set (containing exactly 68 failure cases and 1,932 safe operations):

| Model Architecture | Accuracy | Precision (Fail) | Recall (Fail) | F1-Score (Fail) | ROC-AUC | PR-AUC (AP) | Primary Operational Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Support Vector Machine (SVM)** | `91.75%` | `27.85%` | **`89.71%`** | `0.4251` | **`0.9655`** | `0.6119` | **Max Fault Capture**: Detects 61/68 failures (7 false negatives). |
| **Random Forest (RF)** | **`98.00%`** | **`96.67%`** | `42.65%` | **`0.5918`** | `0.9635` | **`0.7642`** | **Min False Alarms**: Only 1 false alarm across 2,000 machines. |
| **Logistic Regression (LR)** | `82.45%` | `14.18%` | `82.35%` | `0.2419` | `0.9070` | `0.3818` | **Linear Baseline**: High sensitivity with 339 false alarms. |

### Operational Takeaway:
- In environments where unplanned stoppage costs exceed inspection costs by orders of magnitude, **SVM** is optimal due to superior recall.
- In high-throughput automated lines where unnecessary technician dispatches cause severe line interruptions, **Random Forest** provides the highest precision and F1-score.

---

## 🔍 Explainable AI (XAI) Framework

### 1. Random Forest Local Attribution (SHAP TreeExplainer)
- **Scale**: Probability space $[0, 1]$ (ensemble class probability leaf proportions).
- **Baseline Expected Value**: $\mathbb{E}[f(X)] \approx 0.4987$ (elevated by `class_weight='balanced'`).
- **Exact Additivity Axiom**:
  $$\mathbb{E}[f(X)] + \sum_{i=1}^8 \phi_i = \hat{P}(\text{Machine failure}=1)$$
  Verified on every sample with numerical error $\Delta < 2.1 \times 10^{-14}$.

### 2. Logistic Regression Logit Decomposition
- **Scale**: Log-odds / logit ($\ln(p/(1-p))$).
- **Exact Logit Reconstruction**:
  $$\text{Logit} = \beta_0 + \sum_{i=1}^8 \beta_i z_i = \text{decision\_function}(X)$$
- **Sigmoid Mapping**: $\sigma(\text{Logit}) = \frac{1}{1 + e^{-\text{Logit}}} = \hat{P}(\text{Fail} \mid X)$ verified with $0.00 \times 10^0$ error.
- Standardized $z$-scores ($\frac{x - \mu}{\sigma}$) are strictly separated from categorical dummy variables ($x \in \{0, 1\}$).

### 3. Support Vector Machine Margin & Permutation Reliance
- **Decision Score**: Signed margin distance $f(x) \in (-\infty, +\infty)$ in reproducing kernel Hilbert space (RKHS).
- **Probability Calibration**: Calibrated via Platt scaling (logistic sigmoid fitted over cross-validation).
- **Validation Permutation Importance**: Evaluated on a 1,600-sample validation fold ($n=10$ repeats, Average Precision drop). Identified as global model reliance, avoiding invalid linear projections.

---

## 🚀 Reproduction & Demonstration Workflow

### 1. Setup Environment
```powershell
# Navigate to directory
cd "C:\Users\tejag\Downloads\I26043415_Predictive Maintenance"

# Install dependencies (tested on Python 3.10.9)
pip install -r requirements.txt
```

### 2. Verify Dataset Integrity
```powershell
python check_data.py
```

### 3. Run Pipeline Training & Benchmarking
```powershell
python train_models.py
```

### 4. Compute SVM Validation Permutation Importance
```powershell
python compute_svm_permutation_importance.py
```

### 5. Execute 11-Point Verification Suite
```powershell
python verify_app_logic.py
```
*(All 11 checks pass with 0 failures).*

### 6. Launch the Interactive Web Demonstration
```powershell
python -m streamlit run app.py
```
*Open **`http://localhost:8501`** in any browser to interact with the dashboard.*

---

## ⚠️ Academic Safety Notice & Physical Limitations

1. **Synthetic Nature**: AI4I 2020 is synthesized to match physical manufacturing distributions. Real industrial cutting tools undergo non-stationary thermal drift, stochastic chatter, and continuous vibrational degradation.
2. **Snapshot Telemetry**: The models evaluate steady-state sensor snapshots rather than continuous vibration time series.
3. **Cost-Sensitive Thresholds**: In production deployments, decision thresholds should be tuned according to plant-specific cost matrices rather than default 50% cutoffs.
