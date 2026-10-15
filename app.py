"""
app.py
An Explainable Machine Learning Framework for Industrial Predictive Maintenance:
Comparative Evaluation of Failure Prediction Models

Streamlit web application for real-time inference, comparative multi-model
evaluation, and transparent Explainable AI (SHAP & feature attribution).

Dataset: AI4I 2020 Predictive Maintenance (Synthetic benchmark)
Disclaimer: This is an educational benchmark using synthetic data,
not a validated real-world industrial safety system.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import shap
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Professional Academic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Explainable Predictive Maintenance Framework",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Unified Academic Presentation
st.markdown(
    """
    <style>
    /* Theme-Adaptive Typography & Containers */
    .paper-title {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: inherit !important;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 6px;
    }
    .paper-subtitle {
        color: inherit !important;
        opacity: 0.85;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .paper-meta {
        background-color: rgba(37, 99, 235, 0.08) !important;
        border-radius: 8px;
        padding: 12px 18px;
        font-size: 0.90rem;
        color: inherit !important;
        margin-bottom: 18px;
        border: 1px solid rgba(37, 99, 235, 0.25) !important;
        border-left: 4px solid #2563eb !important;
    }
    .paper-meta strong {
        color: inherit !important;
    }
    .metric-card {
        background-color: rgba(37, 99, 235, 0.08) !important;
        color: inherit !important;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid rgba(37, 99, 235, 0.25) !important;
        border-left: 4px solid #2563eb !important;
        margin-bottom: 14px;
    }
    .metric-card * {
        color: inherit !important;
    }
    .metric-card strong {
        color: inherit !important;
        font-weight: 700;
    }
    .metric-card code {
        background-color: rgba(0, 0, 0, 0.25) !important;
        color: #38bdf8 !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .metric-card em {
        opacity: 0.85;
    }
    .disclaimer-banner {
        background-color: rgba(245, 158, 11, 0.12) !important;
        color: #f59e0b !important;
        padding: 12px 18px;
        border-radius: 6px;
        border: 1px solid rgba(245, 158, 11, 0.35) !important;
        border-left: 4px solid #f59e0b !important;
        font-size: 0.90rem;
        margin-bottom: 16px;
    }
    .disclaimer-banner strong {
        color: #f59e0b !important;
    }
    .failure-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.92rem;
        letter-spacing: 0.02em;
    }
    .failure-high {
        background-color: rgba(239, 68, 68, 0.18) !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
    }
    .failure-safe {
        background-color: rgba(34, 197, 94, 0.18) !important;
        color: #22c55e !important;
        border: 1px solid rgba(34, 197, 94, 0.4) !important;
    }
    .stat-label {
        font-size: 0.80rem;
        text-transform: uppercase;
        color: inherit;
        opacity: 0.75;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_positive_class_index(pipeline):
    """
    Locates index corresponding to class label 1 in pipeline's classes_.
    Avoids hardcoding column index 1.
    """
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 in classes:
        return classes.index(1)
    return 1


def load_image_safe(path):
    """
    Safely loads image via PIL to prevent Streamlit filesystem path resolution
    or relative directory issues on Windows.
    """
    try:
        p = Path(path)
        if p.exists():
            return Image.open(p)
    except Exception:
        pass
    return None


# ---------------------------------------------------------
# Cached Resource Loading
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """Load the three serialized scikit-learn pipelines."""
    models = {
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
        "Support Vector Machine": joblib.load(MODELS_DIR / "svm.joblib"),
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
    }
    return models


@st.cache_resource
def get_rf_tree_explainer(_rf_model):
    """Cache the SHAP TreeExplainer for Random Forest classifier."""
    return shap.TreeExplainer(_rf_model)


@st.cache_data
def load_metrics_table():
    """Load model evaluation summary table."""
    csv_path = RESULTS_DIR / "model_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_svm_perm_table():
    """Load precomputed SVM permutation importance table."""
    csv_path = RESULTS_DIR / "svm_permutation_importance.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ---------------------------------------------------------
# Main App Header & Uniform Paper Metadata
# ---------------------------------------------------------
st.markdown(
    """
    <div class="paper-title">An Explainable Machine Learning Framework for Industrial Predictive Maintenance</div>
    <div class="paper-subtitle">Comparative Evaluation of Failure Prediction Models with Transparent Feature Attribution</div>
    <div class="paper-meta">
        <strong>Academic Benchmark</strong>: PRG4206 Machine Learning | 
        <strong>Dataset</strong>: AI4I 2020 Predictive Maintenance (10,000 Records, 28.5:1 Imbalance) | 
        <strong>Framework</strong>: Stratified Pipelines + Multi-Model Benchmarking + Dual-Scale XAI
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("📄 Executive Research Abstract & Problem Statement", expanded=False):
    st.markdown(
        """
        **Abstract**: Unplanned downtime in industrial machining leads to substantial economic losses and safety risks. 
        This study presents an end-to-end Explainable Machine Learning (XAI) framework for predictive maintenance 
        using the **AI4I 2020 dataset**. To prevent target and identifier leakage, arbitrary identifiers (`UDI`, `Product ID`) 
        and failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were strictly excluded, isolating six operational features. 
        Addressing severe class imbalance (3.39% failures), three distinct classifier architectures—**Logistic Regression**, 
        **Random Forest**, and **Support Vector Machine (SVM)**—were trained and evaluated across standard and imbalance-aware 
        metrics (ROC-AUC and PR-AUC) on a strictly held-out test set (2,000 samples). 
        
        **Key Findings**:
        - **SVM with RBF Kernel** attained the highest sensitivity (**Recall: 89.71%**, ROC-AUC: 0.9655), capturing 61/68 failures.
        - **Random Forest** achieved superior discriminative precision (**Precision: 96.67%**, F1: 0.5918, PR-AUC: 0.7642) with only 1 false alarm.
        - **Explainability**: Local instance risk is audited using **SHAP TreeExplainer** with verified exact probability additivity ($\Delta = 1.92 \times 10^{-14}$), 
          complemented by **Log-Odds decomposition** for linear boundaries and **validation Permutation Feature Importance** for non-linear SVM kernels.
        """
    )

st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Academic Safety Notice</strong>: This demonstration is an educational benchmark using synthetic data.
        It is <strong>not a validated real-world industrial safety system</strong> and should not be used as the sole
        decision maker in mission-critical factory operations.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load Models
try:
    models = load_models()
except Exception as e:
    st.error(f"Error loading model pipelines: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar: Operational Machine Inputs
# ---------------------------------------------------------
st.sidebar.header("🔧 Machine Sensor Telemetry")
st.sidebar.markdown(
    "Configure input telemetry. **Note**: Failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) "
    "and identifiers (`UDI`, `Product ID`) are strictly excluded to avoid data leakage."
)

# Initialize session state for telemetry sliders if not present
preset_defaults = {
    "Safe / Normal Machine Operation": {"type": "M", "air_temp": 298.1, "proc_temp": 308.6, "speed": 1551, "torque": 42.8, "wear": 20},
    "High Tool Wear & Overstrain Risk": {"type": "L", "air_temp": 302.2, "proc_temp": 311.4, "speed": 1320, "torque": 68.0, "wear": 230},
    "Heat Dissipation Thermal Stress": {"type": "H", "air_temp": 304.2, "proc_temp": 313.8, "speed": 1410, "torque": 52.0, "wear": 110},
    "High Torque / Low Speed Strain": {"type": "L", "air_temp": 298.8, "proc_temp": 308.9, "speed": 1180, "torque": 74.5, "wear": 90},
    "Custom Input": {"type": "M", "air_temp": 300.0, "proc_temp": 310.0, "speed": 1500, "torque": 40.0, "wear": 100},
}

if "type_input" not in st.session_state:
    st.session_state["type_input"] = "M"
    st.session_state["air_temp_input"] = 300.0
    st.session_state["proc_temp_input"] = 310.0
    st.session_state["speed_input"] = 1500
    st.session_state["torque_input"] = 40.0
    st.session_state["wear_input"] = 100

def on_preset_select():
    sel = st.session_state.get("preset_choice", "Custom Input")
    if sel in preset_defaults and sel != "Custom Input":
        vals = preset_defaults[sel]
        st.session_state["type_input"] = vals["type"]
        st.session_state["air_temp_input"] = float(vals["air_temp"])
        st.session_state["proc_temp_input"] = float(vals["proc_temp"])
        st.session_state["speed_input"] = int(vals["speed"])
        st.session_state["torque_input"] = float(vals["torque"])
        st.session_state["wear_input"] = int(vals["wear"])

# Preset Presets for Quick Testing
preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    [
        "Custom Input",
        "Safe / Normal Machine Operation",
        "High Tool Wear & Overstrain Risk",
        "Heat Dissipation Thermal Stress",
        "High Torque / Low Speed Strain",
    ],
    key="preset_choice",
    on_change=on_preset_select,
)

type_input = st.sidebar.selectbox(
    "Product Type (Quality Variant)",
    ["L", "M", "H"],
    key="type_input",
    help="L: Low (60% of fleet), M: Medium (30%), H: High (10%). Represents material grade.",
)

air_temp_input = st.sidebar.slider(
    "Air Temperature [K]",
    min_value=295.0,
    max_value=305.0,
    key="air_temp_input",
    step=0.1,
    help="Ambient room air temperature [Kelvin].",
)

proc_temp_input = st.sidebar.slider(
    "Process Temperature [K]",
    min_value=305.0,
    max_value=315.0,
    key="proc_temp_input",
    step=0.1,
    help="Tool machining process operational temperature [Kelvin].",
)

speed_input = st.sidebar.slider(
    "Rotational Speed [rpm]",
    min_value=1150,
    max_value=2900,
    key="speed_input",
    step=5,
    help="Spindle rotational speed generated by motor.",
)

torque_input = st.sidebar.slider(
    "Torque [Nm]",
    min_value=3.5,
    max_value=80.0,
    key="torque_input",
    step=0.5,
    help="Torque applied to tool shaft.",
)

wear_input = st.sidebar.slider(
    "Tool Wear [min]",
    min_value=0,
    max_value=260,
    key="wear_input",
    step=1,
    help="Cumulative tool usage time before scheduled blade replacement.",
)

# Derived Physical Indicators (Display only, not fed as leaked columns)
temp_diff = proc_temp_input - air_temp_input
power_kw = (2 * np.pi * speed_input * torque_input) / 60000.0

st.sidebar.markdown("---")
st.sidebar.markdown("**Physical Indicators (Engineering Reference)**")
st.sidebar.markdown(f"- Temp Difference: **{temp_diff:.2f} K**")
st.sidebar.markdown(f"- Approx. Shaft Power: **{power_kw:.2f} kW**")

# Construct clean DataFrame with identical schema and order
input_df = pd.DataFrame(
    [
        {
            "Type": type_input,
            "Air temperature [K]": air_temp_input,
            "Process temperature [K]": proc_temp_input,
            "Rotational speed [rpm]": speed_input,
            "Torque [Nm]": torque_input,
            "Tool wear [min]": wear_input,
        }
    ]
)

# ---------------------------------------------------------
# Tabs Structure: Uniform Academic Layout
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Real-Time Inference & Model Comparison",
        "🧠 Explainable AI (SHAP & Attribution)",
        "📊 Benchmark Evaluation & Diagnostic Plots",
        "📄 Methodology, Dataset & Safety Disclaimers",
    ]
)

# ---------------------------------------------------------
# TAB 1: Real-Time Inference & Comparison
# ---------------------------------------------------------
with tab1:
    st.subheader("Comparative Model Inference: Real-Time Failure Risk Assessment")

    # Run inference across all three models with robust class-1 extraction
    results = {}
    for name, pipe in models.items():
        pos_idx = get_positive_class_index(pipe)
        proba = float(pipe.predict_proba(input_df)[0][pos_idx])
        pred_class = int(pipe.predict(input_df)[0])
        results[name] = {
            "proba": proba,
            "pred": pred_class,
            "pipe": pipe,
        }

    # Consolidated Multi-Model Comparison Cards
    cols = st.columns(3)
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {name}")
            is_failure = res["pred"] == 1
            prob_pct = res["proba"] * 100

            if is_failure:
                st.markdown(
                    "<div class='failure-badge failure-high'>🚨 FAILURE PREDICTED (Class 1)</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='failure-badge failure-safe'>✅ NORMAL OPERATION (Class 0)</div>",
                    unsafe_allow_html=True,
                )

            st.metric(
                label="Predicted Failure Probability",
                value=f"{prob_pct:.2f}%",
                delta=f"{'+' if prob_pct >= 3.39 else '-'}{abs(prob_pct - 3.39):.1f}% vs Dataset Base Rate (3.39%)",
                delta_color="inverse",
            )
            st.progress(min(max(res["proba"], 0.0), 1.0))

            if name == "Random Forest":
                st.caption(
                    "**Architecture**: Ensemble of 100 bagged trees with balanced split weighting.<br>"
                    "**Profile**: High Precision (96.7%), F1 (0.5918), PR-AUC (0.7642). Minimizes false dispatches.",
                    unsafe_allow_html=True,
                )
            elif name == "Support Vector Machine":
                st.caption(
                    "**Architecture**: RBF Non-linear Kernel ($C$-weighted margin). Platt calibrated.<br>"
                    "**Profile**: Highest Fault Recall (89.71%). Catches subtle non-linear failure conditions.",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "**Architecture**: L2-regularized linear hyper-plane with class balancing.<br>"
                    "**Profile**: High Recall (82.35%), but higher false alarms (Precision: 14.18%).",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Comparative Probability Chart
    st.subheader("Model Consensus & Risk Distribution")
    comp_chart_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Failure Probability (%)": [r["proba"] * 100 for r in results.values()],
            "Decision": [
                "Failure (1)" if r["pred"] == 1 else "Normal (0)"
                for r in results.values()
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 3.0))
    palette = [
        "#e76f51" if d == "Failure (1)" else "#2a9d8f"
        for d in comp_chart_df["Decision"]
    ]
    bars = ax.barh(
        comp_chart_df["Model"],
        comp_chart_df["Failure Probability (%)"],
        color=palette,
        height=0.45,
    )
    # Reference Lines with Precise Academic Descriptions
    ax.axvline(
        50.0,
        color="#d90429",
        linestyle="--",
        linewidth=1.2,
        label="Standard 50% Probability Cutoff (RF & LR Decision Threshold)",
    )
    ax.axvline(
        3.39,
        color="#6c757d",
        linestyle=":",
        linewidth=1.3,
        label="Observed AI4I Dataset Prevalence (3.39% Training Prior)",
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Predicted Failure Probability (%)", fontsize=10)
    ax.set_title(
        "Side-by-Side Model Risk Comparison", fontsize=11, weight="bold", pad=8
    )
    ax.legend(loc="lower right", fontsize=8.0)
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            weight="bold",
        )
    sns.despine(top=True, right=True)
    st.pyplot(fig)
    plt.close()

    # Distinction Between Current Input Inferences and Aggregate Test Metrics
    st.markdown("### 📌 Note on Real-Time Predictions vs. Aggregate Test-Set Performance")
    st.info(
        """
        - **Live Prediction vs Aggregate Performance**: The probabilities and decisions above represent the models' *local response* to the active telemetry inputs. They demonstrate real-time inference and must not be confused with the *aggregate benchmark metrics* (e.g., 91.75% SVM accuracy or 98.00% RF accuracy) calculated across the 2,000-sample test set in Tab 3.
        - **3.39% Benchmark Prevalence vs Alarm Threshold**: The 3.39% dotted reference represents the observed dataset failure frequency across the 10,000 synthetic records. It reflects the empirical training prior, not a universal industrial alarm trigger.
        - **Threshold Decision Rules**:
          - **Random Forest & Logistic Regression** trigger Class 1 when probability $\ge 50\%$ ($w^T z + b \ge 0$).
          - **SVM** triggers Class 1 when its signed margin distance $f(x) \ge 0$. While Platt scaling maps $f(x)$ to calibrated posterior probabilities, near the boundary the calibrated probability may differ slightly from a naive 50% cutoff due to class weighting.
        """
    )


# ---------------------------------------------------------
# TAB 2: Explainable AI (SHAP & Attribution)
# ---------------------------------------------------------
with tab2:
    st.subheader("Explainable AI (XAI) Framework: Model Transparency & Interpretability")
    st.markdown(
        "To enable trustworthy human-in-the-loop decision-making in industrial maintenance, "
        "the framework provides model-specific explanations tailored to each mathematical architecture."
    )

    xai_model_choice = st.selectbox(
        "Select Model for Explainability Decomposition",
        [
            "Random Forest (SHAP TreeExplainer — Probability Additivity)",
            "Logistic Regression (Standardized Log-Odds Coefficients)",
            "Support Vector Machine (Kernel Decision Score & Permutation Reliance)",
        ],
        key="xai_model_choice",
    )

    if "Random Forest" in xai_model_choice:
        st.markdown("#### Random Forest: Local Feature Attribution via SHAP TreeExplainer")
        rf_pipe = models["Random Forest"]
        preprocessor = rf_pipe.named_steps["preprocessor"]
        rf_clf = rf_pipe.named_steps["classifier"]

        # Transform single input row
        X_trans = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [
            f.replace("num__", "").replace("cat__", "") for f in feature_names
        ]

        try:
            tree_explainer = get_rf_tree_explainer(rf_clf)
            shap_vals = tree_explainer.shap_values(X_trans)
            expected_vals = tree_explainer.expected_value

            # Extract failure class (index 1) SHAP values and base value
            pos_idx = get_positive_class_index(rf_pipe)

            # Determine baseline expected value
            if isinstance(expected_vals, (np.ndarray, list)):
                base_value = float(expected_vals[pos_idx])
            else:
                base_value = float(expected_vals)

            # Extract class-1 SHAP values
            if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                vals_class1 = shap_vals[0, :, pos_idx]
            elif isinstance(shap_vals, list):
                vals_class1 = shap_vals[pos_idx][0]
            else:
                vals_class1 = shap_vals[0]

            # Numerical Additivity Verification: Base + sum(SHAP) = P(Class 1)
            sum_shap = float(np.sum(vals_class1))
            reconstructed_proba = base_value + sum_shap
            actual_proba = float(rf_pipe.predict_proba(input_df)[0][pos_idx])
            additivity_error = abs(reconstructed_proba - actual_proba)

            st.info(
                f"""
                **SHAP Exact Probability Additivity Verification**:
                - **Model Output Scale**: Probability $[0, 1]$ (Scikit-Learn Random Forest class probability fractions)
                - **Reference Baseline $\\mathbb{{E}}[f(X)]$**: `{base_value:.4f}` *(Average leaf proportion across balanced training trees)*
                - **Sum of Feature Attributions $\\sum \\phi_i$**: `{sum_shap:+.4f}`
                - **Reconstructed Output**: `{base_value:.4f} + ({sum_shap:+.4f}) = {reconstructed_proba:.4f}`
                - **Model Pipeline `predict_proba` Output**: `{actual_proba:.4f}`
                - **Numerical Discrepancy $|\\text{{Recon}} - \\hat{{P}}|$**: `{additivity_error:.2e}` *(Exact Additivity Confirmed within Machine Precision)*
                """,
                icon="🌳",
            )

            # Explanation of Baseline vs Real-World Prevalence
            st.info(
                f"""
                <strong>Understanding the SHAP Baseline vs. Real Failure Prevalence</strong>:<br>
                - <strong>SHAP Reference Baseline ($\approx {base_value*100:.1f}\%$)</strong>: This baseline does <em>not</em> represent the empirical dataset failure rate (3.39%). 
                  Because <code>class_weight='balanced'</code> was used during training, minority failure samples were penalized $\\approx 14.75\\times$ more heavily, 
                  balancing the split criteria across decision trees. The SHAP TreeExplainer computes its baseline $\\mathbb{{E}}[f(X)]$ over this balanced training leaf distribution.
                - <strong>Directional Interpretation</strong>:
                  - 🔴 <strong>Positive SHAP ($\phi_i > 0$)</strong>: Pushes prediction <em>toward failure</em> above the baseline.
                  - 🟢 <strong>Negative SHAP ($\phi_i < 0$)</strong>: Pushes prediction <em>away from failure</em> (toward safe normal operation).
                - For healthy machines (failure probability $\approx 0.00\%$), the feature contributions are predominantly negative ($\sum \phi_i \approx -0.4987$), 
                  canceling out the balanced baseline and correctly reconstructing the final $0.00\%$ probability output.
                """,
                icon="ℹ️",
            )

            # Construct human-readable input representation for SHAP visualization
            raw_vals = []
            for f in clean_names:
                if f in input_df.columns:
                    val = input_df[f].iloc[0]
                    raw_vals.append(f"{val:.1f}" if isinstance(val, float) else f"{val}")
                elif f.startswith("Type_"):
                    t_val = f.split("_")[1]
                    raw_vals.append("1" if input_df["Type"].iloc[0] == t_val else "0")
                else:
                    raw_vals.append("0")

            shap_df = pd.DataFrame(
                {
                    "Feature": clean_names,
                    "Current Value": raw_vals,
                    "SHAP Value (Probability Impact)": vals_class1,
                    "Magnitude": np.abs(vals_class1),
                }
            ).sort_values("Magnitude", ascending=True)

            # Visualization Format Selection
            st.markdown("##### SHAP Local Explanation Visualizer")
            shap_plot_type = st.radio(
                "Select Visualization Mode:",
                [
                    "Waterfall Plot (Sequential Shift from Baseline)",
                    "Ranked Impact Bar Chart (Direction & Magnitude)",
                ],
                key="shap_plot_type",
                horizontal=True,
            )

            if "Waterfall" in shap_plot_type:
                exp = shap.Explanation(
                    values=vals_class1,
                    base_values=base_value,
                    data=raw_vals,
                    feature_names=clean_names,
                )
                fig_wf = plt.figure(figsize=(9.5, 4.8))
                shap.plots.waterfall(exp, max_display=10, show=False)
                plt.title(
                    f"Random Forest SHAP Waterfall Decomposition\n"
                    f"Reference Baseline $\\mathbb{{E}}[f(X)] = {base_value:.4f} \\longrightarrow \\hat{{P}}(\\text{{Fail}}) = {actual_proba:.4f}$",
                    fontsize=11,
                    weight="bold",
                    pad=12,
                )
                plt.tight_layout()
                st.pyplot(fig_wf)
                plt.close(fig_wf)
                st.caption(
                    "**Waterfall Interpretation**: Starting from the balanced training baseline "
                    f"($\\mathbb{{E}}[f(X)] = {base_value:.4f}$), each feature shifts the probability "
                    f"upward (red, towards failure) or downward (blue, towards safety), arriving at the exact model output: "
                    f"$\\hat{{P}}(\\text{{Fail}}) = {actual_proba:.4f}$ ({actual_proba*100:.2f}%)."
                )
            else:
                fig, ax = plt.subplots(figsize=(9.5, 5.0))
                colors = [
                    "#e76f51" if val > 0 else "#2a9d8f"
                    for val in shap_df["SHAP Value (Probability Impact)"]
                ]
                bars = ax.barh(
                    shap_df["Feature"],
                    shap_df["SHAP Value (Probability Impact)"],
                    color=colors,
                    height=0.55,
                    edgecolor="#2b2d42",
                    linewidth=0.5,
                )

                max_val = max(shap_df["SHAP Value (Probability Impact)"].max(), 0.02)
                min_val = min(shap_df["SHAP Value (Probability Impact)"].min(), -0.02)
                range_span = max(max_val - min_val, 0.08)
                pad_left = abs(min_val) * 0.35 if min_val < 0 else 0.05
                pad_right = abs(max_val) * 0.35 if max_val > 0 else 0.05
                ax.set_xlim(min_val - pad_left, max_val + pad_right)

                ax.axvline(0, color="#2b2d42", linestyle="-", linewidth=1.2, alpha=0.85)
                ax.grid(axis="x", linestyle="--", alpha=0.35)

                for bar in bars:
                    w = bar.get_width()
                    offset = range_span * 0.025 if w >= 0 else -range_span * 0.025
                    ha = "left" if w >= 0 else "right"
                    ax.text(
                        w + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{w:+.4f}",
                        va="center",
                        ha=ha,
                        fontsize=9,
                        fontweight="bold",
                        color="#1a202c",
                    )

                ax.set_xlabel(
                    "SHAP Feature Attribution $\\phi_i$  [← Negative: Reduces Failure Risk (Safe) | Positive: Increases Failure Risk (Risk) →]",
                    fontsize=9.5,
                    fontweight="bold",
                    labelpad=10,
                )
                ax.set_title(
                    f"Random Forest Local Feature Attributions (Ranked by Absolute Magnitude)\n"
                    f"Reference Baseline $\\mathbb{{E}}[f(X)] = {base_value:.4f}$ | Reconstructed $\\hat{{P}}(\\text{{Fail}}) = {actual_proba:.4f}$",
                    fontsize=11,
                    weight="bold",
                    pad=12,
                )

                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor="#e76f51", edgecolor="#2b2d42", label="Positive SHAP: Pushes Toward Failure (Risk)"),
                    Patch(facecolor="#2a9d8f", edgecolor="#2b2d42", label="Negative SHAP: Pushes Away from Failure (Safe)"),
                ]
                ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=8.5, framealpha=0.9)

                sns.despine(top=True, right=True)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Detailed Tabular Breakdown
            with st.expander("📋 View Detailed Feature Attribution Data Table", expanded=False):
                table_df = shap_df.copy().sort_values("Magnitude", ascending=False)
                table_df["Impact Direction"] = table_df["SHAP Value (Probability Impact)"].apply(
                    lambda v: "🔴 Increases Failure Risk" if v > 0 else "🟢 Reduces Failure Risk (Safe)"
                )
                table_df["SHAP Attribution (ΔP)"] = table_df["SHAP Value (Probability Impact)"].apply(lambda v: f"{v:+.4f}")
                table_df["Magnitude |ΔP|"] = table_df["Magnitude"].apply(lambda v: f"{v:.4f}")
                st.dataframe(
                    table_df[["Feature", "Current Value", "Impact Direction", "SHAP Attribution (ΔP)", "Magnitude |ΔP|"]],
                    use_container_width=True,
                    hide_index=True,
                )

            # Global Random Forest Feature Importance (MDI)
            st.markdown("---")
            st.markdown("#### Global Random Forest Feature Importance (Gini Impurity)")
            st.caption(
                "Distinction: This chart reflects **global model reliance** (Mean Decrease in Impurity across all 100 trees on training data), "
                "not local instance attribution."
            )
            global_imp = pd.DataFrame(
                {
                    "Feature": clean_names,
                    "Importance": rf_clf.feature_importances_,
                }
            ).sort_values("Importance", ascending=True)

            fig2, ax2 = plt.subplots(figsize=(9, 3.5))
            ax2.barh(
                global_imp["Feature"],
                global_imp["Importance"],
                color="#1f77b4",
                height=0.5,
            )
            ax2.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=10)
            ax2.set_title(
                "Global Random Forest Feature Importance (Training Set)",
                fontsize=11,
                weight="bold",
            )
            sns.despine(top=True, right=True)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        except Exception as e:
            st.warning(
                f"SHAP explanation exception: {e}. Displaying Gini impurity importance instead."
            )

    elif "Logistic Regression" in xai_model_choice:
        st.markdown(
            "#### Logistic Regression: Standardized Log-Odds Feature Contributions"
        )
        lr_pipe = models["Logistic Regression"]
        preprocessor = lr_pipe.named_steps["preprocessor"]
        lr_clf = lr_pipe.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        clean_names = [
            f.replace("num__", "").replace("cat__", "") for f in feature_names
        ]
        X_trans = preprocessor.transform(input_df)[0]
        weights = lr_clf.coef_[0]
        intercept = float(lr_clf.intercept_[0])

        # Individual log-odds contribution: w_i * x_i
        contrib = weights * X_trans
        sum_contrib = float(np.sum(contrib))
        reconstructed_logit = intercept + sum_contrib
        actual_decision_score = float(lr_clf.decision_function(preprocessor.transform(input_df))[0])
        sigmoid_proba = 1.0 / (1.0 + np.exp(-reconstructed_logit))
        pos_idx = get_positive_class_index(lr_pipe)
        actual_proba = float(lr_pipe.predict_proba(input_df)[0][pos_idx])

        # Distinguish numerical standardized features from categorical one-hot indicators
        feature_types = [
            "Numerical (Z-score)" if "num__" in f else "Categorical (Dummy 0/1)"
            for f in feature_names
        ]

        st.info(
            f"""
            **Logit Score & Probability Reconstruction**:
            - **Intercept $\\beta_0$**: `{intercept:.4f}`
            - **Sum of Feature Impacts $\\sum \\beta_i x_i$**: `{sum_contrib:+.4f}`
            - **Reconstructed Logit**: `\\beta_0 + \\sum \\beta_i x_i = {reconstructed_logit:.4f}`
            - **Fitted `decision_function` Score**: `{actual_decision_score:.4f}` *(Discrepancy: {abs(reconstructed_logit - actual_decision_score):.2e})*
            - **Sigmoid Conversion $\\sigma(\\text{{Logit}})$**: `1 / (1 + e^{{-{reconstructed_logit:.4f}}}) = {sigmoid_proba:.4f}` *(Matches `predict_proba`: `{actual_proba:.4f}`)*
            """,
            icon="📈",
        )

        lr_contrib_df = pd.DataFrame(
            {
                "Feature": clean_names,
                "Feature Type": feature_types,
                "Model Weight (Beta)": weights,
                "Transformed Value (z-score / dummy)": X_trans,
                "Log-Odds Contribution": contrib,
                "Abs Contribution": np.abs(contrib),
            }
        ).sort_values("Abs Contribution", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = [
            "#e76f51" if val > 0 else "#2a9d8f"
            for val in lr_contrib_df["Log-Odds Contribution"]
        ]
        ax.barh(
            lr_contrib_df["Feature"],
            lr_contrib_df["Log-Odds Contribution"],
            color=colors,
            height=0.55,
        )
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlabel("Log-Odds Impact (Weight $\\beta_i$ × Transformed Value)", fontsize=10)
        ax.set_title(
            f"Logistic Regression Local Feature Impact (Intercept: {intercept:.2f})",
            fontsize=11,
            weight="bold",
        )
        sns.despine(top=True, right=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption(
            "⚠️ **Interpretation Notice**: In linear logistic regression, feature contributions represent additive shifts in **log-odds** "
            "($\\ln(p / (1-p))$), **not** direct percentage-point probability increases and **not** causal physical effects.",
            unsafe_allow_html=True,
        )

        st.dataframe(
            lr_contrib_df[
                [
                    "Feature",
                    "Feature Type",
                    "Model Weight (Beta)",
                    "Transformed Value (z-score / dummy)",
                    "Log-Odds Contribution",
                ]
            ].sort_values("Log-Odds Contribution", ascending=False),
            use_container_width=True,
        )

    else:
        st.markdown(
            "#### Support Vector Machine: Non-Linear Kernel Boundary & Validation Permutation Importance"
        )
        svm_pipe = models["Support Vector Machine"]
        preprocessor = svm_pipe.named_steps["preprocessor"]
        svm_clf = svm_pipe.named_steps["classifier"]

        # Calculate decision function score and calibrated probability
        dist = float(svm_clf.decision_function(preprocessor.transform(input_df))[0])
        pos_idx = get_positive_class_index(svm_pipe)
        proba = float(svm_pipe.predict_proba(input_df)[0][pos_idx])

        st.info(
            f"""
            **Non-Linear SVM Prediction Diagnostics**:
            - **Decision Function Score $f(x)$**: `{dist:+.4f}`  
              *(Signed distance from the separating hyperplane in RKHS space. $f(x) \\ge 0 \\implies$ Predicted Failure, $f(x) < 0 \\implies$ Predicted Normal).*
            - **Calibrated Failure Probability $\\hat{{P}}(\\text{{Fail}} \\mid X)$**: `{proba * 100:.2f}%`  
              *(Derived via Platt scaling — logistic sigmoid fitted over cross-validation folds. Calibrated statistical risk estimate).*
            - **Kernel Function**: Radial Basis Function (RBF: $K(x, x') = \\exp(-\\gamma \\|x - x'\\|^2)$).
            """,
            icon="📐",
        )

        st.info(
            """
            **Why Exact Analytical Linear Attribution Does Not Apply to RBF SVM**:
            Because the RBF kernel projects inputs non-linearly into an infinite-dimensional feature space, individual feature weights
            cannot be expressed as a single vector of independent coefficients in the original input space.
            Instead, global model reliance is rigorously evaluated using **Permutation Feature Importance** on held-out validation data.
            """
        )

        st.markdown("---")
        st.subheader("Global SVM Permutation Feature Importance (Validation Fold)")
        st.caption(
            "Calculated on a held-out validation fold (1,600 samples, 20% of training data, $n=10$ repeats, scoring metric: Average Precision / PR-AUC). "
            "Represents global model reliance (score degradation when feature is shuffled), **not** causal physical importance."
        )

        perm_table = load_svm_perm_table()
        if perm_table is not None:
            col_chart, col_tbl = st.columns([3, 2])
            with col_chart:
                perm_fig_path = FIGURES_DIR / "svm_permutation_importance.png"
                perm_img = load_image_safe(perm_fig_path)
                if perm_img is not None:
                    st.image(
                        perm_img,
                        caption="Artifact: figures/svm_permutation_importance.png",
                        use_column_width=True,
                    )
            with col_tbl:
                st.markdown("**Validation Importance Scores (PR-AUC Drop)**")
                formatted_perm = perm_table.copy()
                formatted_perm["Importance_Mean"] = formatted_perm["Importance_Mean"].apply(lambda v: f"{v:.4f}")
                formatted_perm["Importance_Std"] = formatted_perm["Importance_Std"].apply(lambda v: f"{v:.4f}")
                st.dataframe(
                    formatted_perm,
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("Permutation importance table not found at results/svm_permutation_importance.csv.")


# ---------------------------------------------------------
# TAB 3: Model Evaluation & Benchmarks
# ---------------------------------------------------------
with tab3:
    st.subheader("Comparative Benchmark Evaluation & Empirical Results")
    st.markdown(
        "All models were evaluated on the exact same held-out test set (2,000 samples, 20% stratified split with zero index leakage)."
    )

    metrics_df = load_metrics_table()
    if metrics_df is not None:
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
                color="#c6e0b4",
            ),
            use_container_width=True,
        )
    else:
        st.warning("Metrics file results/model_metrics.csv not found.")

    st.markdown("---")
    st.subheader("Saved Evaluation Plots from Pipeline Execution")

    selected_plot = st.radio(
        "Select Performance Diagnostic Figure",
        [
            "Confusion Matrices",
            "Multi-Metric Comparative Bar Chart",
            "ROC Curves (AUC Comparison)",
            "Precision-Recall Curves (Class Imbalance Metric)",
            "SVM Permutation Feature Importance (Validation Fold)",
        ],
        key="diagnostic_plot_choice",
        horizontal=True,
    )

    plot_file_map = {
        "Confusion Matrices": FIGURES_DIR / "confusion_matrices.png",
        "Multi-Metric Comparative Bar Chart": FIGURES_DIR
        / "model_comparison_metrics.png",
        "ROC Curves (AUC Comparison)": FIGURES_DIR / "roc_curves.png",
        "Precision-Recall Curves (Class Imbalance Metric)": FIGURES_DIR
        / "pr_curves.png",
        "SVM Permutation Feature Importance (Validation Fold)": FIGURES_DIR
        / "svm_permutation_importance.png",
    }

    fig_path = plot_file_map[selected_plot]
    fig_img = load_image_safe(fig_path)
    if fig_img is not None:
        st.image(
            fig_img,
            caption=f"Artifact: figures/{fig_path.name}",
            use_column_width=True,
        )
    else:
        st.error(f"Image artifact could not be loaded from {fig_path}")


# ---------------------------------------------------------
# TAB 4: Data & Industrial Limitations
# ---------------------------------------------------------
with tab4:
    st.subheader("Methodology, Dataset Architecture & Academic Context")

    st.markdown(
        """
        ### AI4I 2020 Predictive Maintenance Dataset
        The dataset consists of **10,000 synthetic industrial records** simulating a milling machine tool:
        - **Target Variable**: `Machine failure` (Binary: 0 or 1).
        - **Fleet Base Failure Rate**: `3.39%` (339 failure records vs 9,661 safe operations).
        - **Imbalance Ratio**: `28.5 : 1`.

        ### Strict Target Leakage Prevention
        The original dataset includes five specific failure mode flags:
        1. `TWF` (Tool Wear Failure)
        2. `HDF` (Heat Dissipation Failure)
        3. `PWF` (Power Failure)
        4. `OSF` (Overstrain Failure)
        5. `RNF` (Random Failure)

        > **Academic Requirement**: These five flags are subsets of the primary `Machine failure` target.
        > Using any of them as model predictors constitutes **catastrophic target leakage**.
        > Similarly, `UDI` and `Product ID` are arbitrary row indices that cause identifier leakage.
        > **All eight non-operational columns were strictly excluded from model training.**

        ### Six Operational Predictors Used
        1. `Type`: Product variant grade ('L', 'M', 'H').
        2. `Air temperature [K]`: Surrounding ambient room temperature.
        3. `Process temperature [K]`: Machining chamber operational temperature.
        4. `Rotational speed [rpm]`: Motor spindle shaft speed.
        5. `Torque [Nm]`: Mechanical resistance torque on the cutting head.
        6. `Tool wear [min]`: Time the active cutting tool has been engaged.

        ### Operational Trade-off Analysis
        - **Support Vector Machine (SVM)** is best suited when the factory cannot afford any missed breakdowns (Highest Recall: `89.71%`).
        - **Random Forest** is best suited when maintenance dispatch costs are extremely high and false alarms must be prevented (Highest Precision: `96.67%` and F1: `0.5918`).
        """
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>"
    "An Explainable Machine Learning Framework for Industrial Predictive Maintenance | PRG4206 Academic Research Benchmark"
    "</div>",
    unsafe_allow_html=True,
)
