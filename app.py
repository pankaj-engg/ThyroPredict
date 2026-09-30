"""
app.py
======
ThyroPredict — AI-Based Thyroid Abnormality Prediction System
Streamlit Web Application

This application loads a pre-trained Scikit-Learn pipeline (preprocessor + model)
to predict thyroid abnormalities from clinical attributes and laboratory measurements.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ThyroPredict — AI Thyroid Abnormality Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'thyroid_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .result-card-normal {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .result-card-abnormal {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .metric-badge {
        display: inline-block;
        background-color: #edf2f7;
        color: #2d3748;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model & Metadata Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_pipeline():
    """Loads the serialized Scikit-learn Pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None


@st.cache_data
def load_model_metadata():
    """Loads model metadata containing performance metrics and column specs."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


pipeline = load_trained_pipeline()
metadata = load_model_metadata()

# -----------------------------------------------------------------------------
# Sidebar: Project Info, Metrics & Disclaimer
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=400&q=80",
        use_container_width=True
    )
    st.title("About ThyroPredict")
    st.markdown("""
    **ThyroPredict** is an educational Machine Learning system designed to predict 
    the presence of thyroid abnormalities based on standard clinical and biochemical markers.
    """)

    st.subheader("Model Information")
    if metadata:
        st.write(f"**Selected Algorithm:** {metadata.get('best_model', 'Random Forest')}")
        metrics = metadata.get('metrics', {})
        st.markdown(f"""
        - **Accuracy:** `{metrics.get('Accuracy', 0.0)*100:.2f}%`
        - **Precision:** `{metrics.get('Precision', 0.0)*100:.2f}%`
        - **Recall:** `{metrics.get('Recall', 0.0)*100:.2f}%`
        - **F1-Score:** `{metrics.get('F1_Score', 0.0)*100:.2f}%`
        - **ROC-AUC:** `{metrics.get('ROC_AUC', 0.0):.4f}`
        """)
    else:
        st.info("Train the model via `python train_model.py` to view metrics.")

    st.subheader("Key Clinical Markers")
    st.markdown("""
    - **TSH** (Thyroid Stimulating Hormone): *0.4 – 4.0 μIU/mL*
    - **T3** (Triiodothyronine): *1.2 – 3.1 nmol/L*
    - **TT4** (Total Thyroxine): *60 – 150 nmol/L*
    - **T4U** (Thyroxine Uptake): *0.8 – 1.3*
    - **FTI** (Free Thyroxine Index): *70 – 140*
    """)

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🩺 ThyroPredict</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Machine-learning-based prediction of thyroid abnormalities '
    'using clinical data and laboratory thyroid hormone panels.</div>',
    unsafe_allow_html=True
)

# Check model availability
if pipeline is None:
    st.error(
        "⚠️ **Trained model pipeline not found!**\n\n"
        "Please generate the model file first by running:\n\n"
        "```bash\npython train_model.py\n```\n"
        "Once complete, refresh this page."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Preset Sample Buttons for Quick Demonstration / Viva Testing
# -----------------------------------------------------------------------------
st.markdown("### 📋 Quick Preset Test Samples")
st.caption("Click a preset below to populate the form with verified test cases for instant demonstration:")

col_preset1, col_preset2, col_preset_reset = st.columns([1, 1, 1])

if 'preset' not in st.session_state:
    st.session_state['preset'] = 'normal'

if col_preset1.button("🟢 Load Normal Patient Case", use_container_width=True):
    st.session_state['preset'] = 'normal'

if col_preset2.button("🔴 Load Abnormal (Hypothyroid) Case", use_container_width=True):
    st.session_state['preset'] = 'abnormal'

if col_preset_reset.button("🔄 Reset to Default Values", use_container_width=True):
    st.session_state['preset'] = 'default'


# Set initial form values based on preset
if st.session_state['preset'] == 'abnormal':
    # High TSH, low TT4, low FTI (classic primary/compensated hypothyroid profile)
    default_age = 58
    default_sex = "Female"
    default_tsh = 24.5
    default_t3 = 1.1
    default_tt4 = 42.0
    default_t4u = 0.95
    default_fti = 44.0
    default_on_thyroxine = "No"
    default_query_hypo = "Yes"
elif st.session_state['preset'] == 'normal':
    # Normal physiological values
    default_age = 42
    default_sex = "Female"
    default_tsh = 1.6
    default_t3 = 2.1
    default_tt4 = 105.0
    default_t4u = 0.98
    default_fti = 107.0
    default_on_thyroxine = "No"
    default_query_hypo = "No"
else:
    default_age = 45
    default_sex = "Female"
    default_tsh = 2.0
    default_t3 = 2.0
    default_tt4 = 100.0
    default_t4u = 1.00
    default_fti = 100.0
    default_on_thyroxine = "No"
    default_query_hypo = "No"


# -----------------------------------------------------------------------------
# Patient Clinical Input Form
# -----------------------------------------------------------------------------
st.markdown("### 📝 Enter Patient Clinical & Laboratory Parameters")

with st.form("prediction_form"):
    tab_labs, tab_history = st.tabs(["🧪 Laboratory Thyroid Hormone Panel", "📋 Patient History & Symptoms"])

    with tab_labs:
        st.markdown("##### 1. Demographics & Laboratory Blood Tests")
        st.caption("Enter patient demographic information and standard thyroid hormone panel results.")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=110,
                value=int(default_age),
                help="Patient age between 1 and 110 years."
            )
            sex = st.selectbox(
                "Sex",
                options=["Female", "Male"],
                index=0 if default_sex == "Female" else 1,
                help="Biological sex of the patient."
            )

        with col2:
            tsh = st.number_input(
                "TSH — Thyroid Stimulating Hormone (μIU/mL)",
                min_value=0.0,
                max_value=500.0,
                value=float(default_tsh),
                step=0.1,
                format="%.2f",
                help="Normal clinical reference range: 0.4 – 4.0 μIU/mL. Elevated levels (>4.5) often suggest hypothyroidism."
            )
            t3 = st.number_input(
                "T3 — Triiodothyronine (nmol/L)",
                min_value=0.0,
                max_value=15.0,
                value=float(default_t3),
                step=0.1,
                format="%.2f",
                help="Normal clinical reference range: 1.2 – 3.1 nmol/L."
            )

        with col3:
            tt4 = st.number_input(
                "TT4 — Total Thyroxine (nmol/L)",
                min_value=0.0,
                max_value=450.0,
                value=float(default_tt4),
                step=1.0,
                format="%.1f",
                help="Normal clinical reference range: 60 – 150 nmol/L."
            )
            t4u = st.number_input(
                "T4U — Thyroxine Uptake Ratio",
                min_value=0.1,
                max_value=3.0,
                value=float(default_t4u),
                step=0.01,
                format="%.2f",
                help="Normal clinical reference range: 0.8 – 1.3."
            )
            fti = st.number_input(
                "FTI — Free Thyroxine Index",
                min_value=0.0,
                max_value=400.0,
                value=float(default_fti),
                step=1.0,
                format="%.1f",
                help="Normal clinical reference range: 70 – 140."
            )

    with tab_history:
        st.markdown("##### 2. Clinical History & Symptom Observations")
        st.caption("Specify existing medical conditions, thyroid medications, or treatments.")

        hcol1, hcol2, hcol3 = st.columns(3)

        with hcol1:
            on_thyroxine = st.selectbox(
                "Currently on Thyroxine medication?",
                options=["No", "Yes"],
                index=1 if default_on_thyroxine == "Yes" else 0
            )
            query_on_thyroxine = st.selectbox(
                "Query on Thyroxine?",
                options=["No", "Yes"],
                index=0
            )
            on_antithyroid_medication = st.selectbox(
                "On Antithyroid Medication?",
                options=["No", "Yes"],
                index=0
            )
            sick = st.selectbox(
                "General Sick Condition?",
                options=["No", "Yes"],
                index=0
            )
            pregnant = st.selectbox(
                "Pregnant?",
                options=["No", "Yes"],
                index=0
            )

        with hcol2:
            thyroid_surgery = st.selectbox(
                "History of Thyroid Surgery?",
                options=["No", "Yes"],
                index=0
            )
            i131_treatment = st.selectbox(
                "I131 Radiotherapy Treatment?",
                options=["No", "Yes"],
                index=0
            )
            query_hypothyroid = st.selectbox(
                "Query Hypothyroid?",
                options=["No", "Yes"],
                index=1 if default_query_hypo == "Yes" else 0
            )
            query_hyperthyroid = st.selectbox(
                "Query Hyperthyroid?",
                options=["No", "Yes"],
                index=0
            )
            lithium = st.selectbox(
                "Taking Lithium?",
                options=["No", "Yes"],
                index=0
            )

        with hcol3:
            goitre = st.selectbox(
                "Goitre Present?",
                options=["No", "Yes"],
                index=0
            )
            tumor = st.selectbox(
                "Thyroid Tumor Present?",
                options=["No", "Yes"],
                index=0
            )
            hypopituitary = st.selectbox(
                "Hypopituitary History?",
                options=["No", "Yes"],
                index=0
            )
            psych = st.selectbox(
                "Psychological Symptoms?",
                options=["No", "Yes"],
                index=0
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.form_submit_button("🔍 PREDICT THYROID STATUS", use_container_width=True)


# -----------------------------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------------------------
if predict_clicked:
    # 1. Input Validation
    if age <= 0:
        st.error("Invalid Input: Patient Age must be greater than 0.")
        st.stop()

    if tsh < 0 or t3 < 0 or tt4 < 0 or t4u < 0 or fti < 0:
        st.error("Invalid Input: Laboratory hormone measurements cannot be negative.")
        st.stop()

    # 2. Build Input DataFrame matching the exact schema used during training
    # Mapping helper
    def yn_to_tf(val):
        return 't' if val == 'Yes' else 'f'

    input_data = {
        'age': [float(age)],
        'sex': ['F' if sex == 'Female' else 'M'],
        'on_thyroxine': [yn_to_tf(on_thyroxine)],
        'query_on_thyroxine': [yn_to_tf(query_on_thyroxine)],
        'on_antithyroid_medication': [yn_to_tf(on_antithyroid_medication)],
        'sick': [yn_to_tf(sick)],
        'pregnant': [yn_to_tf(pregnant)],
        'thyroid_surgery': [yn_to_tf(thyroid_surgery)],
        'I131_treatment': [yn_to_tf(i131_treatment)],
        'query_hypothyroid': [yn_to_tf(query_hypothyroid)],
        'query_hyperthyroid': [yn_to_tf(query_hyperthyroid)],
        'lithium': [yn_to_tf(lithium)],
        'goitre': [yn_to_tf(goitre)],
        'tumor': [yn_to_tf(tumor)],
        'hypopituitary': [yn_to_tf(hypopituitary)],
        'psych': [yn_to_tf(psych)],
        'TSH': [float(tsh)],
        'T3': [float(t3)],
        'TT4': [float(tt4)],
        'T4U': [float(t4u)],
        'FTI': [float(fti)]
    }

    input_df = pd.DataFrame(input_data)

    try:
        # Run inference through pipeline
        prediction = pipeline.predict(input_df)[0]
        prediction_label = "THYROID ABNORMALITY" if prediction == 1 else "NORMAL"

        # Model probability
        probability_text = "N/A"
        prob_abnormal = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            prob_normal = probabilities[0] * 100
            prob_abnormal = probabilities[1] * 100
            selected_prob = prob_abnormal if prediction == 1 else prob_normal
            probability_text = f"{selected_prob:.1f}%"

        # -------------------------------------------------------------------------
        # Display Prediction Result Card
        # -------------------------------------------------------------------------
        st.markdown("### 📊 Prediction Result")

        if prediction == 0:
            st.markdown(f"""
            <div class="result-card-normal">
                <h2 style="color: #15803d; margin-top:0;">✅ Predicted Class: NORMAL</h2>
                <p style="font-size: 1.1rem; color: #166534;">
                    <strong>Model probability:</strong> {probability_text}
                </p>
                <p style="color: #374151; margin-bottom: 0;">
                    The model classified the provided input as <strong>Normal</strong> 
                    based on patterns learned from the training dataset.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-abnormal">
                <h2 style="color: #b91c1c; margin-top:0;">⚠️ Predicted Class: THYROID ABNORMALITY</h2>
                <p style="font-size: 1.1rem; color: #991b1b;">
                    <strong>Model probability:</strong> {probability_text}
                </p>
                <p style="color: #374151; margin-bottom: 0;">
                    The model classified the provided input as <strong>Thyroid Abnormality</strong> 
                    based on patterns learned from the training dataset.
                </p>
            </div>
            """, unsafe_allow_html=True)

        

        # Breakdown of probabilities
        if prob_abnormal is not None:
            st.markdown("#### Model Class Probability Distribution")
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Class: Normal", f"{prob_normal:.1f}%")
            pcol2.metric("Class: Thyroid Abnormality", f"{prob_abnormal:.1f}%")
            st.progress(float(prob_abnormal / 100.0))

        # Physiological summary observation
        st.markdown("#### Clinical Feature Observation")
        obs_text = []
        if tsh > 4.5:
            obs_text.append(f"• **TSH is elevated** ({tsh:.2f} μIU/mL, normal range: 0.4–4.0 μIU/mL), which is the primary hallmark of hypothyroid conditions.")
        elif tsh < 0.4:
            obs_text.append(f"• **TSH is suppressed** ({tsh:.2f} μIU/mL), which may occur in hyperthyroid or secondary conditions.")
        else:
            obs_text.append(f"• **TSH is within the normal reference range** ({tsh:.2f} μIU/mL).")

        if tt4 < 60.0:
            obs_text.append(f"• **Total T4 is below normal** ({tt4:.1f} nmol/L, normal range: 60–150 nmol/L).")
        elif tt4 > 150.0:
            obs_text.append(f"• **Total T4 is elevated** ({tt4:.1f} nmol/L).")

        if fti < 70.0:
            obs_text.append(f"• **Free Thyroxine Index (FTI) is low** ({fti:.1f}, normal range: 70–140).")

        for obs in obs_text:
            st.write(obs)

    except Exception as err:
        st.error(f"Error during prediction pipeline execution: {err}")
        st.info("Ensure all input variables correspond to the expected schema.")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.85rem;'>"
    "ThyroPredict — AI-Based Thyroid Abnormality Prediction System &bull; "
    "Educational Machine Learning Project &bull; Built with Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)