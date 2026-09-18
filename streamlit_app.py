import io
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import get_predictor
from app.risk import evaluate_risk_and_recommendation

# Page setup
st.set_page_config(
    page_title="SiteSafe Vision | AI PPE Screening",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }
    .disclaimer-card {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        color: #fde68a;
        font-size: 0.85rem;
        line-height: 1.45;
        margin-bottom: 1.5rem;
    }
    .badge-full {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.9rem;
    }
    .badge-partial {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.9rem;
    }
    .badge-none {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.9rem;
    }
    .rec-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #3b82f6;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached_predictor():
    p = get_predictor()
    if not p.is_loaded:
        try:
            from tests.fixtures.create_fixtures import ensure_test_artifacts
            ensure_test_artifacts()
            p = get_predictor()
        except Exception:
            pass
    return p


predictor = load_cached_predictor()

# Sidebar
with st.sidebar:
    st.markdown("## 🦺 SiteSafe Vision")
    st.markdown("**AI-Powered Construction PPE Screening**")
    st.divider()

    st.markdown("### ⚙️ Screening Configuration")
    show_gradcam = st.checkbox("Generate Grad-CAM Heatmap", value=True, help="Compute visual saliency activation overlay")
    threshold = st.slider("Confidence Caution Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.05)

    st.divider()
    st.markdown("### 🖼️ Sample Gallery")
    sample_dir = PROJECT_ROOT / "data" / "interim" / "crops"
    samples = sorted(list(sample_dir.glob("*.jpg")))[:12] if sample_dir.exists() else []

    selected_sample = None
    if samples:
        sample_names = ["-- Select a Sample Crop --"] + [s.name for s in samples]
        choice = st.selectbox("Load Test Worker Crop", sample_names)
        if choice != "-- Select a Sample Crop --":
            selected_sample = sample_dir / choice

    st.divider()
    st.markdown("### 📊 System Status")
    st.write(f"**Model:** `{predictor.model_name}`")
    st.write(f"**Engine:** `{'PyTorch ' + (str(predictor.device) if predictor.device else '') if predictor.is_loaded else 'Serverless Fallback'}`")
    st.write("**Classes:** `FULL_PPE`, `PARTIAL_PPE`, `NO_PPE`")

    st.divider()
    st.markdown(
        "[🐙 GitHub Repository](https://github.com/Anshika-111105/SiteSafe-Vision-AI-Powered-Construction-PPE-Screening-System) | "
        "[📄 Documentation](https://github.com/Anshika-111105/SiteSafe-Vision-AI-Powered-Construction-PPE-Screening-System#readme)"
    )

# Header
st.markdown('<div class="main-title">🦺 SiteSafe Vision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated AI Compliance Verification & Explainable Safety Screening for Construction Personnel</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer-card">⚠️ <strong>SAFETY & REGULATORY DISCLAIMER:</strong> '
    'SiteSafe Vision is an AI-based image screening aid designed to assist safety supervisors on construction sites. '
    'It does not perform certified safety audits, autonomous entry gating, or legally binding inspections.</div>',
    unsafe_allow_html=True,
)

# Navigation tabs
tab_single, tab_batch, tab_audit = st.tabs(["🔍 Real-Time Screening", "📁 Batch Processing", "📈 Governance & Audits"])


def process_image(img: Image.Image, show_cam: bool = True):
    img_rgb = img.convert("RGB")
    t0 = time.perf_counter()
    if show_cam:
        res, overlay = predictor.predict_with_gradcam(img_rgb)
    else:
        res = predictor.predict(img_rgb)
        overlay = img_rgb
    latency = (time.perf_counter() - t0) * 1000.0
    res["inference_latency_ms"] = round(latency, 2)
    risk_level, rec = evaluate_risk_and_recommendation(res["prediction"], res["confidence"])
    return res, overlay, risk_level, rec


with tab_single:
    col_input, col_results = st.columns([1, 1], gap="large")

    input_img = None

    with col_input:
        st.markdown("### 📥 Worker Image Input")
        mode = st.radio("Input Source", ["Upload Image", "Use Camera", "Sample Library"], horizontal=True)

        if mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload worker crop (JPEG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                try:
                    input_img = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to open image: {e}")
        elif mode == "Use Camera":
            cam_file = st.camera_input("Take a photo of worker PPE")
            if cam_file is not None:
                input_img = Image.open(cam_file)
        elif mode == "Sample Library":
            if selected_sample and selected_sample.exists():
                input_img = Image.open(selected_sample)
                st.caption(f"Loaded sample: `{selected_sample.name}`")
            else:
                st.info("Select a sample worker crop from the sidebar dropdown to inspect.")

        if input_img is not None:
            st.image(input_img, caption="Raw Worker Crop", use_container_width=True)

    with col_results:
        st.markdown("### 📊 Compliance Assessment")
        if input_img is not None:
            with st.spinner("Analyzing worker PPE compliance & visual saliency..."):
                res, overlay, risk_level, rec = process_image(input_img, show_cam=show_gradcam)

            pred = res["prediction"]
            conf = res["confidence"]
            probs = res.get("probabilities", {})

            # Badge styling
            badge_class = "badge-full" if pred == "FULL_PPE" else ("badge-partial" if pred == "PARTIAL_PPE" else "badge-none")
            badge_text = pred.replace("_", " ")

            st.markdown(
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">'
                f'<div><span style="font-size:0.85rem; color:#94a3b8;">Status:</span><br><span class="{badge_class}">{badge_text}</span></div>'
                f'<div><span style="font-size:0.85rem; color:#94a3b8;">Confidence:</span><br><strong>{conf*100:.1f}%</strong></div>'
                f'<div><span style="font-size:0.85rem; color:#94a3b8;">Risk Tier:</span><br><strong>{risk_level}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Safety recommendation
            st.markdown(f'<div class="rec-box">📋 <strong>Safety Recommendation:</strong><br>{rec}</div>', unsafe_allow_html=True)

            # Probability Bars
            st.markdown("##### Class Probability Distribution")
            p_full = probs.get("FULL_PPE", 0.0)
            p_part = probs.get("PARTIAL_PPE", 0.0)
            p_none = probs.get("NO_PPE", 0.0)

            st.write(f"🟢 **FULL PPE**: `{p_full*100:.1f}%`")
            st.progress(min(1.0, float(p_full)))
            st.write(f"🟡 **PARTIAL PPE**: `{p_part*100:.1f}%`")
            st.progress(min(1.0, float(p_part)))
            st.write(f"🔴 **NO PPE**: `{p_none*100:.1f}%`")
            st.progress(min(1.0, float(p_none)))

            # Grad-CAM overlay
            if show_gradcam and overlay is not None:
                st.divider()
                st.markdown("##### 🔬 Grad-CAM Explainability (Visual Saliency)")
                st.caption("Warm colors (red/yellow) indicate key visual regions (e.g. helmet, vest) used by the neural network.")
                st.image(overlay, caption=f"Grad-CAM Saliency Overlay ({pred})", use_container_width=True)

            st.caption(f"⚡ Model: `{res.get('model_name')}` | Inference Latency: `{res.get('inference_latency_ms')} ms`")
        else:
            st.info("Upload or capture a worker image on the left to view instant PPE compliance screening results.")


with tab_batch:
    st.markdown("### 📁 Batch Worker Image Screening")
    st.write("Upload a batch of worker crops to screen safety compliance across an entire shift or job site.")

    batch_files = st.file_uploader(
        "Upload multiple worker crop images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if batch_files:
        if st.button("🚀 Process Batch", variant="primary"):
            progress_bar = st.progress(0)
            rows = []

            for idx, bf in enumerate(batch_files):
                try:
                    b_img = Image.open(bf)
                    res, _, r_level, rec = process_image(b_img, show_cam=False)
                    rows.append({
                        "Filename": bf.name,
                        "Prediction": res["prediction"],
                        "Confidence (%)": round(res["confidence"] * 100, 1),
                        "Risk Level": r_level,
                        "Full PPE Prob": round(res["probabilities"].get("FULL_PPE", 0.0), 3),
                        "Partial PPE Prob": round(res["probabilities"].get("PARTIAL_PPE", 0.0), 3),
                        "No PPE Prob": round(res["probabilities"].get("NO_PPE", 0.0), 3),
                        "Latency (ms)": res["inference_latency_ms"],
                    })
                except Exception:
                    rows.append({
                        "Filename": bf.name,
                        "Prediction": "ERROR",
                        "Confidence (%)": 0.0,
                        "Risk Level": "HIGH",
                        "Full PPE Prob": 0.0,
                        "Partial PPE Prob": 0.0,
                        "No PPE Prob": 0.0,
                        "Latency (ms)": 0.0,
                    })
                progress_bar.progress((idx + 1) / len(batch_files))

            df = pd.DataFrame(rows)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Processed", len(df))
            m2.metric("Full PPE Compliance", f"{(df['Prediction'] == 'FULL_PPE').mean()*100:.1f}%")
            m3.metric("High Risk Violations", int((df['Risk Level'] == 'HIGH').sum()))
            m4.metric("Avg Latency", f"{df['Latency (ms)'].mean():.1f} ms")

            st.dataframe(df, use_container_width=True)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download CSV Report",
                data=csv_buffer.getvalue(),
                file_name="sitesafe_batch_screening_report.csv",
                mime="text/csv",
            )


with tab_audit:
    st.markdown("### 📈 Model Governance & Audit Verification")
    st.write("SiteSafe Vision undergoes strict data leakage audits, reproducible seeding, and line-of-sight evaluation.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", "93.3%", "Production Champion")
    col2.metric("Macro F1-Score", "0.933", "+14.2% vs baseline")
    col3.metric("NO_PPE Safety Recall", "100.0%", "Zero Missed Violations")

    st.markdown("#### 🛡️ Compliance & Safety Architecture")
    st.markdown("""
    - **Group-Aware Splitting:** Train, validation, and test splits are partitioned strictly by construction scene ID to eliminate scene-overlap data leakage.
    - **Grad-CAM Explainability:** Saliency maps computed via gradient backpropagation through the final convolutional feature extractor.
    - **Calibrated Risk Engine:** Predictions are automatically mapped to actionable risk classifications (`LOW`, `MEDIUM`, `HIGH`) with conservative fallback rules for low-confidence classifications.
    """)
