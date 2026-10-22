"""Streamlit Interactive Diagnostic Dashboard for AI-Assisted Single-Cell Diagnosis.

Research Prototype integrating Optical Microscopy Computer Vision,
Electrochemical Sensor Simulation, and Multimodal AI Fusion.
"""

import sys
from pathlib import Path
from io import BytesIO
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.image_processing.preprocessor import preprocess_image, load_image
from src.image_processing.segmentation import segment_cell
from src.image_processing.feature_extractor import extract_image_features
from src.sensor_simulation.simulator import (
    simulate_cyclic_voltammetry,
    simulate_impedance_spectroscopy,
    extract_sensor_features,
    get_sensor_feature_names,
)
from src.feature_engineering.fusion import prepare_single_sample_features
from src.model.multimodal_model import (
    load_multimodal_model,
    compute_research_risk_score,
)
from src.database.db_manager import (
    init_db,
    save_prediction,
    fetch_all_predictions,
    get_summary_statistics,
    clear_history,
)
from reports.generate_report import generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title="AI Single-Cell Diagnosis Prototype",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D3B66;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #FFF9E6;
        border-left: 5px solid #E28743;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #7A4B00;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-healthy {
        background-color: #E8F8F5;
        color: #145A32;
        padding: 0.4rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.4rem;
        display: inline-block;
        border: 1px solid #A2D9CE;
    }
    .status-abnormal {
        background-color: #FDEDEC;
        color: #922B21;
        padding: 0.4rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.4rem;
        display: inline-block;
        border: 1px solid #F5B7B1;
    }
</style>
""", unsafe_allow_html=True)


def plot_sensor_signals(cv_data: dict, eis_data: dict):
    """Render interactive dual-panel plot for Cyclic Voltammetry and EIS Nyquist."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)

    # 1. Cyclic Voltammogram
    ax1.plot(cv_data["potential_v"], cv_data["current_ua"], color="#0066cc", lw=2)
    ax1.axhline(0, color="#888888", linestyle="--", lw=0.8, alpha=0.7)
    ax1.axvline(0, color="#888888", linestyle="--", lw=0.8, alpha=0.7)
    ax1.set_title("Cyclic Voltammogram (CV)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Potential vs. Ref (V)", fontsize=9)
    ax1.set_ylabel("Current (µA)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Nyquist Plot
    ax2.plot(eis_data["z_real_ohm"], eis_data["z_imag_ohm"], "o-", color="#800080", ms=3.5, lw=1.5)
    ax2.set_title("EIS Nyquist Spectrum (-Z'' vs Z')", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Z' Real (Ω)", fontsize=9)
    ax2.set_ylabel("-Z'' Imaginary (Ω)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig


def main():
    init_db()

    # Header
    st.markdown('<div class="main-header">🔬 AI-Assisted Single-Cell Diagnostic System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multimodal Biosensing Platform Integrating Optical Microscopy & Electrochemical Impedance Detection</div>', unsafe_allow_html=True)

    # Regulatory Disclaimer Banner
    st.markdown(f'<div class="disclaimer-box"><b>⚠️ RESEARCH PROTOTYPE NOTICE:</b> {config.DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/microscope.png", width=64)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Module:",
        ["Single-Cell Live Analysis", "Audit Log & History", "Architecture & Diagnostics"]
    )

    if app_mode == "Single-Cell Live Analysis":
        render_analysis_studio()
    elif app_mode == "Audit Log & History":
        render_history_view()
    elif app_mode == "Architecture & Diagnostics":
        render_system_specs()


def render_analysis_studio():
    st.subheader("Single-Cell Specimen Analysis")

    col_input_left, col_input_right = st.columns([1, 1])

    # Left: Image Selection
    with col_input_left:
        st.markdown("#### 1. Optical Microscopy Input")
        img_source = st.radio(
            "Image Input Source:",
            ["Choose from Test Library", "Upload Image File (.png, .jpg)"],
            horizontal=True
        )

        image_to_process = None
        sample_name = "Specimen"

        if img_source == "Choose from Test Library":
            preset_class = st.selectbox("Select Target Class:", ["Healthy (Uninfected RBC)", "Abnormal (Parasitized RBC)"])
            class_folder = "Healthy" if "Healthy" in preset_class else "Abnormal"
            sample_dir = config.TEST_DIR / class_folder

            if sample_dir.exists():
                available_files = sorted(list(sample_dir.glob("*.png")))
                if available_files:
                    selected_file = st.selectbox("Select Benchmark Cell Sample:", available_files, format_func=lambda x: x.name)
                    image_to_process = str(selected_file)
                    sample_name = selected_file.stem
                else:
                    st.warning("No sample files found in library directory. Please train or download the dataset.")
            else:
                st.warning("Dataset directory not initialized.")
        else:
            uploaded_file = st.file_uploader("Upload Single-Cell Micrograph", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image_to_process = uploaded_file.read()
                sample_name = Path(uploaded_file.name).stem

    # Right: Sensor Selection
    with col_input_right:
        st.markdown("#### 2. Electrochemical Sensor Input")
        sensor_mode = st.radio(
            "Sensor Data Source:",
            ["Simulate Realistic Electrochemical Signal", "Upload Custom Sensor CSV"],
            horizontal=True
        )

        custom_sensor_data = None
        sim_class = "Healthy"

        if sensor_mode == "Simulate Realistic Electrochemical Signal":
            st.info("Simulating Cyclic Voltammetry (CV) & Impedance Spectroscopy (EIS) across gold microelectrodes.")
            sim_class = st.selectbox("Simulate Cell Membrane Condition:", ["Healthy (Intact Membrane)", "Abnormal (Compromised Membrane)"])
            sim_class_key = "Healthy" if "Healthy" in sim_class else "Abnormal"
            noise_slider = st.slider("Electrochemical Noise Level (%):", min_value=1.0, max_value=15.0, value=4.0, step=0.5) / 100.0

            # Generate sensor curves
            cv_sim = simulate_cyclic_voltammetry(cell_type=sim_class_key, noise_level=noise_slider)
            eis_sim = simulate_impedance_spectroscopy(cell_type=sim_class_key, noise_level=noise_slider)
            sensor_features = extract_sensor_features(cv_sim, eis_sim)
        else:
            sensor_file = st.file_uploader("Upload Sensor Measurement CSV", type=["csv"])
            if sensor_file is not None:
                df = pd.read_csv(sensor_file)
                st.write("Uploaded Sensor Metrics:", df.head(1))
                sensor_features = {k: float(df[k].iloc[0]) for k in get_sensor_feature_names() if k in df.columns}
                cv_sim = simulate_cyclic_voltammetry(cell_type="Healthy", noise_level=0.03)
                eis_sim = simulate_impedance_spectroscopy(cell_type="Healthy", noise_level=0.03)
            else:
                st.info("Please upload a CSV or switch to simulation mode.")
                cv_sim = simulate_cyclic_voltammetry(cell_type="Healthy", noise_level=0.04)
                eis_sim = simulate_impedance_spectroscopy(cell_type="Healthy", noise_level=0.04)
                sensor_features = extract_sensor_features(cv_sim, eis_sim)

    st.markdown("---")
    analyst_notes = st.text_input("Analyst Observations / Study Notes:", placeholder="E.g., In-vitro assay batch #4, 37°C incubation.")

    analyze_btn = st.button("🚀 Run Multimodal AI Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if image_to_process is None:
            st.error("Please provide or select a single-cell image before analyzing.")
            return

        with st.spinner("Processing optical micrograph, segmenting cell, and performing multimodal AI inference..."):
            # Check model files
            if not config.FUSION_MODEL_PATH.exists() or not config.SCALER_PATH.exists():
                st.error("Model artifacts not found! Please run 'python train.py' first to train the models.")
                return

            # Preprocess & Segment
            norm_img, enhanced_uint8 = preprocess_image(image_to_process)
            mask, contour, overlay = segment_cell(enhanced_uint8)
            img_features = extract_image_features(enhanced_uint8, mask=mask)

            # Feature Fusion & Inference
            _, scaled_vec = prepare_single_sample_features(img_features, sensor_features)
            model = load_multimodal_model()

            pred_class_idx = int(model.predict(scaled_vec)[0])
            probabilities = model.predict_proba(scaled_vec)[0]
            p_abnormal = float(probabilities[1])
            predicted_class = config.INV_LABEL_MAP[pred_class_idx]
            confidence_pct = round(float(np.max(probabilities)) * 100.0, 1)

            risk_score = compute_research_risk_score(
                p_abnormal=p_abnormal,
                image_features=img_features,
                sensor_features=sensor_features
            )

            # Save to SQLite
            rec_id = save_prediction(
                sample_id=sample_name,
                image_name=sample_name,
                prediction=predicted_class,
                confidence=confidence_pct,
                risk_score=risk_score,
                cell_area=img_features.get("cell_area"),
                circularity=img_features.get("circularity"),
                peak_current=sensor_features.get("peak_anodic_current"),
                rct_impedance=sensor_features.get("charge_transfer_resistance"),
                notes=analyst_notes,
            )

        # Presentation of Results
        st.success(f"Analysis complete for sample **{sample_name}** (Audit Record #{rec_id})")

        # Top Metric Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Multimodal Classification")
            if predicted_class == "Abnormal":
                st.markdown('<span class="status-abnormal">ABNORMAL</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-healthy">HEALTHY</span>', unsafe_allow_html=True)
            st.caption("Morphological + Electrochemical Consensus")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Model Confidence")
            st.markdown(f"<h2 style='color:#0D3B66;margin:0;'>{confidence_pct}%</h2>", unsafe_allow_html=True)
            st.caption(f"Posterior Probability: {probabilities[pred_class_idx]:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Research Risk Score")
            risk_color = "#922B21" if risk_score >= 0.65 else ("#E28743" if risk_score >= 0.35 else "#145A32")
            st.markdown(f"<h2 style='color:{risk_color};margin:0;'>{risk_score:.2f} <small style='font-size:0.9rem;color:#666;'>/ 1.00</small></h2>", unsafe_allow_html=True)
            risk_desc = "High Cellular Atypia" if risk_score >= 0.65 else ("Moderate Anomaly" if risk_score >= 0.35 else "Low Anomaly")
            st.caption(risk_desc)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Visualizations
        tab_vis1, tab_vis2 = st.tabs(["🖼️ Image Processing & Segmentation", "⚡ Electrochemical Signatures"])

        with tab_vis1:
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.image(load_image(image_to_process), caption="Raw Microscopic Cell", use_container_width=True)
            with col_v2:
                st.image(enhanced_uint8, caption="Contrast Enhanced (CLAHE)", use_container_width=True)
            with col_v3:
                st.image(overlay, caption="Segmented Cell & Boundary Outline", use_container_width=True)

        with tab_vis2:
            sensor_fig = plot_sensor_signals(cv_sim, eis_sim)
            st.pyplot(sensor_fig)

        # Quantitative Breakdown Tables
        st.markdown("#### Quantitative Multimodal Biomarkers")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Morphological & Texture Descriptors**")
            df_img = pd.DataFrame([
                {"Biomarker": "Cell Area (px²)", "Value": img_features.get("cell_area")},
                {"Biomarker": "Circularity (0 to 1)", "Value": img_features.get("circularity")},
                {"Biomarker": "Aspect Ratio", "Value": img_features.get("aspect_ratio")},
                {"Biomarker": "Texture Variance", "Value": img_features.get("texture_variance")},
                {"Biomarker": "Mean Intensity", "Value": img_features.get("mean_intensity")},
            ])
            st.table(df_img)

        with col_f2:
            st.markdown("**Electrochemical Sensor Descriptors**")
            df_sen = pd.DataFrame([
                {"Biomarker": "Peak Anodic Current (Ipa)", "Value": f"{sensor_features.get('peak_anodic_current')} µA"},
                {"Biomarker": "Charge Transfer Res. (Rct)", "Value": f"{sensor_features.get('charge_transfer_resistance')} Ω"},
                {"Biomarker": "Capacitance (Cdl)", "Value": f"{sensor_features.get('double_layer_capacitance_uf')} µF"},
                {"Biomarker": "Peak Separation (ΔEp)", "Value": f"{sensor_features.get('delta_peak_potential')} mV"},
                {"Biomarker": "Impedance |Z| at 1 Hz", "Value": f"{sensor_features.get('impedance_low_freq')} Ω"},
            ])
            st.table(df_sen)

        # Generate & Download Report
        st.markdown("#### Export Documentation")
        pdf_path = generate_pdf_report(
            sample_id=sample_name,
            prediction=predicted_class,
            confidence=confidence_pct,
            risk_score=risk_score,
            image_features=img_features,
            sensor_features=sensor_features,
            notes=analyst_notes
        )
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📄 Download Official PDF Research Report",
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
            type="secondary"
        )


def render_history_view():
    st.subheader("Audit Log & Specimen History")
    stats = get_summary_statistics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cells Scanned", stats["total_scans"])
    c2.metric("Abnormal Detection Rate", f"{stats['abnormal_rate']}%")
    c3.metric("Average Model Confidence", f"{stats['avg_confidence']}%")
    c4.metric("Average Research Risk Score", f"{stats['avg_risk_score']}")

    df = fetch_all_predictions()
    if df.empty:
        st.info("No prediction history recorded yet. Run a single-cell analysis to log results.")
    else:
        st.dataframe(df, use_container_width=True)

        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export History to CSV", csv_data, "prediction_audit_log.csv", "text/csv")
        with col_h2:
            if st.button("🗑️ Clear Audit History", type="secondary"):
                clear_history()
                st.rerun()


def render_system_specs():
    st.subheader("System Architecture & Benchmark Specs")
    st.markdown("""
    ### Multimodal Pipeline Architecture
    ```text
    ┌───────────────────────────────────┐     ┌───────────────────────────────────┐
    │     Single-Cell Micrograph        │     │  Simulated Electrochemical Data   │
    │   (Giemsa Optical Microscopy)     │     │      (CV & EIS Spectrograms)      │
    └─────────────────┬─────────────────┘     └─────────────────┬─────────────────┘
                      │                                         │
                      ▼                                         ▼
    ┌───────────────────────────────────┐     ┌───────────────────────────────────┐
    │ Image Preprocessing & CLAHE       │     │ Randles Circuit Extraction        │
    │ Morphological Otsu Segmentation   │     │ (Peak Current, Rct, Cdl, Phase)   │
    └─────────────────┬─────────────────┘     └─────────────────┬─────────────────┘
                      │                                         │
                      └───────────────────┬─────────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │ Multimodal Feature Fusion & Scaling   │
                      └───────────────────┬───────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │ Random Forest Multimodal Ensemble     │
                      └───────────────────┬───────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Healthy / Abnormal Classification  │
                      │      + Research Prototype Risk Score  │
                      └───────────────────────────────────────┘
    ```
    """)

    st.markdown("### Published Evaluation Metrics")
    metrics_file = config.REPORTS_DIR / "confusion_matrix.png"
    if metrics_file.exists():
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.image(str(config.REPORTS_DIR / "confusion_matrix.png"), caption="Confusion Matrix")
        with c_m2:
            st.image(str(config.REPORTS_DIR / "roc_curve.png"), caption="ROC Curve")

    feat_imp = config.REPORTS_DIR / "feature_importance.png"
    if feat_imp.exists():
        st.image(str(feat_imp), caption="Top Multimodal Biomarker Importances")


if __name__ == "__main__":
    main()
