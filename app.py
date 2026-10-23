"""Production-Grade Flask Web Application for CytoBioScan AI.

Single-Cell Imaging & Electrochemical Detection System (Research Prototype).
Zero third-party proprietary branding or external copyright dependencies.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys

# Seamless dual-mode support: If launched via `streamlit run app.py`, delegate to Streamlit UI
try:
    import streamlit.runtime as st_runtime
    if st_runtime.exists():
        import streamlit_app
        streamlit_app.main()
        sys.exit(0)
except ImportError:
    pass
import base64
import io
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    Response,
)

# Project Root Setup
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
)
from src.feature_engineering.fusion import prepare_single_sample_features
from src.model.multimodal_model import (
    load_multimodal_model,
    compute_research_risk_score,
)
from src.database.db_manager import (
    init_db,
    save_prediction,
    fetch_recent_predictions,
    fetch_all_predictions,
    get_summary_statistics,
    clear_history,
)
from reports.generate_report import generate_pdf_report

# Initialize Flask App
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


def np_image_to_base64(img_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array into a base64 PNG string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


@app.before_request
def setup_database_once():
    """Ensure SQLite database is ready before servicing requests."""
    init_db()


@app.route("/")
def index():
    """Render the main interactive diagnostic studio dashboard."""
    return render_template("index.html")


@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    """Retrieve available benchmark library images for fast demo testing."""
    category = request.args.get("category", "Healthy")
    target_dir = config.TEST_DIR / category

    if not target_dir.exists():
        return jsonify({"samples": []})

    samples = []
    for p in sorted(target_dir.glob("*.png"))[:10]:
        samples.append({
            "name": p.name,
            "path": str(p.relative_to(config.BASE_DIR)),
            "url": f"/api/image-preview?path={str(p.relative_to(config.BASE_DIR))}",
        })
    return jsonify({"samples": samples})


@app.route("/api/image-preview", methods=["GET"])
def image_preview():
    """Serve image preview from repository file path."""
    rel_path = request.args.get("path")
    if not rel_path:
        return "Missing path", 400

    full_path = (config.BASE_DIR / rel_path).resolve()
    # Path traversal protection
    if not str(full_path).startswith(str(config.BASE_DIR)):
        return "Unauthorized path", 403

    if not full_path.exists():
        return "File not found", 404

    return send_file(full_path, mimetype="image/png")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Execute complete multimodal AI analysis on a single cell."""
    try:
        # 1. Resolve Image Input
        if "image_file" in request.files and request.files["image_file"].filename != "":
            file = request.files["image_file"]
            image_bytes = file.read()
            sample_id = Path(file.filename).stem
            raw_rgb = load_image(image_bytes)
        elif "preset_path" in request.form and request.form["preset_path"]:
            rel_path = request.form["preset_path"]
            full_path = (config.BASE_DIR / rel_path).resolve()
            if not full_path.exists() or not str(full_path).startswith(str(config.BASE_DIR)):
                return jsonify({"error": "Invalid benchmark sample path"}), 400
            sample_id = full_path.stem
            raw_rgb = load_image(full_path)
        else:
            return jsonify({"error": "No image file or benchmark sample provided"}), 400

        # 2. Parse Sensor Simulation Options
        cell_condition = request.form.get("cell_condition", "Healthy")
        noise_level = float(request.form.get("noise_level", 0.04))
        notes = request.form.get("notes", "")

        # 3. Image Preprocessing & Feature Extraction
        norm_img, enhanced_uint8 = preprocess_image(raw_rgb)
        mask, primary_contour, overlay = segment_cell(enhanced_uint8)
        img_features = extract_image_features(enhanced_uint8, mask=mask)

        # 4. Electrochemical Sensor Simulation (CV & EIS)
        cv_data = simulate_cyclic_voltammetry(cell_type=cell_condition, noise_level=noise_level)
        eis_data = simulate_impedance_spectroscopy(cell_type=cell_condition, noise_level=noise_level)
        sensor_features = extract_sensor_features(cv_data, eis_data)

        # 5. Multimodal Feature Fusion & Inference
        _, scaled_vec = prepare_single_sample_features(img_features, sensor_features)
        model = load_multimodal_model()

        pred_idx = int(model.predict(scaled_vec)[0])
        probabilities = model.predict_proba(scaled_vec)[0]
        p_abnormal = float(probabilities[1])
        prediction = config.INV_LABEL_MAP[pred_idx]
        confidence_pct = round(float(np.max(probabilities)) * 100.0, 1)

        # 6. Continuous Research Prototype Risk Score
        risk_score = compute_research_risk_score(
            p_abnormal=p_abnormal,
            image_features=img_features,
            sensor_features=sensor_features,
        )

        # 7. Database Persistence
        db_id = save_prediction(
            sample_id=sample_id,
            image_name=sample_id,
            prediction=prediction,
            confidence=confidence_pct,
            risk_score=risk_score,
            cell_area=img_features.get("cell_area"),
            circularity=img_features.get("circularity"),
            peak_current=sensor_features.get("peak_anodic_current"),
            rct_impedance=sensor_features.get("charge_transfer_resistance"),
            notes=notes,
        )

        # 8. Clinical PDF Report Generation
        pdf_path = generate_pdf_report(
            sample_id=sample_id,
            prediction=prediction,
            confidence=confidence_pct,
            risk_score=risk_score,
            image_features=img_features,
            sensor_features=sensor_features,
            notes=notes,
        )

        # 9. Format Response Payload
        response_data = {
            "sample_id": sample_id,
            "prediction": prediction,
            "confidence": confidence_pct,
            "risk_score": risk_score,
            "p_abnormal": round(p_abnormal, 4),
            "db_record_id": db_id,
            "report_filename": pdf_path.name,
            "image_features": img_features,
            "sensor_features": sensor_features,
            "vis_raw": np_image_to_base64(raw_rgb),
            "vis_enhanced": np_image_to_base64(enhanced_uint8),
            "vis_overlay": np_image_to_base64(overlay),
            "cv_data": {
                "potential_v": cv_data["potential_v"].tolist(),
                "current_ua": cv_data["current_ua"].tolist(),
            },
            "eis_data": {
                "frequency_hz": eis_data["frequency_hz"].tolist(),
                "z_real_ohm": eis_data["z_real_ohm"].tolist(),
                "z_imag_ohm": eis_data["z_imag_ohm"].tolist(),
            },
        }
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Inference error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Retrieve audit history records and telemetry summary statistics."""
    records = fetch_recent_predictions(limit=100)
    stats = get_summary_statistics()
    return jsonify({"records": records, "stats": stats})


@app.route("/api/history/clear", methods=["POST"])
def clear_audit_history():
    """Clear all records from audit database."""
    clear_history()
    return jsonify({"status": "cleared"})


@app.route("/api/download-report/<filename>", methods=["GET"])
def download_report(filename):
    """Download generated clinical research PDF report."""
    safe_filename = Path(filename).name
    report_file = (config.REPORTS_DIR / safe_filename).resolve()

    if not report_file.exists() or not str(report_file).startswith(str(config.REPORTS_DIR)):
        return "Report file not found", 404

    return send_file(report_file, mimetype="application/pdf", as_attachment=True, download_name=safe_filename)


@app.route("/api/export-csv", methods=["GET"])
def export_csv():
    """Stream SQLite audit history table as a CSV download."""
    df = fetch_all_predictions()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cytobioscan_audit_log.csv"}
    )


if __name__ == "__main__":
    print("=" * 70)
    print("CytoBioScan AI — Multimodal Single-Cell Diagnostic System")
    print("Production Flask Web Interface")
    print("Starting server at: http://localhost:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)
