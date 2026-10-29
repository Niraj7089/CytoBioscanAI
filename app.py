"""Production-Grade Flask Web Application for CytoBioScan AI.

Single-Cell Imaging & Electrochemical Detection System (Research Prototype).
Zero third-party proprietary branding or external copyright dependencies.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
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
    simulate_secm_approach_curve,
    simulate_secm_2d_surface_scan,
    simulate_differential_pulse_voltammetry,
    simulate_chronoamperometry,
    format_instrument_telemetry_csv,
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
    get_sample_images_by_category,
    get_sample_image_by_id,
)
from reports.generate_report import generate_pdf_report

# Initialize Flask App
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
app.config["TEMPLATES_AUTO_RELOAD"] = True
TELEMETRY_CACHE = {}


def np_image_to_base64(img_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array into a base64 PNG string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


@app.before_request
def setup_database_once():
    """Ensure SQLite database is ready before servicing requests."""
    init_db()


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/")
@app.route("/index")
@app.route("/api/index")
@app.route("/api/index.py")
@app.route("/api")
def index():
    """Render the main interactive diagnostic studio dashboard."""
    return render_template("index.html")


@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    """Retrieve available benchmark library images from the SQLite sample database."""
    category = request.args.get("category", "Healthy")
    samples = get_sample_images_by_category(category)
    return jsonify({"samples": samples})


@app.route("/api/sample-image/<sample_id>", methods=["GET"])
def get_single_sample_image(sample_id):
    """Serve image binary directly from sample database."""
    sample = get_sample_image_by_id(sample_id)
    if not sample:
        stem = Path(sample_id).stem
        sample = get_sample_image_by_id(stem)

    if sample:
        raw_bytes = base64.b64decode(sample["image_base64"])
        return send_file(io.BytesIO(raw_bytes), mimetype="image/png")

    return jsonify({"error": "Sample image not found in database"}), 404


@app.route("/api/image-preview", methods=["GET"])
def image_preview():
    """Serve image preview from database or repository file path."""
    sample_id = request.args.get("sample_id")
    rel_path = request.args.get("path")

    key = sample_id or rel_path
    if not key:
        return "Missing sample identifier", 400

    stem = Path(key).stem
    db_sample = get_sample_image_by_id(stem)
    if db_sample:
        raw_bytes = base64.b64decode(db_sample["image_base64"])
        return send_file(io.BytesIO(raw_bytes), mimetype="image/png")

    clean_rel = key.replace("\\", "/").lstrip("/")
    full_path = (config.BASE_DIR / clean_rel).resolve()
    if str(full_path).startswith(str(config.BASE_DIR)) and full_path.exists():
        return send_file(full_path, mimetype="image/png")

    return "File not found", 404


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Execute complete multimodal AI analysis on a single cell."""
    try:
        # 1. Resolve Image Input from Upload or Database Library
        if "image_file" in request.files and request.files["image_file"].filename != "":
            file = request.files["image_file"]
            image_bytes = file.read()
            sample_id = Path(file.filename).stem
            raw_rgb = load_image(image_bytes)
        elif "sample_id" in request.form and request.form["sample_id"]:
            sample_id = request.form["sample_id"]
            stem = Path(sample_id).stem
            db_sample = get_sample_image_by_id(stem)
            if db_sample:
                raw_bytes = base64.b64decode(db_sample["image_base64"])
                raw_rgb = load_image(raw_bytes)
            else:
                return jsonify({"error": f"Sample '{sample_id}' not found in database"}), 404
        elif "preset_path" in request.form and request.form["preset_path"]:
            preset_path = request.form["preset_path"]
            stem = Path(preset_path).stem
            db_sample = get_sample_image_by_id(stem)
            if db_sample:
                sample_id = db_sample["sample_id"]
                raw_bytes = base64.b64decode(db_sample["image_base64"])
                raw_rgb = load_image(raw_bytes)
            else:
                clean_rel = preset_path.replace("\\", "/").lstrip("/")
                full_path = (config.BASE_DIR / clean_rel).resolve()
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

        # 4. Electrochemical Sensor Simulation (CV & EIS) or CSV Upload
        sensor_features = None
        cv_data = None
        eis_data = None

        if "sensor_file" in request.files and request.files["sensor_file"].filename != "":
            try:
                sensor_df = pd.read_csv(request.files["sensor_file"])
                if "cell_id" in sensor_df.columns and sample_id in sensor_df["cell_id"].values:
                    matched_row = sensor_df[sensor_df["cell_id"] == sample_id].iloc[0]
                else:
                    matched_row = sensor_df.iloc[0]

                extracted = {}
                for col in config.SENSOR_FEATURE_COLS:
                    if col in matched_row and pd.notnull(matched_row[col]):
                        extracted[col] = float(matched_row[col])
                if len(extracted) == len(config.SENSOR_FEATURE_COLS):
                    sensor_features = extracted
            except Exception as ex:
                app.logger.warning(f"Failed to read custom sensor CSV: {ex}")

        if sensor_features is None:
            cv_data = simulate_cyclic_voltammetry(cell_type=cell_condition, noise_level=noise_level)
            eis_data = simulate_impedance_spectroscopy(cell_type=cell_condition, noise_level=noise_level)
            sensor_features = extract_sensor_features(cv_data, eis_data)
        else:
            cv_data = simulate_cyclic_voltammetry(cell_type=cell_condition, noise_level=noise_level)
            eis_data = simulate_impedance_spectroscopy(cell_type=cell_condition, noise_level=noise_level)

        # 5. High-Precision Scanning Probe & Advanced Voltammetry Simulation
        secm_app = simulate_secm_approach_curve(cell_type=cell_condition, noise_level=noise_level)
        secm_2d = simulate_secm_2d_surface_scan(cell_type=cell_condition, mask=mask, noise_level=noise_level)
        dpv_data = simulate_differential_pulse_voltammetry(cell_type=cell_condition, noise_level=noise_level)
        ca_data = simulate_chronoamperometry(cell_type=cell_condition, noise_level=noise_level)

        instrument_metrics = {
            "secm_feedback": secm_app["feedback_mode"],
            "secm_min_current_na": secm_2d["min_probe_current_na"],
            "dpv_peak_current_ua": dpv_data["peak_current_ua"],
        }

        # Cache laboratory telemetry file
        telemetry_txt = format_instrument_telemetry_csv(
            sample_id=sample_id,
            cell_type=cell_condition,
            cv_data=cv_data,
            eis_data=eis_data,
            secm_approach=secm_app,
            dpv_data=dpv_data,
            ca_data=ca_data,
        )
        TELEMETRY_CACHE[sample_id] = telemetry_txt

        # 6. Multimodal Feature Fusion & Inference
        _, scaled_vec = prepare_single_sample_features(img_features, sensor_features)
        model = load_multimodal_model()

        pred_idx = int(model.predict(scaled_vec)[0])
        probabilities = model.predict_proba(scaled_vec)[0]
        p_abnormal = float(probabilities[1])
        prediction = config.INV_LABEL_MAP[pred_idx]
        confidence_pct = round(float(np.max(probabilities)) * 100.0, 1)

        # 7. Continuous Research Prototype Risk Score
        risk_score = compute_research_risk_score(
            p_abnormal=p_abnormal,
            image_features=img_features,
            sensor_features=sensor_features,
        )

        # 8. Database Persistence
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

        # 9. Clinical PDF Report Generation
        pdf_path = generate_pdf_report(
            sample_id=sample_id,
            prediction=prediction,
            confidence=confidence_pct,
            risk_score=risk_score,
            image_features=img_features,
            sensor_features=sensor_features,
            notes=notes,
            instrument_metrics=instrument_metrics,
        )

        # 10. Format Response Payload
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
            "instrument_metrics": instrument_metrics,
            "vis_raw": np_image_to_base64(raw_rgb),
            "vis_enhanced": np_image_to_base64(enhanced_uint8),
            "vis_overlay": np_image_to_base64(overlay),
            "cv_data": {
                "potential_v": cv_data["potential_v"].tolist(),
                "current_ua": cv_data["current_ua"].tolist(),
                "ipa": cv_data.get("ipa", round(float(np.max(cv_data["current_ua"])), 2)),
                "ipc": cv_data.get("ipc", round(float(np.min(cv_data["current_ua"])), 2)),
                "delta_ep": cv_data.get("delta_ep", 80.0),
            },
            "eis_data": {
                "frequency_hz": eis_data["frequency_hz"].tolist(),
                "z_real_ohm": eis_data["z_real_ohm"].tolist(),
                "z_imag_ohm": eis_data["z_imag_ohm"].tolist(),
                "z_mag_ohm": eis_data["z_mag_ohm"].tolist(),
                "phase_deg": eis_data["phase_deg"].tolist(),
                "rct": eis_data.get("rct", 1000.0),
                "cdl": round(float(eis_data.get("cdl", 2e-6)) * 1e6, 4) if float(eis_data.get("cdl", 2e-6)) < 0.1 else round(float(eis_data.get("cdl", 2e-6)), 4),
                "rs": eis_data.get("rs", 50.0),
            },
            "secm_data": {
                "approach": secm_app,
                "surface_2d": secm_2d,
            },
            "dpv_data": dpv_data,
            "ca_data": ca_data,
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


@app.route("/reports/<path:filename>", methods=["GET"])
def serve_report_file(filename):
    """Serve evaluation static plots (confusion matrix, ROC curve)."""
    safe_filename = Path(filename).name
    target = (config.REPORTS_DIR / safe_filename).resolve()
    if not target.exists():
        target = (config.STATIC_REPORTS_DIR / safe_filename).resolve()
    if not target.exists():
        return "File not found", 404
    return send_file(target)


@app.route("/api/download-report/<filename>", methods=["GET"])
def download_report(filename):
    """Download generated clinical research PDF report."""
    safe_filename = Path(filename).name
    report_file = (config.REPORTS_DIR / safe_filename).resolve()
    if not report_file.exists():
        report_file = (config.STATIC_REPORTS_DIR / safe_filename).resolve()
    if not report_file.exists():
        return "Report file not found", 404

    return send_file(report_file, mimetype="application/pdf", as_attachment=True, download_name=safe_filename)


@app.route("/api/download-telemetry/<sample_id>", methods=["GET"])
def download_telemetry(sample_id):
    """Download laboratory-grade multi-technique CSV telemetry log."""
    safe_id = Path(sample_id).name
    telemetry_content = TELEMETRY_CACHE.get(safe_id)
    if not telemetry_content:
        # Generate on-demand default
        cv_d = simulate_cyclic_voltammetry()
        eis_d = simulate_impedance_spectroscopy()
        secm_app = simulate_secm_approach_curve()
        dpv_d = simulate_differential_pulse_voltammetry()
        ca_d = simulate_chronoamperometry()
        telemetry_content = format_instrument_telemetry_csv(
            sample_id=safe_id,
            cell_type="Healthy",
            cv_data=cv_d,
            eis_data=eis_d,
            secm_approach=secm_app,
            dpv_data=dpv_d,
            ca_data=ca_d,
        )

    return Response(
        telemetry_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=telemetry_{safe_id}.csv"}
    )


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
    import threading
    from werkzeug.serving import run_simple

    def run_port_5000():
        try:
            run_simple("0.0.0.0", 5000, app, threaded=True, use_reloader=False)
        except Exception as e:
            print("Port 5000 error:", e)

    # Launch dual listener on port 5000 in background
    t = threading.Thread(target=run_port_5000, daemon=True)
    t.start()

    print("=" * 70)
    print("CytoBioScan AI -- Multimodal Single-Cell Diagnostic System")
    print("Production Flask Web Application is LIVE on BOTH ports:")
    print(" * Primary URL (Your existing browser tab): http://localhost:8501")
    print(" * Secondary URL (Standard Flask port):     http://localhost:5000")
    print("=" * 70)
    run_simple("0.0.0.0", 8501, app, threaded=True, use_reloader=False)
