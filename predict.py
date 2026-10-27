"""CLI and programmatic inference pipeline for the AI-Assisted Single-Cell Diagnosis System.

Performs:
1. Image loading, preprocessing, segmentation, and morphological feature extraction
2. Sensor parameter parsing or automated biophysical simulation
3. Feature fusion and standard scaling
4. Multimodal classification (Healthy vs. Abnormal)
5. Calibrated confidence and Research Prototype Risk Score computation
6. Optional audit database logging and PDF report generation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

import config
from src.image_processing.preprocessor import preprocess_image, load_image
from src.image_processing.segmentation import segment_cell
from src.image_processing.feature_extractor import extract_image_features
from src.sensor_simulation.simulator import (
    simulate_full_cell_signal,
    extract_sensor_features,
    get_sensor_feature_names,
)
from src.feature_engineering.fusion import prepare_single_sample_features
from src.model.multimodal_model import (
    load_multimodal_model,
    compute_research_risk_score,
)
from src.database.db_manager import save_prediction
from reports.generate_report import generate_pdf_report


def predict_cell(
    image_input: Union[str, Path, bytes, np.ndarray],
    sensor_input: Optional[Union[Dict[str, Any], str, Path, pd.Series]] = None,
    sample_id: Optional[str] = None,
    save_to_db: bool = False,
    generate_report: bool = False,
    notes: str = ""
) -> Dict[str, Any]:
    """Execute complete end-to-end multimodal inference for a single cell.

    Args:
        image_input: Image path, raw bytes, or numpy array.
        sensor_input: Electrochemical features dict/series, or None to auto-simulate.
        sample_id: Specimen identifier string.
        save_to_db: Whether to persist result in SQLite database.
        generate_report: Whether to produce a PDF report.
        notes: Optional analyst observations.

    Returns:
        Structured dictionary containing prediction, confidence, risk score, and all extracted features.
    """
    if sample_id is None:
        if isinstance(image_input, (str, Path)):
            sample_id = Path(image_input).stem
        else:
            sample_id = "Cell_Specimen"

    # 1. Image Preprocessing and Feature Extraction
    normalized_img, enhanced_uint8 = preprocess_image(image_input)
    mask, primary_contour, overlay = segment_cell(enhanced_uint8)
    image_features = extract_image_features(enhanced_uint8, mask=mask)

    # 2. Sensor Features
    if sensor_input is None:
        # Default biophysical simulation (unbiased standard baseline)
        raw_signal = simulate_full_cell_signal(cell_type="Healthy", noise_level=0.03)
        sensor_features = raw_signal["features"]
    elif isinstance(sensor_input, (str, Path)):
        # Load from CSV
        df = pd.read_csv(sensor_input)
        sensor_features = {k: float(df[k].iloc[0]) for k in get_sensor_feature_names() if k in df.columns}
    elif isinstance(sensor_input, pd.Series):
        sensor_features = {k: float(sensor_input[k]) for k in get_sensor_feature_names() if k in sensor_input}
    elif isinstance(sensor_input, dict):
        sensor_features = {k: float(sensor_input.get(k, 0.0)) for k in get_sensor_feature_names()}
    else:
        raise TypeError(f"Unsupported sensor_input type: {type(sensor_input)}")

    # 3. Multimodal Feature Fusion & Inference
    fused_raw, scaled_vec = prepare_single_sample_features(image_features, sensor_features)
    model = load_multimodal_model()

    pred_class_idx = int(model.predict(scaled_vec)[0])
    probabilities = model.predict_proba(scaled_vec)[0]
    p_abnormal = float(probabilities[1])
    predicted_class = config.INV_LABEL_MAP[pred_class_idx]
    confidence_pct = round(float(np.max(probabilities)) * 100.0, 2)

    # 4. Continuous Research Prototype Risk Score
    risk_score = compute_research_risk_score(
        p_abnormal=p_abnormal,
        image_features=image_features,
        sensor_features=sensor_features
    )

    # 5. Database Logging
    db_id = None
    if save_to_db:
        db_id = save_prediction(
            sample_id=sample_id,
            image_name=sample_id,
            prediction=predicted_class,
            confidence=confidence_pct,
            risk_score=risk_score,
            cell_area=image_features.get("cell_area"),
            circularity=image_features.get("circularity"),
            peak_current=sensor_features.get("peak_anodic_current"),
            rct_impedance=sensor_features.get("charge_transfer_resistance"),
            notes=notes,
        )

    # 6. PDF Report Generation
    pdf_report_path = None
    if generate_report:
        from reports.generate_report import generate_pdf_report
        pdf_report_path = str(generate_pdf_report(
            sample_id=sample_id,
            prediction=predicted_class,
            confidence=confidence_pct,
            risk_score=risk_score,
            image_features=image_features,
            sensor_features=sensor_features,
            notes=notes,
        ))

    return {
        "sample_id": sample_id,
        "prediction": predicted_class,
        "confidence": confidence_pct,
        "risk_score": risk_score,
        "p_abnormal": round(p_abnormal, 4),
        "db_record_id": db_id,
        "pdf_report_path": pdf_report_path,
        "image_features": image_features,
        "sensor_features": sensor_features,
        "segmented_overlay": overlay,
        "binary_mask": mask,
        "disclaimer": config.DISCLAIMER_TEXT,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict cell malignancy state using multimodal AI.")
    parser.add_argument("--image", type=str, required=True, help="Path to cell microscopy image.")
    parser.add_argument("--sensor", type=str, default=None, help="Optional CSV file with electrochemical parameters.")
    parser.add_argument("--save-db", action="store_true", help="Store prediction in SQLite database.")
    parser.add_argument("--report", action="store_true", help="Generate PDF clinical research report.")
    parser.add_argument("--notes", type=str, default="", help="Optional analyst notes.")

    args = parser.parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)

    result = predict_cell(
        image_input=img_path,
        sensor_input=args.sensor,
        sample_id=img_path.stem,
        save_to_db=args.save_db,
        generate_report=args.report,
        notes=args.notes
    )

    print("\n" + "=" * 50)
    print("AI SINGLE-CELL MULTIMODAL INFERENCE RESULT")
    print("=" * 50)
    print(f"Sample ID:        {result['sample_id']}")
    print(f"Prediction:       {result['prediction']}")
    print(f"Confidence:       {result['confidence']}%")
    print(f"Proto Risk Score: {result['risk_score']} / 1.00")
    if result["db_record_id"]:
        print(f"Database ID:      #{result['db_record_id']}")
    if result["pdf_report_path"]:
        print(f"Report Generated: {result['pdf_report_path']}")
    print("=" * 50)
    if result.get("disclaimer"):
        print(f"Notice: {result['disclaimer']}\n")


if __name__ == "__main__":
    main()
