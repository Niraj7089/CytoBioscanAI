"""Unit tests for the Research Prototype Risk Score computation and PDF report generation."""

import tempfile
from pathlib import Path
import pytest
from src.model.multimodal_model import compute_research_risk_score
from reports.generate_report import generate_pdf_report


def test_compute_research_risk_score_healthy():
    # Intact membrane: high Rct, low Ipa, high circularity, low texture variance
    img_feat = {"circularity": 0.92, "texture_variance": 18.0}
    sensor_feat = {"charge_transfer_resistance": 1520.0, "peak_anodic_current": 11.8}

    score = compute_research_risk_score(
        p_abnormal=0.04,
        image_features=img_feat,
        sensor_features=sensor_feat
    )
    assert 0.0 <= score <= 0.25


def test_compute_research_risk_score_abnormal():
    # Damaged membrane: low Rct, high Ipa, low circularity, high texture variance
    img_feat = {"circularity": 0.58, "texture_variance": 75.0}
    sensor_feat = {"charge_transfer_resistance": 590.0, "peak_anodic_current": 25.4}

    score = compute_research_risk_score(
        p_abnormal=0.96,
        image_features=img_feat,
        sensor_features=sensor_feat
    )
    assert 0.70 <= score <= 1.0


def test_generate_pdf_report(tmp_path):
    img_feat = {"cell_area": 1450.0, "circularity": 0.91, "aspect_ratio": 1.05, "texture_variance": 22.0, "mean_intensity": 115.0}
    sensor_feat = {"peak_anodic_current": 12.3, "charge_transfer_resistance": 1490.0, "double_layer_capacitance_uf": 2.1, "delta_peak_potential": 76.0, "impedance_low_freq": 1540.0}

    pdf_path = generate_pdf_report(
        sample_id="UNIT_TEST_SPECIMEN",
        prediction="Healthy",
        confidence=96.8,
        risk_score=0.14,
        image_features=img_feat,
        sensor_features=sensor_feat,
        notes="Automated pipeline validation report",
        output_filename="test_report.pdf"
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
