"""Automated PDF Diagnostic Research Report Generation using ReportLab."""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import config


def generate_pdf_report(
    sample_id: str,
    prediction: str,
    confidence: float,
    risk_score: float,
    image_features: Dict[str, float],
    sensor_features: Dict[str, float],
    notes: str = "",
    output_filename: Optional[str] = None,
    instrument_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a structured, clinical-grade Research Prototype PDF Report.

    Args:
        sample_id: Unique cell specimen identifier.
        prediction: "Healthy" or "Abnormal".
        confidence: Prediction confidence percentage (e.g. 96.4).
        risk_score: Model-derived continuous risk score (0.00 to 1.00).
        image_features: Extracted morphological and intensity metrics.
        sensor_features: Simulated electrochemical sensor metrics.
        notes: Optional analyst observations.
        output_filename: Optional custom PDF file name.
        instrument_metrics: Optional SECM and DPV/CA high-precision metrics.

    Returns:
        Path to the generated PDF report.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Diagnosis_Report_{sample_id}_{timestamp_str}.pdf"

    pdf_path = config.REPORTS_DIR / output_filename
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0D3B66"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555"),
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0D3B66"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#222222"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#8A4F00"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("AI-Assisted Single-Cell Diagnostic Prototype", title_style))
    story.append(Paragraph("Multimodal Optical Microscopy & High-Precision Electrochemical Analysis", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D3B66"), spaceAfter=10))

    # Disclaimer Box (only if configured)
    if config.DISCLAIMER_TEXT:
        disclaimer_table = Table(
            [[Paragraph(f"<b>CAUTION:</b> {config.DISCLAIMER_TEXT}", disclaimer_style)]],
            colWidths=[532],
        )
        disclaimer_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E7")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F4D06F")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(disclaimer_table)
        story.append(Spacer(1, 10))

    # Specimen & Metadata Table
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [Paragraph("<b>Sample ID:</b>", body_style), Paragraph(sample_id, body_style),
         Paragraph("<b>Analysis Date:</b>", body_style), Paragraph(now_str, body_style)],
        [Paragraph("<b>Specimen Type:</b>", body_style), Paragraph("Single Erythrocyte (RBC)", body_style),
         Paragraph("<b>Sensory Setup:</b>", body_style), Paragraph("Optical Brightfield + CV / EIS", body_style)],
        [Paragraph("<b>Instrumentation:</b>", body_style), Paragraph("Potentiostat DAQ (16-bit)", body_style),
         Paragraph("<b>System Core:</b>", body_style), Paragraph("Multimodal AI Pipeline", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[100, 166, 110, 156])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Diagnostic Findings Card
    story.append(Paragraph("Diagnostic Classification Findings", section_style))
    is_abnormal = (prediction.upper() == "ABNORMAL")
    badge_bg = colors.HexColor("#FDEDEC") if is_abnormal else colors.HexColor("#E8F8F5")
    badge_text_color = colors.HexColor("#922B21") if is_abnormal else colors.HexColor("#145A32")

    finding_p = Paragraph(f"<font size=13 color='{badge_text_color.hexval()}'><b>{prediction.upper()}</b></font>", body_style)
    conf_p = Paragraph(f"<b>{confidence:.1f}%</b>", body_style)
    risk_p = Paragraph(f"<b>{risk_score:.2f} / 1.00</b>", body_style)

    findings_table = Table(
        [
            [Paragraph("<b>Multimodal Prediction</b>", body_style), Paragraph("<b>Confidence</b>", body_style), Paragraph("<b>Research Risk Score</b>", body_style)],
            [finding_p, conf_p, risk_p],
        ],
        colWidths=[177, 177, 178],
    )
    findings_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F4F8")),
        ("BACKGROUND", (0, 1), (0, 1), badge_bg),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#B0C4DE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 14))

    # Quantitative Biophysical & Sensor Tables Side-by-Side
    story.append(Paragraph("Multimodal Biomarker Quantitative Extraction", section_style))

    img_rows = [
        [Paragraph("<b>Image Feature</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        [Paragraph("Cell Area", body_style), Paragraph(str(image_features.get("cell_area", "-")), body_style)],
        [Paragraph("Circularity (0-1)", body_style), Paragraph(str(image_features.get("circularity", "-")), body_style)],
        [Paragraph("Aspect Ratio", body_style), Paragraph(str(image_features.get("aspect_ratio", "-")), body_style)],
        [Paragraph("Texture Variance", body_style), Paragraph(str(image_features.get("texture_variance", "-")), body_style)],
        [Paragraph("Mean Intensity", body_style), Paragraph(str(image_features.get("mean_intensity", "-")), body_style)],
    ]
    img_tbl = Table(img_rows, colWidths=[160, 96])
    img_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAECEE")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))

    sensor_rows = [
        [Paragraph("<b>Electrochemical Feature</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        [Paragraph("Peak Anodic Current (Ipa)", body_style), Paragraph(f"{sensor_features.get('peak_anodic_current', '-')} µA", body_style)],
        [Paragraph("Charge Transfer Res. (Rct)", body_style), Paragraph(f"{sensor_features.get('charge_transfer_resistance', '-')} Ω", body_style)],
        [Paragraph("Capacitance (Cdl)", body_style), Paragraph(f"{sensor_features.get('double_layer_capacitance_uf', '-')} µF", body_style)],
        [Paragraph("Peak Potential Diff (ΔEp)", body_style), Paragraph(f"{sensor_features.get('delta_peak_potential', '-')} mV", body_style)],
        [Paragraph("Impedance |Z| at 1 Hz", body_style), Paragraph(f"{sensor_features.get('impedance_low_freq', '-')} Ω", body_style)],
    ]
    sensor_tbl = Table(sensor_rows, colWidths=[170, 96])
    sensor_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAECEE")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))

    side_by_side = Table([[img_tbl, sensor_tbl]], colWidths=[261, 271])
    side_by_side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(side_by_side)
    story.append(Spacer(1, 10))

    # Scanning Probe & Advanced Voltammetry Diagnostics
    if instrument_metrics:
        story.append(Paragraph("Scanning Probe & Voltammetric Telemetry", section_style))
        probe_rows = [
            [
                Paragraph("<b>SECM Feedback Regime:</b>", body_style),
                Paragraph(str(instrument_metrics.get("secm_feedback", "Negative Feedback")), body_style),
                Paragraph("<b>Min Local Current:</b>", body_style),
                Paragraph(f"{instrument_metrics.get('secm_min_current_na', '-')} nA", body_style),
            ],
            [
                Paragraph("<b>DPV Peak Amplitude:</b>", body_style),
                Paragraph(f"{instrument_metrics.get('dpv_peak_current_ua', '-')} µA", body_style),
                Paragraph("<b>Peak Phase Angle:</b>", body_style),
                Paragraph(f"{sensor_features.get('min_phase_angle', '-')}°", body_style),
            ],
        ]
        probe_tbl = Table(probe_rows, colWidths=[140, 160, 120, 112])
        probe_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(probe_tbl)
        story.append(Spacer(1, 10))

    # Notes & Auditor Sign-off
    if notes:
        story.append(Paragraph("Analyst Observations", section_style))
        story.append(Paragraph(notes, body_style))
        story.append(Spacer(1, 10))

    # Footer Signoff
    footer_text = (
        "Generated by CytoBioScan AI Diagnostic Pipeline. "
        "Algorithms: Deep CNN (Keras) + Multimodal Random Forest Ensemble. "
        f"Confidential Research Artifact — {now_str}."
    )
    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#A0AEC0"), spaceAfter=6))
    story.append(Paragraph(footer_text, subtitle_style))

    doc.build(story)
    return pdf_path
