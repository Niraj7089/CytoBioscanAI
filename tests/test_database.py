"""Unit tests for the SQLite database manager."""

import tempfile
from pathlib import Path
import pytest
from src.database.db_manager import (
    init_db,
    save_prediction,
    fetch_all_predictions,
    get_summary_statistics,
    clear_history,
    get_sample_images_by_category,
    get_sample_image_by_id,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_predictions.db"
    init_db(db_file)
    return db_file


def test_database_crud(temp_db):
    stats0 = get_summary_statistics(temp_db)
    assert stats0["total_scans"] == 0

    rec_id = save_prediction(
        sample_id="SAMPLE_TEST_01",
        image_name="cell_01.png",
        prediction="Healthy",
        confidence=95.5,
        risk_score=0.12,
        cell_area=1200.0,
        circularity=0.92,
        peak_current=12.5,
        rct_impedance=1480.0,
        notes="Automated test note",
        db_path=temp_db
    )
    assert rec_id == 1

    df = fetch_all_predictions(temp_db)
    assert len(df) == 1
    assert df["sample_id"].iloc[0] == "SAMPLE_TEST_01"
    assert df["prediction"].iloc[0] == "Healthy"

    stats1 = get_summary_statistics(temp_db)
    assert stats1["total_scans"] == 1
    assert stats1["healthy_count"] == 1

    clear_history(temp_db)
    df_empty = fetch_all_predictions(temp_db)
    assert len(df_empty) == 0


def test_sample_library_database(temp_db):
    healthy_samples = get_sample_images_by_category("Healthy", db_path=temp_db)
    abnormal_samples = get_sample_images_by_category("Abnormal", db_path=temp_db)
    assert len(healthy_samples) >= 10
    assert len(abnormal_samples) >= 10

    first_id = healthy_samples[0]["sample_id"]
    sample_rec = get_sample_image_by_id(first_id, db_path=temp_db)
    assert sample_rec is not None
    assert sample_rec["sample_id"] == first_id
    assert sample_rec["category"] == "Healthy"
    assert len(sample_rec["image_base64"]) > 100
