"""Unit tests for the SQLite database manager."""

from pathlib import Path
import pytest
from src.database.db_manager import (
    init_db,
    save_prediction,
    fetch_all_predictions,
    fetch_recent_predictions,
    get_summary_statistics,
    clear_history,
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

    recent = fetch_recent_predictions(limit=10, db_path=temp_db)
    assert len(recent) == 1
    assert recent[0]["sample_id"] == "SAMPLE_TEST_01"

    stats1 = get_summary_statistics(temp_db)
    assert stats1["total_scans"] == 1
    assert stats1["healthy_count"] == 1

    clear_history(temp_db)
    df_empty = fetch_all_predictions(temp_db)
    assert len(df_empty) == 0
