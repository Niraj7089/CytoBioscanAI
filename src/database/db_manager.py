"""SQLite database manager for persisting single-cell diagnostic history and audit trail."""

import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

import config


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get SQLite database connection, ensuring parent directory exists with serverless fallback."""
    if db_path is None:
        db_path = config.DB_PATH

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
    except (OSError, PermissionError, sqlite3.OperationalError):
        # Fallback to temp directory or in-memory database in read-only serverless runtimes
        fallback_dir = Path(tempfile.gettempdir())
        try:
            fallback_path = fallback_dir / "predictions.db"
            conn = sqlite3.connect(str(fallback_path))
        except Exception:
            conn = sqlite3.connect(":memory:")

    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables for clinical audit history."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                image_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_score REAL NOT NULL,
                cell_area REAL,
                circularity REAL,
                peak_current REAL,
                rct_impedance REAL,
                notes TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_history (timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction ON prediction_history (prediction);")
    conn.close()


def save_prediction(
    sample_id: str,
    image_name: str,
    prediction: str,
    confidence: float,
    risk_score: float,
    cell_area: Optional[float] = None,
    circularity: Optional[float] = None,
    peak_current: Optional[float] = None,
    rct_impedance: Optional[float] = None,
    notes: str = "",
    db_path: Optional[Path] = None
) -> int:
    """Insert a new prediction record into SQLite database."""
    init_db(db_path)
    conn = get_connection(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with conn:
        cursor = conn.execute("""
            INSERT INTO prediction_history (
                timestamp, sample_id, image_name, prediction, confidence,
                risk_score, cell_area, circularity, peak_current, rct_impedance, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str, sample_id, image_name, prediction, float(confidence),
            float(risk_score), cell_area, circularity, peak_current, rct_impedance, notes
        ))
        record_id = cursor.lastrowid
    conn.close()
    return int(record_id)


def fetch_all_predictions(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Fetch all prediction history as a pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", conn)
    conn.close()
    return df


def fetch_recent_predictions(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch recent prediction records as a list of dictionaries."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_summary_statistics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculate summary statistics for dashboard telemetry."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    total_count = cursor.fetchone()[0]

    if total_count == 0:
        conn.close()
        return {
            "total_scans": 0,
            "healthy_count": 0,
            "abnormal_count": 0,
            "abnormal_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_risk_score": 0.0,
        }

    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE prediction = 'Abnormal'")
    abnormal_count = cursor.fetchone()[0]
    healthy_count = total_count - abnormal_count

    cursor.execute("SELECT AVG(confidence), AVG(risk_score) FROM prediction_history")
    avg_conf, avg_risk = cursor.fetchone()
    conn.close()

    return {
        "total_scans": total_count,
        "healthy_count": healthy_count,
        "abnormal_count": abnormal_count,
        "abnormal_rate": round(float(abnormal_count / total_count) * 100.0, 1),
        "avg_confidence": round(float(avg_conf or 0.0), 2),
        "avg_risk_score": round(float(avg_risk or 0.0), 2),
    }


def clear_history(db_path: Optional[Path] = None) -> None:
    """Clear all records from prediction history."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM prediction_history")
    conn.close()
