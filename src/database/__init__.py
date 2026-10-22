"""Database storage and audit log module using SQLite."""

from .db_manager import (
    init_db,
    save_prediction,
    fetch_all_predictions,
    fetch_recent_predictions,
    get_summary_statistics,
    clear_history,
)

__all__ = [
    "init_db",
    "save_prediction",
    "fetch_all_predictions",
    "fetch_recent_predictions",
    "get_summary_statistics",
    "clear_history",
]
