"""Unit tests for the electrochemical sensor simulation module."""

import numpy as np
import pytest
from src.sensor_simulation.simulator import (
    simulate_cyclic_voltammetry,
    simulate_impedance_spectroscopy,
    simulate_full_cell_signal,
    extract_sensor_features,
    get_sensor_feature_names,
)


def test_simulate_cyclic_voltammetry():
    cv_healthy = simulate_cyclic_voltammetry(cell_type="Healthy", seed=42)
    cv_abnormal = simulate_cyclic_voltammetry(cell_type="Abnormal", seed=42)

    assert "potential_v" in cv_healthy
    assert "current_ua" in cv_healthy
    assert len(cv_healthy["potential_v"]) == len(cv_healthy["current_ua"])

    # Abnormal cells have higher peak anodic current due to permeabilized membrane
    assert cv_abnormal["ipa"] > cv_healthy["ipa"]


def test_simulate_impedance_spectroscopy():
    eis_healthy = simulate_impedance_spectroscopy(cell_type="Healthy", seed=42)
    eis_abnormal = simulate_impedance_spectroscopy(cell_type="Abnormal", seed=42)

    assert len(eis_healthy["frequency_hz"]) == 50
    assert len(eis_healthy["z_real_ohm"]) == 50

    # Healthy cells maintain higher intact charge transfer resistance Rct
    assert eis_healthy["rct"] > eis_abnormal["rct"]


def test_extract_sensor_features():
    signal = simulate_full_cell_signal(cell_type="Healthy", seed=10)
    features = signal["features"]
    required_names = get_sensor_feature_names()

    for name in required_names:
        assert name in features
        assert isinstance(features[name], (int, float))
