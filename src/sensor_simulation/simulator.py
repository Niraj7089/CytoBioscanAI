"""Simulation of realistic electrochemical signals (Cyclic Voltammetry and Impedance Spectroscopy).

Simulates biophysical responses for healthy vs. abnormal single cells based on
membrane permeability, dielectric impedance, and Faradaic redox kinetics.
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd

import config


def simulate_cyclic_voltammetry(
    cell_type: str = "Healthy",
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Simulate a Cyclic Voltammogram (CV) current response across a potential sweep.

    Args:
        cell_type: "Healthy" or "Abnormal".
        noise_level: Relative standard deviation of measurement noise.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            - 'potential_v': Array of potentials (V)
            - 'current_ua': Array of measured currents (microAmps)
            - 'ipa': Anodic peak current
            - 'ipc': Cathodic peak current
            - 'delta_ep': Peak potential separation (mV)
    """
    rng = np.random.RandomState(seed)
    profile = config.ELECTROCHEMICAL_PROFILES.get(cell_type, config.ELECTROCHEMICAL_PROFILES["Healthy"])

    # Draw stochastic biophysical parameters
    ipa = float(rng.normal(profile["ipa_mean"], profile["ipa_std"]))
    ipc = float(rng.normal(profile["ipc_mean"], profile["ipc_std"]))
    delta_ep_v = float(rng.normal(profile["delta_ep_mean"], profile["delta_ep_std"])) / 1000.0

    # Potential waveform: forward -0.4V -> +0.8V, reverse +0.8V -> -0.4V
    half_points = config.CV_POINTS // 2
    e_forward = np.linspace(config.CV_E_START, config.CV_E_VERTEX, half_points)
    e_reverse = np.linspace(config.CV_E_VERTEX, config.CV_E_END, half_points)
    potential = np.concatenate([e_forward, e_reverse])

    # Redox peaks centered around formal potential E0 = +0.20 V
    e0 = 0.20
    e_pa = e0 + (delta_ep_v / 2.0)
    e_pc = e0 - (delta_ep_v / 2.0)

    # Gaussian diffusion-limited peak approximations
    sigma = 0.12
    forward_peak = ipa * np.exp(-((e_forward - e_pa) ** 2) / (2 * (sigma ** 2)))
    reverse_peak = ipc * np.exp(-((e_reverse - e_pc) ** 2) / (2 * (sigma ** 2)))
    peaks = np.concatenate([forward_peak, reverse_peak])

    # Baseline capacitive current (scan rate dependent, positive on forward, negative on reverse)
    cap_baseline = np.concatenate([
        np.full(half_points, 1.2),
        np.full(half_points, -1.2)
    ])

    # Add Gaussian noise
    noise = rng.normal(0, max(0.1, abs(ipa) * noise_level), size=config.CV_POINTS)
    current = peaks + cap_baseline + noise

    return {
        "potential_v": potential,
        "current_ua": current,
        "ipa": round(float(np.max(current)), 2),
        "ipc": round(float(np.min(current)), 2),
        "delta_ep": round(delta_ep_v * 1000.0, 2),
    }


def simulate_impedance_spectroscopy(
    cell_type: str = "Healthy",
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Simulate Electrochemical Impedance Spectroscopy (EIS) across a frequency spectrum.

    Uses a Randles equivalent circuit model:
        Rs + [ Rct // Cdl ]
    Z(w) = Rs + Rct / (1 + j * w * Rct * Cdl)

    Returns:
        Dictionary with keys:
            - 'frequency_hz': Frequency array (Hz)
            - 'z_real_ohm': Real impedance component Z' (Ohms)
            - 'z_imag_ohm': Imaginary impedance component -Z'' (Ohms)
            - 'z_mag_ohm': Magnitude |Z| (Ohms)
            - 'phase_deg': Phase angle theta (degrees)
            - 'rct': Charge transfer resistance (Ohms)
            - 'cdl': Double-layer capacitance (Farads)
    """
    rng = np.random.RandomState(seed)
    profile = config.ELECTROCHEMICAL_PROFILES.get(cell_type, config.ELECTROCHEMICAL_PROFILES["Healthy"])

    rct = float(rng.normal(profile["rct_mean"], profile["rct_std"]))
    cdl = float(rng.normal(profile["cdl_mean"], profile["cdl_std"]))
    rs = config.EIS_R_SOL + float(rng.normal(0, 2.0))

    freqs = np.logspace(
        np.log10(config.EIS_FREQ_MIN),
        np.log10(config.EIS_FREQ_MAX),
        config.EIS_NUM_POINTS
    )
    omega = 2.0 * np.pi * freqs

    # Randles impedance calculation
    # Z_parallel = Rct / (1 + j * omega * Rct * Cdl)
    denom = 1.0 + (omega * rct * cdl) ** 2
    z_real = rs + (rct / denom)
    z_imag = -(omega * (rct ** 2) * cdl) / denom

    # Add realistic stochastic noise
    noise_r = rng.normal(0, np.abs(z_real) * noise_level)
    noise_i = rng.normal(0, np.abs(z_imag) * noise_level)

    z_real_noisy = z_real + noise_r
    z_imag_noisy = z_imag + noise_i

    z_mag = np.sqrt(z_real_noisy ** 2 + z_imag_noisy ** 2)
    phase_deg = np.degrees(np.arctan2(z_imag_noisy, z_real_noisy))

    return {
        "frequency_hz": freqs,
        "z_real_ohm": z_real_noisy,
        "z_imag_ohm": -z_imag_noisy,  # Standard Nyquist display: -Z''
        "z_mag_ohm": z_mag,
        "phase_deg": phase_deg,
        "rct": round(rct, 2),
        "cdl": round(cdl, 8),
        "rs": round(rs, 2),
    }


def simulate_full_cell_signal(
    cell_type: str = "Healthy",
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate both Cyclic Voltammetry and EIS simulated measurements for a cell."""
    cv = simulate_cyclic_voltammetry(cell_type=cell_type, noise_level=noise_level, seed=seed)
    eis_seed = (seed + 9999) if seed is not None else None
    eis = simulate_impedance_spectroscopy(cell_type=cell_type, noise_level=noise_level, seed=eis_seed)

    return {
        "cell_type": cell_type,
        "cv": cv,
        "eis": eis,
        "features": extract_sensor_features(cv, eis),
    }


def extract_sensor_features(cv_data: Dict[str, Any], eis_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract key electrochemical scalar features from CV and EIS curves."""
    currents = cv_data["current_ua"]
    ipa = float(np.max(currents))
    ipc = float(np.min(currents))
    mean_curr = float(np.mean(currents))
    std_curr = float(np.std(currents))

    rct = float(eis_data["rct"])
    cdl = float(eis_data["cdl"]) * 1e6  # Convert to microFarads for numerical stability
    rs = float(eis_data.get("rs", config.EIS_R_SOL))

    # Low frequency impedance (|Z| at ~1 Hz) and high frequency impedance (|Z| at ~100 kHz)
    z_mag = eis_data["z_mag_ohm"]
    z_low_freq = float(z_mag[0])
    z_high_freq = float(z_mag[-1])

    # Peak phase angle
    phase = eis_data["phase_deg"]
    min_phase = float(np.min(phase))

    return {
        "peak_anodic_current": round(ipa, 2),
        "peak_cathodic_current": round(ipc, 2),
        "delta_peak_potential": round(float(cv_data.get("delta_ep", 80.0)), 2),
        "mean_current": round(mean_curr, 2),
        "std_current": round(std_curr, 2),
        "charge_transfer_resistance": round(rct, 2),
        "double_layer_capacitance_uf": round(cdl, 4),
        "solution_resistance": round(rs, 2),
        "impedance_low_freq": round(z_low_freq, 2),
        "impedance_high_freq": round(z_high_freq, 2),
        "min_phase_angle": round(min_phase, 2),
    }


def get_sensor_feature_names() -> List[str]:
    """Return ordered list of sensor feature keys."""
    return [
        "peak_anodic_current",
        "peak_cathodic_current",
        "delta_peak_potential",
        "mean_current",
        "std_current",
        "charge_transfer_resistance",
        "double_layer_capacitance_uf",
        "solution_resistance",
        "impedance_low_freq",
        "impedance_high_freq",
        "min_phase_angle",
    ]


def generate_sensor_dataset(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Generate and persist paired simulated electrochemical data for all raw cell images."""
    if output_path is None:
        output_path = config.SENSOR_CSV_PATH

    records = []
    # Iterate through train, valid, test sets
    splits = ["train", "valid", "test"]
    cell_idx = 0

    for split in splits:
        split_dir = config.RAW_DATA_DIR / split
        for class_name in config.CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in sorted(class_dir.glob("*.png")):
                cell_id = img_path.stem
                signal = simulate_full_cell_signal(
                    cell_type=class_name,
                    noise_level=0.04,
                    seed=cell_idx + 100
                )
                feat = signal["features"]
                record = {
                    "cell_id": cell_id,
                    "split": split,
                    "label": class_name,
                    "image_rel_path": str(img_path.relative_to(config.BASE_DIR)),
                    **feat
                }
                records.append(record)
                cell_idx += 1

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated sensor dataset with {len(df)} records at {output_path}")
    return df
