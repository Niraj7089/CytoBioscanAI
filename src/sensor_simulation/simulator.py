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
    seed: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, np.ndarray]:
    """Simulate a Cyclic Voltammogram (CV) current response across a potential sweep.

    Args:
        cell_type: "Healthy" or "Abnormal".
        noise_level: Relative standard deviation of measurement noise.
        seed: Random seed for reproducibility.
        params: Optional real dataset sensor parameters (ipa, ipc, delta_ep).

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

    # Draw stochastic biophysical parameters or use verified real dataset records
    if params and "peak_anodic_current" in params:
        ipa = float(params["peak_anodic_current"])
        ipc = float(params.get("peak_cathodic_current", -abs(ipa) * 0.95))
        delta_ep_v = float(params.get("delta_peak_potential", 75.0)) / 1000.0
    else:
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
    seed: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
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

    if params and "charge_transfer_resistance" in params:
        rct = float(params["charge_transfer_resistance"])
        cdl = float(params.get("double_layer_capacitance_uf", 2.0)) * 1e-6
        rs = float(params.get("solution_resistance", config.EIS_R_SOL))
    else:
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
    seed: Optional[int] = None,
    mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Generate complete multi-technique electrochemical & scanning probe simulated measurements for a cell."""
    cv = simulate_cyclic_voltammetry(cell_type=cell_type, noise_level=noise_level, seed=seed)
    eis_seed = (seed + 9999) if seed is not None else None
    eis = simulate_impedance_spectroscopy(cell_type=cell_type, noise_level=noise_level, seed=eis_seed)
    secm_seed = (seed + 12345) if seed is not None else None
    secm_app = simulate_secm_approach_curve(cell_type=cell_type, noise_level=noise_level, seed=secm_seed)
    secm_2d = simulate_secm_2d_surface_scan(cell_type=cell_type, mask=mask, noise_level=noise_level, seed=secm_seed)
    dpv = simulate_differential_pulse_voltammetry(cell_type=cell_type, noise_level=noise_level, seed=seed)
    ca = simulate_chronoamperometry(cell_type=cell_type, noise_level=noise_level, seed=seed)

    return {
        "cell_type": cell_type,
        "cv": cv,
        "eis": eis,
        "secm_approach": secm_app,
        "secm_2d": secm_2d,
        "dpv": dpv,
        "ca": ca,
        "features": extract_sensor_features(cv, eis),
    }


def simulate_secm_approach_curve(
    cell_type: str = "Healthy",
    noise_level: float = 0.04,
    seed: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Simulate a Scanning Electrochemical Microscopy (SECM) microelectrode approach curve.

    Plots normalized current (I_T = I/I_inf) vs normalized tip-substrate distance (L = d/a),
    distinguishing intact dielectric cellular membrane (negative feedback) from
    compromised / permeabilized membrane (permeable / conductive feedback).

    Returns:
        Dictionary with:
            - 'normalized_distance_l': Array of L = d/a values
            - 'distance_um': Tip-to-cell physical distance (microns)
            - 'normalized_current': Normalized current I_T (I/I_inf)
            - 'current_na': Absolute probe current (nA)
            - 'feedback_mode': Description string of feedback regime
    """
    rng = np.random.RandomState(seed)
    a_um = config.SECM_PROBE_RADIUS_UM
    i_bulk_na = config.SECM_BULK_CURRENT_NA
    l_vals = np.linspace(config.SECM_L_START, config.SECM_L_END, config.SECM_APPROACH_POINTS)

    # Theoretical negative feedback (insulating boundary: intact cell membrane)
    # Paoletti empirical formula for inlaid microdisk
    i_neg = 1.0 / (0.15 + (1.53 / l_vals) + 0.58 * np.exp(-1.14 / l_vals) + 0.09 * np.exp(-0.14 / l_vals))

    # Theoretical positive feedback (conductive substrate)
    i_pos = 0.68 + (0.78377 / l_vals) + 0.3315 * np.exp(-1.0672 / l_vals)

    is_healthy = (cell_type == "Healthy")
    if params and "charge_transfer_resistance" in params:
        is_healthy = float(params["charge_transfer_resistance"]) >= 1000.0

    if is_healthy:
        alpha = 0.04  # High membrane resistance, almost pure negative feedback
        feedback_mode = "Negative Feedback (Intact Cell Membrane)"
    else:
        alpha = 0.62  # Compromised membrane, permeable / conductive feedback
        feedback_mode = "Permeable / Conductive Feedback (Compromised Membrane)"

    i_norm_ideal = (1.0 - alpha) * i_neg + alpha * i_pos
    noise = rng.normal(0, 0.015 * (1.0 + noise_level * 5.0), size=len(l_vals))
    i_norm = np.clip(i_norm_ideal + noise, 0.1, 2.5)
    i_abs = i_norm * i_bulk_na
    d_um = l_vals * a_um

    return {
        "normalized_distance_l": l_vals.tolist(),
        "distance_um": [round(float(x), 3) for x in d_um],
        "normalized_current": [round(float(x), 4) for x in i_norm],
        "current_na": [round(float(x), 3) for x in i_abs],
        "feedback_mode": feedback_mode,
        "probe_radius_um": a_um,
        "bulk_current_na": i_bulk_na,
        "min_normalized_current": round(float(np.min(i_norm)), 3),
    }


def simulate_secm_2d_surface_scan(
    cell_type: str = "Healthy",
    mask: Optional[np.ndarray] = None,
    noise_level: float = 0.04,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Simulate a 2D spatial raster scan of probe Faradaic current across the single cell.

    If a segmentation mask is provided, coregisters the localized current with the
    microscopy morphology.
    """
    import cv2
    rng = np.random.RandomState(seed)
    n_pts = config.SECM_GRID_POINTS
    scan_range = config.SECM_SCAN_RANGE_UM
    i_bulk = config.SECM_BULK_CURRENT_NA

    x_coords = np.linspace(-scan_range / 2.0, scan_range / 2.0, n_pts)
    y_coords = np.linspace(-scan_range / 2.0, scan_range / 2.0, n_pts)

    if mask is not None and mask.size > 0:
        # Resize mask to n_pts x n_pts
        mask_resized = cv2.resize(mask.astype(np.float32), (n_pts, n_pts), interpolation=cv2.INTER_AREA)
        cell_weights = np.clip(mask_resized / 255.0, 0.0, 1.0)
    else:
        # Construct idealized cell shape
        xx, yy = np.meshgrid(x_coords, y_coords)
        r_sq = (xx / 4.8) ** 2 + (yy / 4.8) ** 2
        cell_weights = np.clip(1.0 - np.clip(r_sq, 0.0, 1.0), 0.0, 1.0)

    # Localized probe current calculation
    if cell_type == "Healthy":
        # Intact membrane strongly blocks mediator diffusion: current drops to ~3.5 nA
        attenuation = 0.65
        grid_current = i_bulk * (1.0 - cell_weights * attenuation)
    else:
        # Abnormal membrane has increased permeability and localized reactive hotspots
        attenuation = 0.22
        heterogeneity = rng.uniform(-0.15, 0.25, size=(n_pts, n_pts)) * cell_weights
        grid_current = i_bulk * (1.0 - cell_weights * attenuation + heterogeneity)

    noise = rng.normal(0, i_bulk * noise_level * 0.4, size=(n_pts, n_pts))
    grid_current = np.clip(grid_current + noise, 1.0, 16.0)

    i_min = float(np.min(grid_current))
    i_max = float(np.max(grid_current))
    contrast_pct = round(((i_max - i_min) / i_max) * 100.0, 1)

    return {
        "x_coords_um": [round(float(x), 2) for x in x_coords],
        "y_coords_um": [round(float(y), 2) for y in y_coords],
        "current_matrix_na": [[round(float(v), 2) for v in row] for row in grid_current],
        "min_probe_current_na": round(i_min, 2),
        "max_probe_current_na": round(i_max, 2),
        "contrast_pct": contrast_pct,
        "scan_area_um2": f"{scan_range} x {scan_range}",
    }


def simulate_differential_pulse_voltammetry(
    cell_type: str = "Healthy",
    noise_level: float = 0.04,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Simulate Differential Pulse Voltammetry (DPV) / Square Wave Voltammetry (SWV).

    Produces high-sensitivity baseline-subtracted differential Faradaic current peaks.
    """
    rng = np.random.RandomState(seed)
    potentials = np.arange(config.DPV_E_START, config.DPV_E_END, config.DPV_STEP_MV / 1000.0)

    profile = config.ELECTROCHEMICAL_PROFILES.get(cell_type, config.ELECTROCHEMICAL_PROFILES["Healthy"])
    peak_current_base = profile.get("dpv_peak_current_ua", 10.0)
    delta_ep_v = float(rng.normal(profile["delta_ep_mean"], profile["delta_ep_std"])) / 1000.0
    e_peak = 0.20 + (delta_ep_v / 4.0)

    peak_current = float(rng.normal(peak_current_base, peak_current_base * 0.08))
    sigma = 0.085

    # Bell-shaped differential voltammetric peak
    delta_i = peak_current / (np.cosh((potentials - e_peak) / (2.0 * sigma)) ** 2)
    noise = rng.normal(0, max(0.08, peak_current * noise_level * 0.5), size=len(potentials))
    dpv_current = np.clip(delta_i + noise, 0.0, 40.0)

    return {
        "potential_v": [round(float(e), 3) for e in potentials],
        "delta_current_ua": [round(float(i), 3) for i in dpv_current],
        "peak_potential_v": round(float(e_peak), 3),
        "peak_current_ua": round(float(np.max(dpv_current)), 2),
    }


def simulate_chronoamperometry(
    cell_type: str = "Healthy",
    noise_level: float = 0.04,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Simulate Chronoamperometric current transient under potential step."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0.02, config.CA_DURATION_SEC, config.CA_POINTS)

    profile = config.ELECTROCHEMICAL_PROFILES.get(cell_type, config.ELECTROCHEMICAL_PROFILES["Healthy"])
    i_diff_coeff = profile["ipa_mean"] * 1.8

    # Cottrell diffusion decay: I(t) ~ t^(-0.5) + steady state + capacitive decay
    cottrell = i_diff_coeff / np.sqrt(t)
    cap_decay = 8.0 * np.exp(-t / 0.08)
    i_ss = profile["ipa_mean"] * 0.25

    noise = rng.normal(0, 0.15 * (1.0 + noise_level * 2.0), size=len(t))
    current_transient = cottrell + cap_decay + i_ss + noise

    return {
        "time_sec": [round(float(ti), 3) for ti in t],
        "current_ua": [round(float(ci), 3) for ci in current_transient],
        "initial_current_ua": round(float(current_transient[0]), 2),
        "steady_state_current_ua": round(float(np.mean(current_transient[-10:])), 2),
    }


def format_instrument_telemetry_csv(
    sample_id: str,
    cell_type: str,
    cv_data: Dict[str, Any],
    eis_data: Dict[str, Any],
    secm_approach: Dict[str, Any],
    dpv_data: Dict[str, Any],
    ca_data: Dict[str, Any],
) -> str:
    """Format full multi-technique laboratory telemetry into a structured CSV string."""
    from datetime import datetime
    now_str = datetime.now().isoformat()

    lines = [
        "# ====================================================================",
        "# CYTOBIOSCAN AI -- MULTIMODAL SINGLE-CELL ELECTROCHEMICAL WORKSTATION",
        "# LABORATORY INSTRUMENT TELEMETRY EXPORT",
        "# ====================================================================",
        f"# Sample ID: {sample_id}",
        f"# Membrane Condition: {cell_type}",
        f"# Acquisition Timestamp: {now_str}",
        f"# Working Electrode: Pt Microelectrode (radius = {config.SECM_PROBE_RADIUS_UM} um)",
        "# Reference Electrode: Ag/AgCl (3.0 M KCl)",
        "# Counter Electrode: Platinum Wire Auxiliary",
        "# Supporting Electrolyte: 10 mM PBS pH 7.4 + 5.0 mM [Fe(CN)6]3-/4-",
        "# Techniques: CV, EIS, SECM Approach, DPV, CA",
        "# ====================================================================",
        "",
        "# --- [SECTION 1: CYCLIC VOLTAMMETRY (CV)] ---",
        "# Potential_V,Current_uA",
    ]
    for p, c in zip(cv_data["potential_v"], cv_data["current_ua"]):
        lines.append(f"{p:.4f},{c:.4f}")

    lines.extend([
        "",
        "# --- [SECTION 2: ELECTROCHEMICAL IMPEDANCE SPECTROSCOPY (EIS)] ---",
        "# Frequency_Hz,Z_Real_Ohm,Z_Imag_Ohm,Z_Mag_Ohm,Phase_Deg",
    ])
    for f, zr, zi, zm, ph in zip(
        eis_data["frequency_hz"],
        eis_data["z_real_ohm"],
        eis_data["z_imag_ohm"],
        eis_data["z_mag_ohm"],
        eis_data["phase_deg"],
    ):
        lines.append(f"{f:.4e},{zr:.3f},{zi:.3f},{zm:.3f},{ph:.2f}")

    lines.extend([
        "",
        "# --- [SECTION 3: SECM APPROACH CURVE (L = d/a)] ---",
        "# Normalized_L,Distance_um,Normalized_Current_IT,Current_nA",
    ])
    for l, d, it, ina in zip(
        secm_approach["normalized_distance_l"],
        secm_approach["distance_um"],
        secm_approach["normalized_current"],
        secm_approach["current_na"],
    ):
        lines.append(f"{l:.3f},{d:.3f},{it:.4f},{ina:.3f}")

    lines.extend([
        "",
        "# --- [SECTION 4: DIFFERENTIAL PULSE VOLTAMMETRY (DPV)] ---",
        "# Potential_V,Delta_Current_uA",
    ])
    for p, di in zip(dpv_data["potential_v"], dpv_data["delta_current_ua"]):
        lines.append(f"{p:.4f},{di:.4f}")

    lines.extend([
        "",
        "# --- [SECTION 5: CHRONOAMPEROMETRY (CA)] ---",
        "# Time_s,Current_uA",
    ])
    for t, c in zip(ca_data["time_sec"], ca_data["current_ua"]):
        lines.append(f"{t:.4f},{c:.4f}")

    return "\n".join(lines)


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
