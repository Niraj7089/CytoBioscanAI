"""Configuration settings for AI-Assisted Single-Cell Diagnosis Research Prototype.

This module consolidates all file paths, model hyperparameters, image processing
parameters, and electrochemical simulation constants.
"""

import os
import tempfile
from pathlib import Path

# Scientific & Instrumentation Metadata
PROJECT_TITLE = "AI-Assisted Single-Cell Imaging and Electrochemical Detection System"
HARDWARE_PLATFORM = "High-Precision 3-Electrode Potentiostat & Impedance Spectrometer"
DATA_ACQUISITION_INTERFACE = "16-bit Differential Low-Noise DAQ (400 kHz Aggregate Throughput)"
OPTICAL_SYNC_PLATFORM = "Synchronized High-Resolution Optical Microscopy"
RESEARCH_SCOPE_NOTE = "High-resolution single-cell optical imaging and multimodal AI-driven electrochemical detection."

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SENSOR_DATA_DIR = DATA_DIR / "sensor"
MODELS_DIR = BASE_DIR / "models"
STATIC_REPORTS_DIR = BASE_DIR / "reports"

# Detect serverless environment (Vercel, AWS Lambda) where filesystem is read-only except /tmp
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
TMP_DIR = Path(tempfile.gettempdir())

if IS_SERVERLESS:
    DB_PATH = TMP_DIR / "predictions.db"
    REPORTS_DIR = TMP_DIR / "reports"
else:
    DB_PATH = DATA_DIR / "predictions.db"
    REPORTS_DIR = STATIC_REPORTS_DIR

# Sub-dataset paths
TRAIN_DIR = RAW_DATA_DIR / "train"
VALID_DIR = RAW_DATA_DIR / "valid"
TEST_DIR = RAW_DATA_DIR / "test"

# Database and Model Artifacts
IMAGE_MODEL_PATH = MODELS_DIR / "image_model.keras"
FUSION_MODEL_PATH = MODELS_DIR / "fusion_model.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
SENSOR_CSV_PATH = SENSOR_DATA_DIR / "sensor_data.csv"

# Classes and Labels
CLASS_NAMES = ["Healthy", "Abnormal"]
LABEL_MAP = {"Healthy": 0, "Abnormal": 1}
INV_LABEL_MAP = {0: "Healthy", 1: "Abnormal"}

# Image Processing Settings
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
CHANNELS = 3
COLOR_MODE = "rgb"

# Model Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Electrochemical Simulation Constants
# Cyclic Voltammetry (CV) Parameters
CV_E_START = -0.4      # Start potential (V)
CV_E_VERTEX = 0.8      # Vertex/turnaround potential (V)
CV_E_END = -0.4        # End potential (V)
CV_SCAN_RATE = 0.1     # Scan rate (V/s)
CV_POINTS = 200        # Number of sweep points

# Electrochemical Impedance Spectroscopy (EIS) Parameters
EIS_FREQ_MIN = 1.0       # Minimum frequency (Hz)
EIS_FREQ_MAX = 100000.0  # Maximum frequency (Hz)
EIS_NUM_POINTS = 50      # Number of logarithmically spaced frequency points
EIS_R_SOL = 50.0         # Solution resistance Rs (Ohms)

# Scanning Electrochemical Microscopy (SECM / e-SPM) Parameters
SECM_PROBE_RADIUS_UM = 1.0       # Ultramicroelectrode (UME) radius a (microns)
SECM_SCAN_RANGE_UM = 20.0        # Spatial scan width & height (microns)
SECM_GRID_POINTS = 25            # Raster resolution (25x25 spatial points)
SECM_BULK_CURRENT_NA = 10.0      # Asymptotic bulk steady-state current I_infinity (nA)
SECM_L_START = 4.0               # Normalized approach distance L = d/a start
SECM_L_END = 0.25                # Normalized approach distance L = d/a end
SECM_APPROACH_POINTS = 40        # Number of approach curve sample points

# Differential Pulse Voltammetry (DPV) Parameters
DPV_PULSE_HEIGHT_MV = 50.0       # Pulse modulation amplitude (mV)
DPV_STEP_MV = 4.0                # Potential increment step (mV)
DPV_E_START = -0.3               # Start potential (V)
DPV_E_END = 0.7                  # End potential (V)

# Chronoamperometry (CA) Parameters
CA_DURATION_SEC = 5.0            # Duration (seconds)
CA_POINTS = 100                  # Number of transient points
CA_E_STEP = 0.4                  # Step potential (V)

# Biophysical / Electrochemical Profiles by Class
# Intact healthy cell membrane: higher charge transfer resistance (Rct), lower capacitance, lower peak Faradaic current,
# negative feedback SECM approach (hindered diffusion), lower DPV peak.
ELECTROCHEMICAL_PROFILES = {
    "Healthy": {
        "rct_mean": 1500.0,
        "rct_std": 120.0,
        "cdl_mean": 2.0e-6,
        "cdl_std": 0.2e-6,
        "ipa_mean": 12.0,     # microAmperes
        "ipa_std": 1.2,
        "ipc_mean": -11.5,
        "ipc_std": 1.1,
        "delta_ep_mean": 75.0, # mV
        "delta_ep_std": 6.0,
        "secm_feedback": "Negative Feedback (Intact Dielectric Membrane)",
        "secm_min_current_na": 3.4,
        "dpv_peak_current_ua": 8.5,
    },
    # Compromised / abnormal cell membrane: membrane permeabilization causes lower Rct, higher capacitance, higher redox current,
    # intermediate/permeable SECM approach (increased mediator flux across compromised membrane), elevated DPV peak.
    "Abnormal": {
        "rct_mean": 620.0,
        "rct_std": 70.0,
        "cdl_mean": 4.8e-6,
        "cdl_std": 0.4e-6,
        "ipa_mean": 24.5,     # microAmperes
        "ipa_std": 2.0,
        "ipc_mean": -23.8,
        "ipc_std": 1.9,
        "delta_ep_mean": 115.0,# mV
        "delta_ep_std": 10.0,
        "secm_feedback": "Permeable / Conductive Feedback (Membrane Disruption)",
        "secm_min_current_na": 8.2,
        "dpv_peak_current_ua": 18.2,
    }
}

# Scientific Metadata
DISCLAIMER_TEXT = ""
