"""Configuration settings for AI-Assisted Single-Cell Diagnosis Research Prototype.

This module consolidates all file paths, model hyperparameters, image processing
parameters, and electrochemical simulation constants.
"""

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
REPORTS_DIR = BASE_DIR / "reports"

# Sub-dataset paths
TRAIN_DIR = RAW_DATA_DIR / "train"
VALID_DIR = RAW_DATA_DIR / "valid"
TEST_DIR = RAW_DATA_DIR / "test"

# Database and Model Artifacts
DB_PATH = DATA_DIR / "predictions.db"
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

# Biophysical / Electrochemical Profiles by Class
# Intact healthy cell membrane: higher charge transfer resistance (Rct), lower capacitance, lower peak Faradaic current
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
    },
    # Compromised / abnormal cell membrane: membrane permeabilization causes lower Rct, higher capacitance, higher redox current
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
    }
}

# Scientific Disclaimer
DISCLAIMER_TEXT = (
    "RESEARCH PROTOTYPE NOTICE: This system is developed strictly for research, educational, "
    "and computational evaluation purposes. It is not an FDA-cleared medical device and must not "
    "be used as a clinical diagnostic tool or to guide medical treatments."
)
