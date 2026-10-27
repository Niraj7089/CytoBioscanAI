"""Vercel Serverless Function entry point for CytoBioScan AI.

Exposes the WSGI Flask application instance for Vercel Serverless Functions.
"""

import os
import sys
from pathlib import Path

# Add repository root to sys.path so app, config, and src packages resolve
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app
