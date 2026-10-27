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


class VercelPathFixMiddleware:
    """Normalize PATH_INFO when Vercel rewrites incoming requests to /api/index."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path in ("/api/index", "/api/index/", "/api", "/api/"):
            environ["PATH_INFO"] = "/"
        elif path.startswith("/api/index/"):
            environ["PATH_INFO"] = path[len("/api/index"):]
        return self.wsgi_app(environ, start_response)


# Wrap Flask's WSGI application with path normalization for Vercel
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
