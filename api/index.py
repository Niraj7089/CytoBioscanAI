"""Vercel Serverless Function entry point for CytoBioScan AI.

Exposes the WSGI Flask application instance for Vercel Serverless Functions.
"""

import os
import sys
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode

# Add repository root to sys.path so app, config, and src packages resolve
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app


class VercelRouteMiddleware:
    """Normalize PATH_INFO and QUERY_STRING when Vercel forwards requests via __path."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query_string = environ.get("QUERY_STRING", "")
        params = parse_qs(query_string)

        if "__path" in params:
            raw_path = params["__path"][0] if params["__path"] else ""
            clean_path = "/" + raw_path.lstrip("/").split("?")[0]
            environ["PATH_INFO"] = re.sub(r"/+", "/", clean_path)

            # Strip __path from query string so request.args is pristine
            clean_params = {k: v for k, v in params.items() if k != "__path"}
            environ["QUERY_STRING"] = urlencode(clean_params, doseq=True)
        else:
            path = environ.get("PATH_INFO", "")
            if path in ("/api/index", "/api/index.py", "/api", "/api/"):
                environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


# Wrap Flask application with route middleware
app.wsgi_app = VercelRouteMiddleware(app.wsgi_app)
