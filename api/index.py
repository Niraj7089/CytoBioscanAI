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


class VercelPathFixMiddleware:
    """Normalize PATH_INFO when Vercel rewrites incoming requests to /api/index."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # 1. Check for explicit path parameter injected by Vercel rewrite (__vpath)
        query_string = environ.get("QUERY_STRING", "")
        params = parse_qs(query_string)

        target_path = None
        if "__vpath" in params and params["__vpath"]:
            target_path = params["__vpath"][0]
            clean_params = {k: v for k, v in params.items() if k != "__vpath"}
            environ["QUERY_STRING"] = urlencode(clean_params, doseq=True)

        # 2. Check Vercel edge routing headers as fallback
        if not target_path or target_path in ("/api/index", "/api/index.py"):
            target_path = (
                environ.get("HTTP_X_MATCHED_PATH")
                or environ.get("HTTP_X_FORWARDED_PATH")
                or (environ.get("REQUEST_URI", "").split("?")[0] if environ.get("REQUEST_URI") else None)
                or (environ.get("RAW_URI", "").split("?")[0] if environ.get("RAW_URI") else None)
            )

        # 3. Clean and assign the resolved target path to PATH_INFO
        if target_path and target_path not in ("/api/index", "/api/index.py", ""):
            if not target_path.startswith("/"):
                target_path = "/" + target_path
            environ["PATH_INFO"] = re.sub(r"/+", "/", target_path)
        else:
            orig = environ.get("PATH_INFO", "")
            if orig in ("/api/index", "/api/index.py", "/api", "/api/"):
                environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


# Wrap Flask's WSGI application with path normalization for Vercel
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
