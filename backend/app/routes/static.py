"""Serve static Ishihara test plates and other assets."""
from flask import Blueprint, send_from_directory
from pathlib import Path

static_blueprint = Blueprint("static", __name__)

STATIC_DIR = Path(__file__).parent.parent / "static"


@static_blueprint.route("/ishihara/<filename>")
def serve_ishihara(filename):
    """Serve Ishihara test plate images."""
    return send_from_directory(STATIC_DIR / "ishihara", filename)


@static_blueprint.route("/assets/<filename>")
def serve_assets(filename):
    """Serve other static assets."""
    return send_from_directory(STATIC_DIR, filename)
