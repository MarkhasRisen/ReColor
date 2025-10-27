"""Executable entrypoint for running the Flask development server."""
from . import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
