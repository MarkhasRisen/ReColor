"""Flask application factory for the adaptive daltonization API."""
from flask import Flask
from flask_cors import CORS

from .config import load_config
from .routes.calibration import calibration_blueprint
from .routes.processing import processing_blueprint
from .routes.static import static_blueprint
from .routes.health import health_blueprint
from .services.firebase import initialize_firebase
from .utils.logging_config import setup_logging

def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)
    app.config.from_mapping(load_config())
    
    # Setup logging
    setup_logging(app)
    
    # Enable CORS for all routes (adjust origins in production)
    CORS(app, resources={r"/*": {"origins": "*"}})

    initialize_firebase(app)

    app.register_blueprint(health_blueprint, url_prefix="/health")
    app.register_blueprint(calibration_blueprint, url_prefix="/calibration")
    app.register_blueprint(processing_blueprint, url_prefix="/process")
    app.register_blueprint(static_blueprint, url_prefix="/static")

    return app
