"""Flask application factory for the adaptive daltonization API."""
import os
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .config import load_config
from .routes.calibration import calibration_blueprint
from .routes.processing import processing_blueprint
from .routes.static import static_blueprint
from .routes.health import health_blueprint
from .routes.notifications import bp as notifications_blueprint
from .routes.ishihara import bp as ishihara_blueprint
from .services.firebase import initialize_firebase
from .utils.logging_config import setup_logging

# Initialize rate limiter (will be attached to app in create_app)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)
    app.config.from_mapping(load_config())
    
    # Setup logging
    setup_logging(app)
    
    # Configure CORS based on environment
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
    if allowed_origins == "*":
        app.logger.warning("CORS configured for ALL origins (*). Set ALLOWED_ORIGINS env var for production!")
    
    CORS(app, resources={
        r"/*": {
            "origins": allowed_origins.split(",") if allowed_origins != "*" else "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    })
    
    # Initialize rate limiter
    limiter.init_app(app)

    initialize_firebase(app)

    app.register_blueprint(health_blueprint, url_prefix="/health")
    app.register_blueprint(calibration_blueprint, url_prefix="/calibration")
    app.register_blueprint(processing_blueprint, url_prefix="/process")
    app.register_blueprint(static_blueprint, url_prefix="/static")
    app.register_blueprint(notifications_blueprint)
    app.register_blueprint(ishihara_blueprint)

    return app
