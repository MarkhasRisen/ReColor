"""Health check and system status endpoints."""
from flask import Blueprint, jsonify
import sys
from datetime import datetime

health_blueprint = Blueprint("health", __name__)


@health_blueprint.route("/")
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "service": "Daltonization API"
    }), 200


@health_blueprint.route("/ready")
def readiness_check():
    """Readiness check - verify all dependencies are available."""
    checks = {
        "firebase": False,
        "models": False,
    }
    
    # Check Firebase
    try:
        from ..services import firebase
        firebase.get_firestore_client()
        checks["firebase"] = True
    except:
        pass
    
    # Check models directory
    try:
        from pathlib import Path
        models_dir = Path(__file__).parent.parent.parent / "models" / "centroids"
        checks["models"] = models_dir.exists() and any(models_dir.glob("*.npy"))
    except:
        pass
    
    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    
    return jsonify({
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }), status_code
