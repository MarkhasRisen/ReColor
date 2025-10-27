"""Calibration endpoints for deriving user vision profiles."""
from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from ..middleware.auth import optional_auth
from ..pipeline.profile import VisionProfile
from ..schemas.calibration import CalibrationRequest, CalibrationResponse
from ..services import firebase
from ..utils.responses import parse_request

calibration_blueprint = Blueprint("calibration", __name__)


@calibration_blueprint.post("/")
@optional_auth
def handle_calibration():
    payload = parse_request(request)
    schema = CalibrationRequest(**payload)
    profile: VisionProfile = schema.to_profile()

    # Use authenticated user ID if available, otherwise use provided user_id
    user_id = getattr(request, 'user_id', None) or schema.user_id
    
    metadata = {
        "calibratedAt": datetime.now(timezone.utc).isoformat(),
        "source": "ishihara",  # `ishihara module` identifier
        "authenticated": hasattr(request, 'user_id'),
    }

    try:
        firebase.save_profile(user_id, profile, metadata)
        current_app.logger.info(f"Profile saved for user: {user_id}")
    except RuntimeError as exc:
        current_app.logger.warning("Skipping Firebase persistence: %s", exc)

    response = CalibrationResponse(
        deficiency=profile.deficiency,
        severity=profile.severity,
        confidence=profile.confidence,
    )
    return jsonify(response.dict()), 201
