"""Processing endpoints for the adaptive color correction pipeline."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from PIL import Image

from ..middleware.auth import optional_auth
from ..pipeline.processing import AdaptiveColorPipeline, PipelineConfig
from ..pipeline.profile import VisionProfile
from ..schemas.processing import ProcessRequest, ProcessResponse
from ..services import firebase
from ..utils.responses import parse_request

processing_blueprint = Blueprint("processing", __name__)


@processing_blueprint.post("/")
@optional_auth
def process_image():
    payload = parse_request(request)
    schema = ProcessRequest(**payload)

    # Use authenticated user ID if available, otherwise use provided user_id
    user_id = getattr(request, 'user_id', None) or schema.user_id
    
    profile = _load_profile(user_id) or VisionProfile(
        deficiency="normal", severity=0.0, confidence=0.0
    )
    
    current_app.logger.info(f"Processing image for user: {user_id} (profile: {profile.deficiency})")

    frame = _decode_image(schema.image_base64)
    pipeline = AdaptiveColorPipeline(
        config=PipelineConfig(),
        model_dir=Path(current_app.config["TFLITE_MODEL_DIR"]),
        profile=profile,
    )
    corrected = pipeline.run(frame)

    response = ProcessResponse(
        content_type="image/png",
        data=_encode_image(corrected),
    )
    return jsonify(response.dict()), 200


def _load_profile(user_id: str) -> VisionProfile | None:
    try:
        return firebase.load_profile(user_id)
    except RuntimeError as exc:
        current_app.logger.warning("Profile lookup skipped: %s", exc)
        return None


def _decode_image(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)
    with Image.open(BytesIO(raw)) as img:
        return np.array(img.convert("RGB"))


def _encode_image(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
