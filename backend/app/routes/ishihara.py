"""
Ishihara test routes for color vision screening.
Supports both comprehensive (38 plates) and quick (14 plates) tests.
"""
from flask import Blueprint, request, jsonify, current_app
from firebase_admin import firestore
from ..middleware.auth import optional_auth, require_auth
from ..ishihara.test import IshiharaTest, CVDType
from ..services import firebase
from ..pipeline.profile import VisionProfile

bp = Blueprint("ishihara", __name__, url_prefix="/ishihara")


@bp.route("/plates", methods=["GET"])
@optional_auth  # Kept optional - viewing plates can be open
def get_plates():
    """
    Get list of Ishihara test plates.
    
    Query params:
        mode: 'quick' (14 plates) or 'comprehensive' (38 plates). Default: 'quick'
    
    Returns:
        {
            "mode": "quick" | "comprehensive",
            "total_plates": 14 | 38,
            "plates": [
                {
                    "plate_number": 1,
                    "image_url": "/static/ishihara/plate_01.png",
                    "is_control": true
                },
                ...
            ]
        }
    """
    mode = request.args.get("mode", "quick").lower()
    use_comprehensive = mode == "comprehensive"
    
    test = IshiharaTest(use_comprehensive=use_comprehensive)
    
    plates_info = [
        {
            "plate_number": plate.plate_number,
            "image_url": f"/static/ishihara/plate_{plate.plate_number:02d}.png",
            "is_control": plate.is_control,
            "description": plate.description if current_app.debug else None  # Only in debug
        }
        for plate in test.plates
    ]
    
    return jsonify({
        "mode": mode,
        "total_plates": len(test.plates),
        "plates": plates_info
    }), 200


@bp.route("/evaluate", methods=["POST"])
@optional_auth
def evaluate_test():
    """
    Evaluate Ishihara test responses and return diagnosis.
    
    Request body:
        {
            "user_id": "optional_user_id",
            "mode": "quick" | "comprehensive",
            "responses": {
                "1": "12",
                "3": "6",
                "4": "29",
                ...
            },
            "save_profile": true  // Optional, default false
        }
    
    Returns:
        {
            "result": {
                "total_plates": 14,
                "correct_normal": 12,
                "correct_protan": 2,
                "correct_deutan": 1,
                "incorrect": 0,
                "control_failed": 0,
                "classification_score": {
                    "protan": 2,
                    "deutan": 0
                }
            },
            "diagnosis": {
                "cvd_type": "protan" | "deutan" | "normal" | "tritan" | "total",
                "severity": 0.6,
                "confidence": 0.85,
                "interpretation": "Moderate Protanomaly detected..."
            },
            "profile_saved": false
        }
    """
    data = request.json
    
    if not data or "responses" not in data:
        return jsonify({"error": "Missing responses data"}), 400
    
    mode = data.get("mode", "quick").lower()
    use_comprehensive = mode == "comprehensive"
    
    # Convert response keys to integers
    responses = {}
    for key, value in data["responses"].items():
        try:
            plate_num = int(key)
            responses[plate_num] = value
        except (ValueError, TypeError):
            continue
    
    # Evaluate test
    test = IshiharaTest(use_comprehensive=use_comprehensive)
    result = test.evaluate_test(responses)
    
    # Prepare response
    response_data = {
        "result": {
            "total_plates": result.total_plates,
            "correct_normal": result.correct_normal,
            "correct_protan": result.correct_protan,
            "correct_deutan": result.correct_deutan,
            "incorrect": result.incorrect,
            "control_failed": result.control_failed,
            "classification_score": result.classification_score,
        },
        "diagnosis": {
            "cvd_type": result.cvd_type.value,
            "severity": result.severity,
            "confidence": result.confidence,
            "interpretation": result.interpretation,
        },
        "profile_saved": False
    }
    
    # Save profile if requested
    user_id = data.get("user_id")
    save_profile = data.get("save_profile", False)
    
    if user_id and save_profile and result.control_failed == 0:
        try:
            profile = VisionProfile(
                deficiency=result.cvd_type.value,
                severity=result.severity,
                confidence=result.confidence
            )
            
            metadata = {
                "test_type": "ishihara",
                "test_mode": mode,
                "total_plates": result.total_plates,
                "correct_normal": result.correct_normal,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
            
            firebase.save_profile(user_id, profile, metadata)
            response_data["profile_saved"] = True
            
            current_app.logger.info(
                f"Ishihara profile saved for user {user_id}: "
                f"{result.cvd_type.value} (severity: {result.severity:.2f})"
            )
        except Exception as e:
            current_app.logger.error(f"Failed to save Ishihara profile: {e}")
            # Don't fail the request if profile save fails
    
    return jsonify(response_data), 200


@bp.route("/info", methods=["GET"])
def get_test_info():
    """
    Get information about the Ishihara test.
    
    Returns:
        {
            "name": "Ishihara Color Vision Test",
            "modes": {
                "quick": {
                    "plates": 14,
                    "duration": "2-3 minutes",
                    "description": "..."
                },
                "comprehensive": {
                    "plates": 38,
                    "duration": "5-7 minutes",
                    "description": "..."
                }
            },
            "instructions": [...],
            "clinical_standards": {...}
        }
    """
    return jsonify({
        "name": "Ishihara Color Vision Test",
        "version": "DaltonLens compatible implementation",
        "license": "BSD-2-Clause",
        "modes": {
            "quick": {
                "plates": 14,
                "duration": "2-3 minutes",
                "description": "Standard screening test with most diagnostic plates. Suitable for routine screening.",
                "accuracy": "High sensitivity for detecting red-green CVD"
            },
            "comprehensive": {
                "plates": 38,
                "duration": "5-7 minutes",
                "description": "Complete test with all plates including tracing plates. Provides detailed classification.",
                "accuracy": "Clinical-grade diagnosis with protan/deutan classification"
            }
        },
        "instructions": [
            "Ensure proper lighting (natural daylight or daylight-equivalent LED recommended)",
            "View each plate for 3-5 seconds maximum",
            "State what number or pattern you see, or 'nothing' if unclear",
            "Do not guess - respond with your immediate perception",
            "If you wear corrective lenses, keep them on during the test"
        ],
        "clinical_standards": {
            "normal_threshold": "≥12/14 correct on quick test (86%), ≥33/38 on comprehensive (87%)",
            "mild_cvd": "8-11/14 or 22-32/38 correct",
            "moderate_cvd": "4-7/14 or 11-21/38 correct",
            "strong_cvd": "<4/14 or <11/38 correct",
            "control_plates": "Must be answered correctly for valid test"
        },
        "cvd_types": {
            "protan": "Protanomaly/Protanopia - Red color deficiency (1% of males)",
            "deutan": "Deuteranomaly/Deuteranopia - Green color deficiency (5% of males)",
            "normal": "Normal color vision - No deficiency detected"
        },
        "disclaimer": "This is a screening tool. Clinical diagnosis should be confirmed by an optometrist or ophthalmologist."
    }), 200
