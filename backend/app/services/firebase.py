"""Firebase service integration for profile storage and analytics."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from firebase_admin import credentials, firestore, initialize_app
from flask import Flask

from ..pipeline.profile import VisionProfile

_firebase_app = None


def initialize_firebase(app: Flask) -> None:
    """Initialize Firebase Admin if credentials are configured."""
    global _firebase_app
    if _firebase_app is not None:
        app.logger.info("Firebase already initialized, skipping")
        return

    credential_path = app.config.get("FIREBASE_CREDENTIAL_PATH")
    if not credential_path:
        app.logger.warning("Firebase credential path not set; skipping initialization")
        return

    try:
        # Check if there's already a default app (from other scripts)
        from firebase_admin import _apps
        if _apps:
            app.logger.info("Using existing Firebase app")
            _firebase_app = list(_apps.values())[0]
            return
            
        cred = credentials.Certificate(credential_path)
        _firebase_app = initialize_app(cred, {
            "projectId": app.config.get("FIREBASE_PROJECT_ID"),
        })
        app.logger.info(f"Firebase initialized successfully with project: {app.config.get('FIREBASE_PROJECT_ID')}")
    except Exception as e:
        app.logger.error(f"Firebase initialization error: {str(e)}")
        raise


def get_firestore_client():
    if _firebase_app is None:
        raise RuntimeError("Firebase has not been initialized")
    return firestore.client(_firebase_app)


def save_profile(user_id: str, profile: VisionProfile, metadata: Dict[str, Any]) -> None:
    client = get_firestore_client()
    document = client.collection("visionProfiles").document(user_id)
    payload = {"profile": asdict(profile), "metadata": metadata}
    document.set(payload, merge=True)


def load_profile(user_id: str) -> Optional[VisionProfile]:
    client = get_firestore_client()
    document = client.collection("visionProfiles").document(user_id).get()
    if not document.exists:
        return None
    payload = document.to_dict() or {}
    profile_data = payload.get("profile")
    if not profile_data:
        return None
    return VisionProfile(
        deficiency=profile_data.get("deficiency", "normal"),
        severity=float(profile_data.get("severity", 0.0)),
        confidence=float(profile_data.get("confidence", 0.0)),
    )
