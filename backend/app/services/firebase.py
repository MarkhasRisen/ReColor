"""Firebase service integration for profile storage and analytics."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from firebase_admin import credentials, firestore, initialize_app, messaging
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


# ========== Cloud Messaging (FCM) Functions ==========

def save_device_token(user_id: str, device_token: str, device_info: Optional[Dict[str, Any]] = None) -> None:
    """Save FCM device token for a user."""
    client = get_firestore_client()
    document = client.collection("deviceTokens").document(user_id)
    
    payload = {
        "token": device_token,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    
    if device_info:
        payload["deviceInfo"] = device_info
    
    document.set(payload, merge=True)


def get_device_token(user_id: str) -> Optional[str]:
    """Retrieve FCM device token for a user."""
    client = get_firestore_client()
    document = client.collection("deviceTokens").document(user_id).get()
    
    if not document.exists:
        return None
    
    data = document.to_dict() or {}
    return data.get("token")


def send_notification(
    user_id: str,
    title: str,
    body: str,
    data: Optional[Dict[str, str]] = None,
    image_url: Optional[str] = None
) -> bool:
    """Send push notification to a specific user.
    
    Args:
        user_id: Target user ID
        title: Notification title
        body: Notification body text
        data: Optional custom data payload
        image_url: Optional image URL for rich notification
        
    Returns:
        True if sent successfully, False otherwise
    """
    token = get_device_token(user_id)
    if not token:
        return False
    
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
                image=image_url if image_url else None,
            ),
            data=data or {},
            token=token,
        )
        
        response = messaging.send(message)
        return bool(response)
    except Exception:
        return False


def send_multicast_notification(
    user_ids: List[str],
    title: str,
    body: str,
    data: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Send push notification to multiple users.
    
    Args:
        user_ids: List of target user IDs
        title: Notification title
        body: Notification body text
        data: Optional custom data payload
        
    Returns:
        Dict with success_count, failure_count, and failed_tokens
    """
    # Get all tokens
    tokens = []
    for user_id in user_ids:
        token = get_device_token(user_id)
        if token:
            tokens.append(token)
    
    if not tokens:
        return {"success_count": 0, "failure_count": 0, "failed_tokens": []}
    
    try:
        message = messaging.MulticastMessage(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data=data or {},
            tokens=tokens,
        )
        
        response = messaging.send_multicast(message)
        
        return {
            "success_count": response.success_count,
            "failure_count": response.failure_count,
            "failed_tokens": [
                tokens[idx] for idx, resp in enumerate(response.responses)
                if not resp.success
            ]
        }
    except Exception as e:
        return {
            "success_count": 0,
            "failure_count": len(tokens),
            "failed_tokens": tokens,
            "error": str(e)
        }


def send_topic_notification(
    topic: str,
    title: str,
    body: str,
    data: Optional[Dict[str, str]] = None
) -> bool:
    """Send push notification to a topic (broadcast).
    
    Args:
        topic: FCM topic name (e.g., 'calibration_reminders')
        title: Notification title
        body: Notification body text
        data: Optional custom data payload
        
    Returns:
        True if sent successfully, False otherwise
    """
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data=data or {},
            topic=topic,
        )
        
        response = messaging.send(message)
        return bool(response)
    except Exception:
        return False


def subscribe_to_topic(token: str, topic: str) -> bool:
    """Subscribe a device token to an FCM topic."""
    try:
        response = messaging.subscribe_to_topic([token], topic)
        return response.success_count > 0
    except Exception:
        return False


def unsubscribe_from_topic(token: str, topic: str) -> bool:
    """Unsubscribe a device token from an FCM topic."""
    try:
        response = messaging.unsubscribe_from_topic([token], topic)
        return response.success_count > 0
    except Exception:
        return False
