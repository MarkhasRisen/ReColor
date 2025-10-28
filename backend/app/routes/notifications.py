"""Routes for Firebase Cloud Messaging (FCM) notifications."""
from flask import Blueprint, request, jsonify, current_app
from ..middleware.auth import optional_auth, require_auth
from ..services import firebase

bp = Blueprint("notifications", __name__, url_prefix="/notifications")


@bp.route("/register-device", methods=["POST"])
@optional_auth
def register_device():
    """Register FCM device token for push notifications.
    
    Request body:
    {
        "user_id": "user123",
        "device_token": "fcm_token_here",
        "device_info": {  // optional
            "platform": "android",
            "model": "Pixel 6",
            "app_version": "1.0.0"
        }
    }
    """
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_id = data.get("user_id")
    device_token = data.get("device_token")
    
    if not user_id or not device_token:
        return jsonify({"error": "user_id and device_token are required"}), 400
    
    try:
        device_info = data.get("device_info")
        firebase.save_device_token(user_id, device_token, device_info)
        
        current_app.logger.info(f"Registered device token for user: {user_id}")
        
        return jsonify({
            "success": True,
            "message": "Device token registered successfully"
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to register device token: {e}")
        return jsonify({"error": "Failed to register device token"}), 500


@bp.route("/send", methods=["POST"])
@require_auth
def send_notification():
    """Send push notification to a user (requires authentication).
    
    Request body:
    {
        "user_id": "user123",
        "title": "Calibration Complete",
        "body": "Your color vision profile has been updated",
        "data": {  // optional custom data
            "type": "calibration",
            "profile_id": "abc123"
        },
        "image_url": "https://..."  // optional image
    }
    """
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_id = data.get("user_id")
    title = data.get("title")
    body = data.get("body")
    
    if not all([user_id, title, body]):
        return jsonify({"error": "user_id, title, and body are required"}), 400
    
    try:
        success = firebase.send_notification(
            user_id=user_id,
            title=title,
            body=body,
            data=data.get("data"),
            image_url=data.get("image_url")
        )
        
        if success:
            current_app.logger.info(f"Sent notification to user: {user_id}")
            return jsonify({
                "success": True,
                "message": "Notification sent successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "No device token found or send failed"
            }), 404
            
    except Exception as e:
        current_app.logger.error(f"Failed to send notification: {e}")
        return jsonify({"error": "Failed to send notification"}), 500


@bp.route("/send-multicast", methods=["POST"])
@require_auth
def send_multicast():
    """Send push notification to multiple users (requires authentication).
    
    Request body:
    {
        "user_ids": ["user1", "user2", "user3"],
        "title": "System Update",
        "body": "New features are now available",
        "data": {  // optional
            "type": "announcement"
        }
    }
    """
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_ids = data.get("user_ids")
    title = data.get("title")
    body = data.get("body")
    
    if not all([user_ids, title, body]):
        return jsonify({"error": "user_ids, title, and body are required"}), 400
    
    if not isinstance(user_ids, list):
        return jsonify({"error": "user_ids must be an array"}), 400
    
    try:
        result = firebase.send_multicast_notification(
            user_ids=user_ids,
            title=title,
            body=body,
            data=data.get("data")
        )
        
        current_app.logger.info(
            f"Multicast sent: {result['success_count']} success, "
            f"{result['failure_count']} failed"
        )
        
        return jsonify({
            "success": True,
            "result": result
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to send multicast: {e}")
        return jsonify({"error": "Failed to send multicast notification"}), 500


@bp.route("/topic/send", methods=["POST"])
@require_auth
def send_topic_notification():
    """Send push notification to a topic (broadcast).
    
    Request body:
    {
        "topic": "calibration_reminders",
        "title": "Time for Recalibration",
        "body": "It's been 30 days since your last calibration",
        "data": {  // optional
            "type": "reminder"
        }
    }
    """
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    topic = data.get("topic")
    title = data.get("title")
    body = data.get("body")
    
    if not all([topic, title, body]):
        return jsonify({"error": "topic, title, and body are required"}), 400
    
    try:
        success = firebase.send_topic_notification(
            topic=topic,
            title=title,
            body=body,
            data=data.get("data")
        )
        
        if success:
            current_app.logger.info(f"Sent topic notification to: {topic}")
            return jsonify({
                "success": True,
                "message": "Topic notification sent successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to send topic notification"
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Failed to send topic notification: {e}")
        return jsonify({"error": "Failed to send topic notification"}), 500


@bp.route("/topic/subscribe", methods=["POST"])
@optional_auth
def subscribe_topic():
    """Subscribe a device to an FCM topic.
    
    Request body:
    {
        "device_token": "fcm_token_here",
        "topic": "calibration_reminders"
    }
    """
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    device_token = data.get("device_token")
    topic = data.get("topic")
    
    if not device_token or not topic:
        return jsonify({"error": "device_token and topic are required"}), 400
    
    try:
        success = firebase.subscribe_to_topic(device_token, topic)
        
        if success:
            current_app.logger.info(f"Subscribed device to topic: {topic}")
            return jsonify({
                "success": True,
                "message": f"Successfully subscribed to topic: {topic}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to subscribe to topic"
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Failed to subscribe to topic: {e}")
        return jsonify({"error": "Failed to subscribe to topic"}), 500


@bp.route("/topic/unsubscribe", methods=["POST"])
@optional_auth
def unsubscribe_topic():
    """Unsubscribe a device from an FCM topic.
    
    Request body:
    {
        "device_token": "fcm_token_here",
        "topic": "calibration_reminders"
    }
    """
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    device_token = data.get("device_token")
    topic = data.get("topic")
    
    if not device_token or not topic:
        return jsonify({"error": "device_token and topic are required"}), 400
    
    try:
        success = firebase.unsubscribe_from_topic(device_token, topic)
        
        if success:
            current_app.logger.info(f"Unsubscribed device from topic: {topic}")
            return jsonify({
                "success": True,
                "message": f"Successfully unsubscribed from topic: {topic}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to unsubscribe from topic"
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Failed to unsubscribe from topic: {e}")
        return jsonify({"error": "Failed to unsubscribe from topic"}), 500
