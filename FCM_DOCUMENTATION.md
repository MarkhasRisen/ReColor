# Firebase Cloud Messaging (FCM) - ReColor Implementation

## Overview

Firebase Cloud Messaging (FCM) has been integrated into ReColor to enable push notifications for:
- **Calibration reminders** - Notify users to recalibrate periodically
- **Processing completion** - Alert when color correction is complete
- **Profile updates** - Notify when vision profile changes
- **System announcements** - Broadcast updates to all users

## Architecture

### Backend Components

1. **FCM Service** (`backend/app/services/firebase.py`)
   - Device token management
   - Single user notifications
   - Multicast notifications (multiple users)
   - Topic-based broadcasting
   - Topic subscriptions/unsubscriptions

2. **Notification Routes** (`backend/app/routes/notifications.py`)
   - REST API endpoints for all FCM operations
   - Authentication middleware integration
   - Error handling and logging

### Firestore Collections

1. **deviceTokens**
   ```
   Document ID: user_id
   Fields:
     - token: string (FCM device token)
     - updatedAt: timestamp
     - deviceInfo: {
         platform: string (android/ios)
         model: string
         app_version: string
       }
   ```

2. **visionProfiles** (existing)
   - Used for user profile data

## API Endpoints

### 1. Register Device Token
**POST** `/notifications/register-device`

Register a device to receive push notifications.

**Authentication:** Optional (public endpoint)

**Request:**
```json
{
  "user_id": "user123",
  "device_token": "fcm_token_from_mobile_app",
  "device_info": {
    "platform": "android",
    "model": "Pixel 6",
    "app_version": "1.0.0"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Device token registered successfully"
}
```

### 2. Send Notification
**POST** `/notifications/send`

Send push notification to a specific user.

**Authentication:** Required (Bearer token)

**Request:**
```json
{
  "user_id": "user123",
  "title": "Calibration Complete",
  "body": "Your color vision profile has been updated",
  "data": {
    "type": "calibration",
    "profile_id": "abc123"
  },
  "image_url": "https://example.com/image.png"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Notification sent successfully"
}
```

### 3. Send Multicast Notification
**POST** `/notifications/send-multicast`

Send push notification to multiple users at once.

**Authentication:** Required (Bearer token)

**Request:**
```json
{
  "user_ids": ["user1", "user2", "user3"],
  "title": "System Update",
  "body": "New features are now available",
  "data": {
    "type": "announcement"
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "success_count": 2,
    "failure_count": 1,
    "failed_tokens": ["expired_token_xyz"]
  }
}
```

### 4. Send Topic Notification
**POST** `/notifications/topic/send`

Broadcast notification to all subscribers of a topic.

**Authentication:** Required (Bearer token)

**Request:**
```json
{
  "topic": "calibration_reminders",
  "title": "Time for Recalibration",
  "body": "It's been 30 days since your last calibration",
  "data": {
    "type": "reminder"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Topic notification sent successfully"
}
```

### 5. Subscribe to Topic
**POST** `/notifications/topic/subscribe`

Subscribe a device to receive topic-based notifications.

**Authentication:** Optional

**Request:**
```json
{
  "device_token": "fcm_token_here",
  "topic": "calibration_reminders"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully subscribed to topic: calibration_reminders"
}
```

### 6. Unsubscribe from Topic
**POST** `/notifications/topic/unsubscribe`

Unsubscribe a device from topic-based notifications.

**Authentication:** Optional

**Request:**
```json
{
  "device_token": "fcm_token_here",
  "topic": "calibration_reminders"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully unsubscribed from topic: calibration_reminders"
}
```

## Recommended Topics

- **`calibration_reminders`** - Periodic recalibration reminders
- **`system_updates`** - App updates and new features
- **`research_announcements`** - Academic research updates
- **`maintenance_alerts`** - Scheduled maintenance notifications

## Mobile Integration

### Android (React Native)

1. **Install FCM package:**
   ```bash
   npm install @react-native-firebase/app @react-native-firebase/messaging
   ```

2. **Request notification permission:**
   ```javascript
   import messaging from '@react-native-firebase/messaging';
   
   async function requestUserPermission() {
     const authStatus = await messaging().requestPermission();
     const enabled =
       authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
       authStatus === messaging.AuthorizationStatus.PROVISIONAL;
     
     if (enabled) {
       console.log('Authorization status:', authStatus);
     }
   }
   ```

3. **Get FCM token:**
   ```javascript
   const getFCMToken = async () => {
     const token = await messaging().getToken();
     
     // Register with backend
     await fetch('http://your-api.com/notifications/register-device', {
       method: 'POST',
       headers: {
         'Content-Type': 'application/json',
       },
       body: JSON.stringify({
         user_id: currentUserId,
         device_token: token,
         device_info: {
           platform: Platform.OS,
           model: DeviceInfo.getModel(),
           app_version: DeviceInfo.getVersion(),
         }
       }),
     });
   };
   ```

4. **Handle foreground messages:**
   ```javascript
   useEffect(() => {
     const unsubscribe = messaging().onMessage(async remoteMessage => {
       Alert.alert(
         remoteMessage.notification.title,
         remoteMessage.notification.body
       );
     });
     
     return unsubscribe;
   }, []);
   ```

5. **Handle background messages:**
   ```javascript
   messaging().setBackgroundMessageHandler(async remoteMessage => {
     console.log('Message handled in the background!', remoteMessage);
   });
   ```

### iOS Configuration

1. Add `GoogleService-Info.plist` to iOS project
2. Enable Push Notifications capability in Xcode
3. Request APNS certificate from Apple Developer Portal
4. Upload APNS certificate to Firebase Console

## Use Cases

### 1. Calibration Reminder (Scheduled)
```python
# Backend scheduled job (e.g., cron)
firebase.send_topic_notification(
    topic="calibration_reminders",
    title="Time to Recalibrate",
    body="For best results, recalibrate your color vision profile",
    data={"type": "reminder", "action": "open_calibration"}
)
```

### 2. Processing Complete
```python
# After color correction completes
firebase.send_notification(
    user_id=user_id,
    title="Processing Complete",
    body="Your image has been color corrected",
    data={
        "type": "processing_complete",
        "image_id": result_image_id
    },
    image_url=thumbnail_url
)
```

### 3. Profile Update
```python
# After calibration completes
firebase.send_notification(
    user_id=user_id,
    title="Profile Updated",
    body=f"Your vision profile: {profile.deficiency} (confidence: {profile.confidence:.0%})",
    data={
        "type": "profile_update",
        "deficiency": profile.deficiency,
        "severity": str(profile.severity)
    }
)
```

### 4. System Announcement (Broadcast)
```python
# Announce to all users
firebase.send_topic_notification(
    topic="system_updates",
    title="ReColor v2.0 Released",
    body="New CNN-based color identification now available!",
    data={"type": "announcement", "version": "2.0.0"}
)
```

## Testing

Run the FCM test suite:

```bash
# Start Flask server first
cd backend
..\.venv\Scripts\python.exe -m flask --app app.main run --host 0.0.0.0 --port 8000

# In another terminal, run tests
cd ..
.\.venv\Scripts\python.exe test_fcm.py
```

**Expected output:**
- ✅ Device registration should succeed
- ✅ Send notification should require authentication (401)
- ✅ Topic subscription endpoint should respond
- ✅ Invalid requests should return 400 errors

## Security Considerations

1. **Authentication:**
   - Sending notifications requires Firebase authentication
   - Device registration is public (optional auth)
   - Topic subscription is public

2. **Rate Limiting:**
   - Consider implementing rate limiting on send endpoints
   - FCM has quota limits (check Firebase Console)

3. **Token Management:**
   - Tokens should be refreshed periodically
   - Handle token expiration gracefully
   - Remove invalid tokens from database

4. **Data Privacy:**
   - Don't send sensitive data in notification payload
   - Use `data` field for internal references only
   - Keep notification text generic

## Firebase Console Configuration

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select project: `recolor-7d7fd`
3. Navigate to **Cloud Messaging**
4. Verify Server key is active
5. Check message delivery statistics

## Troubleshooting

### Token Registration Fails
- Verify Firebase credentials are loaded
- Check Firestore permissions
- Ensure user_id is valid

### Notifications Not Received
- Verify device token is registered
- Check Firebase Console for delivery status
- Ensure mobile app has notification permissions
- Verify FCM service is enabled in Firebase Console

### Topic Subscription Fails
- Check Firebase service account permissions
- Verify topic name format (alphanumeric + underscore)
- Ensure FCM is properly initialized

## Future Enhancements

- [ ] Implement notification scheduling (delayed delivery)
- [ ] Add notification templates for common messages
- [ ] Create notification history in Firestore
- [ ] Add analytics for notification engagement
- [ ] Implement notification preferences per user
- [ ] Add support for notification actions/buttons
- [ ] Create admin dashboard for managing notifications

## References

- [Firebase Cloud Messaging Documentation](https://firebase.google.com/docs/cloud-messaging)
- [React Native Firebase](https://rnfirebase.io/)
- [FCM HTTP v1 API](https://firebase.google.com/docs/reference/fcm/rest/v1/projects.messages)
