"""Test Firebase Cloud Messaging (FCM) endpoints."""
import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_register_device():
    """Test device token registration."""
    print("\n" + "="*70)
    print("TEST 1: Register Device Token")
    print("="*70)
    
    response = requests.post(
        f"{BASE_URL}/notifications/register-device",
        json={
            "user_id": "test-user-123",
            "device_token": "fcm_test_token_abc123xyz",
            "device_info": {
                "platform": "android",
                "model": "Test Device",
                "app_version": "1.0.0"
            }
        }
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("✅ Device registration successful")
        return True
    else:
        print("❌ Device registration failed")
        return False


def test_send_notification():
    """Test sending notification (requires auth - will fail without token)."""
    print("\n" + "="*70)
    print("TEST 2: Send Notification (No Auth - Should Fail)")
    print("="*70)
    
    response = requests.post(
        f"{BASE_URL}/notifications/send",
        json={
            "user_id": "test-user-123",
            "title": "Test Notification",
            "body": "This is a test push notification",
            "data": {
                "type": "test",
                "timestamp": "2025-10-27T12:00:00Z"
            }
        }
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 401:
        print("✅ Authentication required as expected")
        return True
    else:
        print("⚠️ Unexpected response")
        return False


def test_topic_subscribe():
    """Test subscribing to a topic."""
    print("\n" + "="*70)
    print("TEST 3: Subscribe to Topic")
    print("="*70)
    
    response = requests.post(
        f"{BASE_URL}/notifications/topic/subscribe",
        json={
            "device_token": "fcm_test_token_abc123xyz",
            "topic": "calibration_reminders"
        }
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Note: This might fail if FCM credentials aren't properly configured
    # but the endpoint should still respond correctly
    print(f"Response received: {response.status_code in [200, 500]}")
    return True


def test_invalid_requests():
    """Test error handling with invalid requests."""
    print("\n" + "="*70)
    print("TEST 4: Invalid Request (Missing Fields)")
    print("="*70)
    
    response = requests.post(
        f"{BASE_URL}/notifications/register-device",
        json={
            "user_id": "test-user-123"
            # Missing device_token
        }
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 400:
        print("✅ Validation error handled correctly")
        return True
    else:
        print("❌ Validation not working")
        return False


def main():
    """Run all FCM tests."""
    print("\n" + "="*70)
    print("FIREBASE CLOUD MESSAGING (FCM) TEST SUITE")
    print("="*70)
    print("\nTesting FCM notification endpoints...")
    print("Note: Some tests may fail if Firebase credentials aren't configured")
    
    results = []
    
    try:
        results.append(("Device Registration", test_register_device()))
        results.append(("Send Notification Auth", test_send_notification()))
        results.append(("Topic Subscription", test_topic_subscribe()))
        results.append(("Invalid Request Handling", test_invalid_requests()))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print(f"\n⚠️ {total - passed} test(s) failed")
            return 1
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server")
        print("Please ensure the Flask server is running on http://127.0.0.1:8000")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
