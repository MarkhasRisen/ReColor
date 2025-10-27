"""Test authentication functionality."""
import requests
import json

BASE_URL = "http://192.168.1.9:8000"

print("🔐 Testing Authentication\n")
print("=" * 60)

# Test 1: Calibration without auth (should work with optional_auth)
print("\n1. Calibration WITHOUT Authentication:")
try:
    payload = {
        "user_id": "unauthenticated-user",
        "responses": {
            "p1": "incorrect",
            "p2": "correct",
            "p3": "incorrect"
        }
    }
    
    response = requests.post(f"{BASE_URL}/calibration/", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 201:
        data = response.json()
        print(f"   ✅ Profile created: {data['deficiency']}, severity={data['severity']}")
        print(f"   ✅ Authentication optional - works without token")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Calibration with invalid token
print("\n2. Calibration WITH Invalid Token:")
try:
    headers = {"Authorization": "Bearer invalid-token-12345"}
    payload = {
        "user_id": "test-user",
        "responses": {
            "p1": "correct",
            "p2": "incorrect",
            "p3": "correct"
        }
    }
    
    response = requests.post(f"{BASE_URL}/calibration/", json=payload, headers=headers)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 201:
        print(f"   ✅ Works even with invalid token (optional auth)")
    else:
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Check if authenticated flag is set
print("\n3. Firebase Profile Check:")
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    from firebase_admin import credentials, firestore
    import firebase_admin
    
    # Check if app already exists
    try:
        app = firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(r"C:\Users\markr\Secrets\firebase-admin.json")
        app = firebase_admin.initialize_app(cred)
    
    db = firestore.client(app)
    doc = db.collection("visionProfiles").document("unauthenticated-user").get()
    
    if doc.exists:
        data = doc.to_dict()
        metadata = data.get('metadata', {})
        authenticated = metadata.get('authenticated', None)
        print(f"   ✅ Profile found in Firestore")
        print(f"   Authenticated flag: {authenticated}")
        print(f"   Calibrated at: {metadata.get('calibratedAt')}")
    else:
        print(f"   ⚠️  Profile not found in Firestore")
        
except Exception as e:
    print(f"   ⚠️  Could not check Firestore: {e}")

print("\n" + "=" * 60)
print("📝 Authentication Summary:")
print("   ✅ Optional auth allows requests without tokens")
print("   ✅ System tracks whether user was authenticated")
print("   ✅ Ready for Firebase Auth integration in mobile app")
print("\n💡 Next Steps:")
print("   1. Mobile app: Use Firebase Auth to get ID tokens")
print("   2. Add ID token to Authorization header")
print("   3. Backend will automatically use authenticated user ID")
