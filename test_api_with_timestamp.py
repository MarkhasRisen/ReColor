"""Test calibration endpoint with a unique user ID."""
import requests
import json
from datetime import datetime

url = "http://127.0.0.1:8000/calibration/"
user_id = f"api-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

payload = {
    "user_id": user_id,
    "responses": {
        "p1": "correct",
        "p2": "incorrect",
        "p3": "correct"
    }
}

print(f"Creating profile for user: {user_id}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print("\nSending POST request...")

try:
    response = requests.post(url, json=payload)
    print(f"✅ Status code: {response.status_code}")
    print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 201:
        print(f"\n✅ Profile created successfully!")
        print(f"   User ID: {user_id}")
except Exception as e:
    print(f"❌ Error: {e}")
