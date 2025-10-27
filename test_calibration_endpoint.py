"""Simple test of the calibration endpoint."""
import requests
import json

url = "http://127.0.0.1:8000/calibration/"
payload = {
    "user_id": "test-from-python",
    "responses": {
        "p1": "incorrect",
        "p2": "incorrect",
        "p3": "incorrect"
    }
}

print("Sending POST request...")
try:
    response = requests.post(url, json=payload)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
