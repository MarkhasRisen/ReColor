"""Test new API features."""
import requests

BASE_URL = "http://192.168.1.9:8000"

print("🧪 Testing New API Features\n")
print("=" * 60)

# Test health check
print("\n1. Health Check Endpoint:")
try:
    response = requests.get(f"{BASE_URL}/health/")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Status: {data['status']}")
        print(f"   ✅ Python: {data['python_version']}")
        print(f"   ✅ Service: {data['service']}")
    else:
        print(f"   ❌ Failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test readiness check
print("\n2. Readiness Check Endpoint:")
try:
    response = requests.get(f"{BASE_URL}/health/ready")
    data = response.json()
    print(f"   Ready: {data['ready']}")
    print(f"   Firebase: {'✅' if data['checks']['firebase'] else '❌'}")
    print(f"   Models: {'✅' if data['checks']['models'] else '❌'}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test CORS
print("\n3. CORS Headers:")
try:
    response = requests.options(f"{BASE_URL}/calibration/")
    cors_header = response.headers.get('Access-Control-Allow-Origin', 'Not set')
    print(f"   CORS: {cors_header}")
    if cors_header == '*':
        print("   ✅ CORS enabled for all origins")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ New features test complete!")
