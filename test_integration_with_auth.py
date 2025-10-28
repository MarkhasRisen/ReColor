"""Test the complete image processing pipeline with authentication."""
import requests
import base64
from pathlib import Path
from PIL import Image
import io
import firebase_admin
from firebase_admin import auth, credentials

# Initialize Firebase Admin (if not already initialized)
try:
    app = firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(r"C:\Users\markr\Secrets\firebase-admin.json")
    app = firebase_admin.initialize_app(cred)

# Create a test image with problematic colors for colorblind users
def create_test_image():
    img = Image.new('RGB', (400, 400))
    pixels = img.load()
    
    # Red-green gradient (problematic for protan/deutan)
    for y in range(400):
        for x in range(200):
            pixels[x, y] = (int(255 * x / 200), int(255 * y / 400), 0)
        for x in range(200, 400):
            pixels[x, y] = (0, int(255 * y / 400), int(255 * (x-200) / 200))
    
    return img

# Create a custom token for testing
user_id = "api-test-20251027-014505"
print(f"🔑 Creating custom Firebase token for user: {user_id}")
try:
    custom_token = auth.create_custom_token(user_id)
    print(f"✅ Custom token created")
    
    # Exchange custom token for ID token (this would normally happen on the client)
    # For testing, we'll use the custom token directly as Firebase Admin can verify it
    id_token = custom_token.decode('utf-8')
    
except Exception as e:
    print(f"❌ Error creating token: {e}")
    exit(1)

# Save and encode image
img = create_test_image()
img.save('test_input.png')

# Convert to base64
with open('test_input.png', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Send to processing endpoint with authentication
url = "http://127.0.0.1:8000/process/"
headers = {
    "Authorization": f"Bearer {id_token}",
    "Content-Type": "application/json"
}
payload = {
    "image_base64": image_b64
}

print("\n🖼️  Sending image for processing...")
print(f"   Input image: test_input.png (400x400)")
print(f"   User ID: {user_id}")
print(f"   Using Firebase Auth token")

try:
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    print(f"\n✅ Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        # The response format is: {"content_type": "image/png", "data": "<base64>"}
        corrected_b64 = result.get('data', '')
        if corrected_b64:
            corrected_bytes = base64.b64decode(corrected_b64)
            corrected_img = Image.open(io.BytesIO(corrected_bytes))
            corrected_img.save('test_output_corrected.png')
            print(f"✅ Corrected image saved: test_output_corrected.png")
            print(f"   Content type: {result.get('content_type', 'N/A')}")
            print(f"   Original size: {len(image_b64)} chars")
            print(f"   Corrected size: {len(corrected_b64)} chars")
            print(f"\n🎉 Integration test PASSED!")
        else:
            print(f"⚠️  No image data in response")
            print(f"   Response: {result}")
        
    else:
        print(f"❌ Error: {response.text}")
        
except requests.exceptions.Timeout:
    print("⚠️  Request timed out (processing may take a while for first run)")
except Exception as e:
    print(f"❌ Error: {e}")
