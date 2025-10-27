"""Test image processing with LAB K-Means clustering and precomputed centroids."""
import base64
import json
import numpy as np
from PIL import Image
import requests

# Create a simple test image with red and green regions (protan confusion colors)
img = np.zeros((100, 200, 3), dtype=np.uint8)
img[:, :100] = [255, 0, 0]  # Left half: red
img[:, 100:] = [0, 255, 0]  # Right half: green

# Save test image
test_img = Image.fromarray(img)
test_img.save("test_red_green.png")
print("✅ Created test image: test_red_green.png (red and green)")

# Encode to base64
with open("test_red_green.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Test with the protan user we just created
payload = {
    "user_id": "test-kmeans-user",
    "image_base64": image_base64
}

print("\n🔄 Sending image to /process/ endpoint...")
print(f"   User: test-kmeans-user (protan, severity=0.83)")
print(f"   Image size: {len(image_base64)} bytes (base64)")

try:
    response = requests.post(
        "http://127.0.0.1:8000/process/",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Processing successful!")
        print(f"   Content type: {result.get('content_type')}")
        print(f"   Output size: {len(result.get('data', ''))} bytes")
        
        # Decode and save result
        output_data = base64.b64decode(result["data"])
        with open("test_red_green_corrected.png", "wb") as f:
            f.write(output_data)
        print(f"   Saved corrected image: test_red_green_corrected.png")
        
        print("\n🎨 Pipeline used:")
        print("   1. LAB color space conversion")
        print("   2. K-Means clustering (9 clusters)")
        print("   3. Profile-specific centroids (protan)")
        print("   4. Daltonization (protan correction)")
        print("   5. Blending with original")
        
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"\n❌ Request failed: {e}")
    print("   Make sure Flask server is running!")
