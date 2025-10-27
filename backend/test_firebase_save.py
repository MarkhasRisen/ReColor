"""Direct test of Firebase save functionality with detailed logging."""
import os
import sys
from datetime import datetime, timezone

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from firebase_admin import credentials, firestore, initialize_app
from app.pipeline.profile import VisionProfile

# Initialize Firebase
print("🔥 Testing Firebase Save Functionality")
print("=" * 60)

credential_path = r"C:\Users\markr\Secrets\firebase-admin.json"
print(f"📁 Loading credentials from: {credential_path}")

cred = credentials.Certificate(credential_path)
firebase_app = initialize_app(cred, {"projectId": "recolor-7d7fd"})
print("✅ Firebase app initialized")

# Get Firestore client
db = firestore.client(firebase_app)
print("✅ Firestore client created")

# Create test profile
profile = VisionProfile(
    deficiency="protan",
    severity=1.0,
    confidence=1.0
)
print(f"✅ Test profile created: {profile}")

# Prepare data
user_id = "test-save-debugging"
metadata = {
    "calibratedAt": datetime.now(timezone.utc).isoformat(),
    "source": "ishihara",
}

payload = {
    "profile": {
        "deficiency": profile.deficiency,
        "severity": profile.severity,
        "confidence": profile.confidence,
    },
    "metadata": metadata,
}

print(f"\n📝 Attempting to save profile for user: {user_id}")
print(f"📦 Payload: {payload}")

try:
    # Save to Firestore
    doc_ref = db.collection("visionProfiles").document(user_id)
    print(f"📍 Document reference: visionProfiles/{user_id}")
    
    doc_ref.set(payload, merge=True)
    print("✅ Document.set() called successfully")
    
    # Verify the write
    print("\n🔍 Verifying write...")
    doc_snapshot = doc_ref.get()
    
    if doc_snapshot.exists:
        print("✅ Document exists in Firestore!")
        print(f"📄 Document data: {doc_snapshot.to_dict()}")
    else:
        print("❌ Document does NOT exist after write!")
        
    # List all profiles
    print("\n📊 All profiles in collection:")
    profiles_ref = db.collection("visionProfiles")
    docs = profiles_ref.stream()
    
    count = 0
    for doc in docs:
        count += 1
        print(f"  {count}. {doc.id}: {doc.to_dict()}")
    
    if count == 0:
        print("  ⚠️ No documents found in collection")
    
except Exception as e:
    print(f"❌ Error during save: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 Test complete")
