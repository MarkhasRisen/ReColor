"""List all profiles with detailed information."""
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\markr\Secrets\firebase-admin.json")
app = firebase_admin.initialize_app(cred)
db = firestore.client(app)

# Get all profiles
profiles_ref = db.collection("visionProfiles")
docs = profiles_ref.stream()

print("🔍 All profiles in Firestore:")
print("=" * 60)

count = 0
for doc in docs:
    count += 1
    data = doc.to_dict()
    print(f"\n{count}. Document ID: {doc.id}")
    print(f"   Full data: {data}")

if count == 0:
    print("⚠️ No profiles found")
else:
    print(f"\n{'=' * 60}")
    print(f"✅ Total: {count} profiles")
