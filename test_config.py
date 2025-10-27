"""Test if Firebase config is loaded correctly."""
from backend.app import create_app

app = create_app()
print(f"Firebase credential path: {app.config.get('FIREBASE_CREDENTIAL_PATH')}")
print(f"Firebase project ID: {app.config.get('FIREBASE_PROJECT_ID')}")
