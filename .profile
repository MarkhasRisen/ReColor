#!/bin/bash
# Heroku startup script to decode Firebase credentials from environment variable
if [ -n "$FIREBASE_CREDENTIALS_BASE64" ]; then
    echo $FIREBASE_CREDENTIALS_BASE64 | base64 --decode > /app/firebase-admin.json
    export FIREBASE_CREDENTIAL_PATH=/app/firebase-admin.json
    echo "Firebase credentials decoded successfully"
fi
