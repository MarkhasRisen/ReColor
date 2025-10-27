# Firebase Authentication Setup Guide

## 🔐 Authentication Architecture

The backend now supports **optional authentication**:
- ✅ Works WITHOUT authentication (for testing/demos)
- ✅ Uses Firebase ID tokens when provided
- ✅ Automatically uses authenticated user ID
- ✅ Tracks authentication status in metadata

## 📱 Mobile App Integration

### 1. Enable Firebase Auth in Console

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Select your project: `recolor-7d7fd`
3. Navigate to **Authentication** → **Sign-in method**
4. Enable providers:
   - ✅ Email/Password
   - ✅ Google Sign-In
   - ✅ Anonymous (for testing)

### 2. Update Mobile App Code

Add authentication to your React Native app:

```typescript
// src/services/auth.ts
import auth from '@react-native-firebase/auth';

export async function signInAnonymously() {
  const userCredential = await auth().signInAnonymously();
  return userCredential.user;
}

export async function signInWithEmail(email: string, password: string) {
  const userCredential = await auth().signInWithEmailAndPassword(email, password);
  return userCredential.user;
}

export async function signUpWithEmail(email: string, password: string) {
  const userCredential = await auth().createUserWithEmailAndPassword(email, password);
  return userCredential.user;
}

export async function signOut() {
  await auth().signOut();
}

export async function getIdToken(): Promise<string | null> {
  const user = auth().currentUser;
  if (!user) return null;
  return await user.getIdToken();
}
```

### 3. Update API Service

Modify `src/services/api.ts` to include auth tokens:

```typescript
import { getIdToken } from './auth';

async function getAuthHeaders() {
  const token = await getIdToken();
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

export async function submitCalibration(payload: CalibrationPayload) {
  const headers = {
    'Content-Type': 'application/json',
    ...(await getAuthHeaders())
  };
  
  const response = await fetch(`${API_BASE_URL}/calibration/`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      user_id: payload.userId,
      responses: payload.responses
    }),
  });

  if (!response.ok) {
    throw new Error(`Calibration failed: ${response.status}`);
  }
  return response.json();
}

export async function submitImage(userId: string, imageBase64: string) {
  const headers = {
    'Content-Type': 'application/json',
    ...(await getAuthHeaders())
  };
  
  const response = await fetch(`${API_BASE_URL}/process/`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ user_id: userId, image_base64: imageBase64 }),
  });

  if (!response.ok) {
    throw new Error(`Processing failed: ${response.status}`);
  }
  return response.json();
}
```

### 4. Add Login Screen

Create `src/screens/Login.tsx`:

```typescript
import React, { useState } from 'react';
import { View, TextInput, Button, StyleSheet } from 'react-native';
import { signInWithEmail, signUpWithEmail, signInAnonymously } from '../services/auth';

export function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSignIn = async () => {
    try {
      await signInWithEmail(email, password);
      navigation.navigate('Calibration');
    } catch (error) {
      console.error('Sign in failed:', error);
    }
  };

  const handleSignUp = async () => {
    try {
      await signUpWithEmail(email, password);
      navigation.navigate('Calibration');
    } catch (error) {
      console.error('Sign up failed:', error);
    }
  };

  const handleAnonymous = async () => {
    try {
      await signInAnonymously();
      navigation.navigate('Calibration');
    } catch (error) {
      console.error('Anonymous sign in failed:', error);
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <Button title="Sign In" onPress={handleSignIn} />
      <Button title="Sign Up" onPress={handleSignUp} />
      <Button title="Continue as Guest" onPress={handleAnonymous} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    marginBottom: 10,
    borderRadius: 5,
  },
});
```

## 🧪 Testing Authentication

### Test Without Auth (Current Behavior):
```powershell
cd C:\Users\markr\OneDrive\Desktop\Daltonization
.\.venv\Scripts\python.exe test_authentication.py
```

### Test With Firebase Auth:

1. **Get a test ID token**:
   ```javascript
   // In browser console or Node.js
   firebase.auth().signInAnonymously().then(user => {
     return user.getIdToken();
   }).then(token => {
     console.log('ID Token:', token);
   });
   ```

2. **Use token in request**:
   ```powershell
   $token = "your-id-token-here"
   $headers = @{
       "Authorization" = "Bearer $token"
       "Content-Type" = "application/json"
   }
   $body = @{
       user_id = "test-user"
       responses = @{
           p1 = "incorrect"
           p2 = "correct"
           p3 = "incorrect"
       }
   } | ConvertTo-Json
   
   Invoke-RestMethod -Method Post http://192.168.1.9:8000/calibration/ -Headers $headers -Body $body
   ```

## 🔒 Firestore Security Rules

Update your Firestore rules to require authentication:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /visionProfiles/{userId} {
      // Users can only read/write their own profile
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

## 🎯 Benefits

1. **Security**: Only authenticated users can access their data
2. **Privacy**: Users can't see other users' profiles
3. **Analytics**: Track authenticated vs anonymous usage
4. **Scalability**: Firebase handles auth infrastructure
5. **Flexibility**: Multiple auth providers (email, Google, etc.)

## 📊 Authentication Metadata

The backend now tracks:
```json
{
  "profile": { ... },
  "metadata": {
    "calibratedAt": "2025-10-27T...",
    "source": "ishihara",
    "authenticated": true,  // NEW: tracks if user was authenticated
    "userId": "firebase-uid"  // Actual Firebase UID when authenticated
  }
}
```

## 🚀 Next Steps

1. Enable Firebase Auth in console
2. Add login screen to mobile app
3. Update API calls to include tokens
4. Deploy Firestore security rules
5. Test authenticated flow
6. Monitor usage in Firebase Console
