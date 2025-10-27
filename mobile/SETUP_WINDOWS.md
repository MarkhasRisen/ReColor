# React Native Setup Guide for Windows

## ⚠️ Prerequisites Required

React Native requires native development tools. Here's what you need:

### 1. Android Studio (Required for Android Development)

**Download & Install:**
1. Download from: https://developer.android.com/studio
2. Run installer (choose "Standard" setup)
3. Install components:
   - Android SDK
   - Android SDK Platform
   - Android Virtual Device (AVD)
   - Android Emulator

**Time:** ~30-45 minutes (large download)

### 2. Android SDK Configuration

After Android Studio installation:

1. **Open Android Studio**
2. Click "More Actions" → "SDK Manager"
3. Under "SDK Platforms" tab, check:
   - ✅ Android 14.0 (UpsideDownCake) - API Level 34
   - ✅ Android 13.0 (Tiramisu) - API Level 33
   
4. Under "SDK Tools" tab, check:
   - ✅ Android SDK Build-Tools
   - ✅ Android Emulator
   - ✅ Android SDK Platform-Tools
   - ✅ Intel x86 Emulator Accelerator (HAXM)

5. Click "Apply" to install

### 3. Environment Variables

Add to your system PATH:

```powershell
# Run as Administrator in PowerShell:
[Environment]::SetEnvironmentVariable("ANDROID_HOME", "$env:LOCALAPPDATA\Android\Sdk", "User")
[Environment]::SetEnvironmentVariable("Path", "$env:Path;$env:LOCALAPPDATA\Android\Sdk\platform-tools;$env:LOCALAPPDATA\Android\Sdk\emulator", "User")
```

**Restart your terminal** after setting environment variables!

### 4. Verify Installation

```powershell
# Check adb (Android Debug Bridge)
adb version
# Should show: Android Debug Bridge version x.x.x

# Check emulator
emulator -list-avds
# Should list available virtual devices
```

## 🚀 Running the App

### Option A: Using Android Emulator (Recommended)

1. **Create Virtual Device** (if not exists):
   ```powershell
   # Open Android Studio
   # Click "More Actions" → "Virtual Device Manager"
   # Click "Create Device"
   # Choose: Pixel 6 (or any modern device)
   # System Image: API 34 (Android 14)
   # Finish
   ```

2. **Start Emulator**:
   ```powershell
   # From Android Studio Device Manager, click ▶️ Play button
   # Or from terminal:
   emulator -avd Pixel_6_API_34
   ```

3. **Run React Native App**:
   ```powershell
   cd mobile
   npm start
   # In another terminal:
   npx react-native run-android
   ```

### Option B: Using Physical Device

1. **Enable Developer Mode** on your Android phone:
   - Go to Settings → About Phone
   - Tap "Build Number" 7 times
   - Go back to Settings → Developer Options
   - Enable "USB Debugging"

2. **Connect via USB**:
   ```powershell
   # Verify device connected
   adb devices
   # Should show your device
   ```

3. **Run App**:
   ```powershell
   cd mobile
   npx react-native run-android
   ```

## 🔧 Troubleshooting

### "adb not found"
- Android Studio not installed or PATH not set
- Restart terminal after setting environment variables
- Check: `$env:ANDROID_HOME` should point to SDK

### "No connected devices"
- Start emulator first, or connect physical device
- Run `adb devices` to verify

### "SDK location not found"
Create `mobile/android/local.properties`:
```properties
sdk.dir=C:\\Users\\YOUR_USERNAME\\AppData\\Local\\Android\\Sdk
```

### Build fails with "SDK not found"
```powershell
# Accept all SDK licenses
cd mobile/android
./gradlew --stop
%ANDROID_HOME%\tools\bin\sdkmanager --licenses
```

### Metro bundler issues
```powershell
# Clear cache
cd mobile
npx react-native start --reset-cache
```

## ⏱️ Estimated Setup Time

- **Quick (if Android Studio already installed):** 10 minutes
- **Full setup (from scratch):** 1-2 hours
  - Android Studio download: 20-30 min
  - Installation: 15-20 min
  - SDK download: 15-30 min
  - Configuration: 10-15 min
  - First app build: 10-15 min

## 🎯 Alternative: Test Backend API Only

If you want to test without setting up React Native:

### Use Postman or Thunder Client (VS Code extension)

**Test Calibration:**
```http
POST http://127.0.0.1:8000/calibration/
Content-Type: application/json

{
  "user_id": "postman-test-user",
  "responses": {
    "p1": "incorrect",
    "p2": "incorrect",
    "p3": "correct"
  }
}
```

**Test Image Processing:**
```http
POST http://127.0.0.1:8000/process/
Content-Type: application/json

{
  "user_id": "postman-test-user",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
}
```

### Use Python Script
```python
# test_api_without_mobile.py
import requests
import base64

# Calibration
cal_response = requests.post('http://127.0.0.1:8000/calibration/', json={
    'user_id': 'python-test',
    'responses': {'p1': 'incorrect', 'p2': 'incorrect'}
})
print("Calibration:", cal_response.json())

# Image processing
with open('test_image.png', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

proc_response = requests.post('http://127.0.0.1:8000/process/', json={
    'user_id': 'python-test',
    'image_base64': image_b64
})
print("Processing:", proc_response.json())
```

## 📱 Mobile App Features (When Running)

Once you get the mobile app running, you'll have:

1. **Calibration Screen**
   - Ishihara color plate tests
   - Response tracking (Correct/Incorrect/Skip)
   - Profile generation and storage

2. **Live Preview Screen**
   - Image picker for testing
   - Real-time color correction preview
   - Before/after comparison

3. **Firebase Integration**
   - User authentication
   - Profile storage in Firestore
   - Cross-device synchronization

## 🎓 Learning Resources

- [React Native Environment Setup](https://reactnative.dev/docs/environment-setup)
- [Android Studio Documentation](https://developer.android.com/studio/intro)
- [React Native Debugging](https://reactnative.dev/docs/debugging)

## ✅ Summary

**Your backend is already working!** The mobile app setup is optional - it provides a nice UI for testing, but you can test all functionality via:
- Python scripts (already working - `test_image_processing.py`)
- Postman/Thunder Client
- cURL commands
- Direct HTTP requests

**If you want the full mobile experience:** Plan for 1-2 hours to set up Android Studio and the development environment.
