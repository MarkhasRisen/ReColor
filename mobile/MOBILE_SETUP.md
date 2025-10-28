# Mobile App Setup Guide

## ✅ Configuration Complete

The mobile app has been configured to connect to your backend at:
- **iOS/Physical Device**: `http://192.168.1.9:8000`
- **Android Emulator**: `http://10.0.2.2:8000`

## 🎯 Quick Test (Without Android Studio)

### Option 1: Expo Go on Physical Device

1. **Install Expo Go** on your Android/iOS device:
   - Android: [Google Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)
   - iOS: [App Store](https://apps.apple.com/app/expo-go/id982107779)

2. **Make sure Flask is running** in a terminal:
   ```powershell
   cd C:\Users\markr\Downloads\Daltonization\backend
   ..\.venv\Scripts\python.exe -m flask --app app.main run --host 0.0.0.0 --port 8000
   ```

3. **Start Expo development server**:
   ```powershell
   cd C:\Users\markr\Downloads\Daltonization\mobile
   npm start
   ```

4. **Scan QR code** with Expo Go app
   - Android: Scan directly in Expo Go
   - iOS: Scan with Camera app, open in Expo Go

5. **Test the app**:
   - Navigate to Calibration screen
   - Test responses (correct/incorrect/skip)
   - Submit and verify profile is saved
   - Try Live Preview to process an image

### Option 2: Android Emulator (Requires Android Studio)

#### Install Android Studio:
1. Download from https://developer.android.com/studio
2. Run installer (takes 5-10 GB disk space)
3. During setup, install:
   - Android SDK Platform 33 or 34
   - Android Virtual Device (AVD)
   - Android SDK Build-Tools

4. Add to PATH (PowerShell Admin):
   ```powershell
   [Environment]::SetEnvironmentVariable(
       "ANDROID_HOME",
       "$env:LOCALAPPDATA\Android\Sdk",
       "User"
   )
   
   $path = [Environment]::GetEnvironmentVariable("Path", "User")
   [Environment]::SetEnvironmentVariable(
       "Path",
       "$path;$env:LOCALAPPDATA\Android\Sdk\platform-tools;$env:LOCALAPPDATA\Android\Sdk\emulator",
       "User"
   )
   ```

5. Create an AVD in Android Studio:
   - Tools → Device Manager
   - Create Virtual Device
   - Select Pixel 6 or similar
   - Download system image (API 33 recommended)

#### Run the app:
```powershell
cd C:\Users\markr\Downloads\Daltonization\mobile
npx react-native run-android
```

## 🧪 Testing Checklist

### Calibration Flow:
- [ ] App connects to backend (check Flask logs for requests)
- [ ] Can select correct/incorrect/skip for each plate
- [ ] Submit button sends request
- [ ] Results display (deficiency, severity, confidence)
- [ ] Profile is saved to Firestore (verify with `verify_firebase.py`)

### Image Processing:
- [ ] Can select image from gallery
- [ ] Image uploads to backend
- [ ] Corrected image displays
- [ ] Processing completes in reasonable time (<5 seconds)

## 🐛 Troubleshooting

### "Network request failed"
- Make sure Flask is running on `0.0.0.0:8000` (not `127.0.0.1`)
- Check firewall isn't blocking port 8000
- On physical device, verify phone is on same WiFi as computer
- For emulator, verify using `http://10.0.2.2:8000`

### "Connection refused"
- Verify Flask server is running: `http://192.168.1.9:8000/calibration/` should work in browser
- Check Windows Firewall allows Python connections

### Camera permissions
- App will request permissions on first use
- If denied, manually enable in device settings

## 📱 Features Implemented

1. **Calibration Screen** (`src/screens/Calibration.tsx`)
   - 8 Ishihara test plates (protan/deutan/tritan)
   - Correct/Incorrect/Skip responses
   - Submits to `/calibration/` endpoint
   - Displays profile results

2. **Live Preview Screen** (`src/screens/LivePreview.tsx`)
   - Image picker from gallery
   - Submits to `/process/` endpoint
   - Displays corrected image

3. **API Service** (`src/services/api.ts`)
   - `submitCalibration()` - Send test responses
   - `submitImage()` - Process image

## 🚀 Next Steps

1. **Add Camera Capture**: Replace image picker with live camera using `react-native-vision-camera`
2. **Add Real Ishihara Images**: Replace placeholder text with actual test plate images
3. **Add Profile Storage**: Use AsyncStorage to save profile locally
4. **Add Loading States**: Show spinners during API calls
5. **Error Handling**: Display user-friendly error messages
6. **Navigation**: Add React Navigation for better flow

## 📊 Expected Behavior

### Calibration:
```
User selects responses → Submit → Backend analyzes → Returns profile → Saves to Firestore
```

### Image Processing:
```
User picks image → Base64 encode → Send to backend → K-Means + Daltonization → Return corrected → Display
```

All API calls should complete in 1-5 seconds depending on image size.
