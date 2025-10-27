# React Native Mobile Client Setup

This guide helps you initialize and run the mobile app for adaptive daltonization.

## Prerequisites

- **Node.js** 18+ and **npm** or **yarn**
- **React Native CLI**: `npm install -g react-native-cli`
- **Android Studio** (for Android) or **Xcode** (for iOS/macOS only)
- **Java JDK 17+** (for Android builds)

## Initial Setup

### 1. Install Dependencies

From the `mobile/` directory:

```bash
cd mobile
npm install
# or
yarn install
```

### 2. Configure Firebase (Android)

1. Download `google-services.json` from Firebase Console → Project Settings → Your Android App
2. Place it in `mobile/android/app/google-services.json`
3. Ensure `mobile/android/build.gradle` includes:
   ```gradle
   classpath 'com.google.gms:google-services:4.4.0'
   ```
4. Ensure `mobile/android/app/build.gradle` includes:
   ```gradle
   apply plugin: 'com.google.gms.google-services'
   ```

### 3. Link Native Dependencies

```bash
npx react-native link
```

For iOS (macOS only):
```bash
cd ios
pod install
cd ..
```

### 4. Update API Endpoint

Edit `mobile/src/services/api.ts` and set the backend URL to match your Flask server (e.g., your local IP if testing on a physical device).

## Running the App

### Android

```bash
npx react-native run-android
```

Or open `mobile/android` in Android Studio and run from there.

### iOS (macOS only)

```bash
npx react-native run-ios
```

Or open `mobile/ios/DaltonizationMobile.xcworkspace` in Xcode and run.

## Development Workflow

1. Start Metro bundler in one terminal:
   ```bash
   npx react-native start
   ```

2. Launch the app in another terminal using the commands above.

3. Live reload is enabled by default—shake the device or press `Cmd+D` (iOS) / `Cmd+M` (Android) to open the dev menu.

## Troubleshooting

- **Module not found errors**: Run `npm install` again and ensure all native modules are linked.
- **Build failures on Android**: Clean the build with `cd android && ./gradlew clean && cd ..`
- **Firebase errors**: Verify `google-services.json` is in the correct location and the package name matches your Firebase app registration.

## Next Steps

- Test the calibration flow by navigating to the Ishihara Test screen.
- Capture or upload an image in the Live Preview screen and observe the corrected output.
- Wire on-device TFLite inference when models are ready to reduce latency.
