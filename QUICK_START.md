# Quick Start Guide - Next Steps

## 🚀 Three Steps to Production

### Step 1: Get google-services.json (5 minutes)

1. Visit: https://console.firebase.google.com/project/recolor-7d7fd/settings/general
2. Scroll to "Your apps" section
3. Click on Android app (or add new Android app if not exists)
4. **Important**: Package name MUST be `com.recolor`
5. Download `google-services.json`
6. Copy to: `C:\Users\markr\Downloads\Daltonization\mobile\android\app\google-services.json`

### Step 2: Deploy Backend to Heroku (20 minutes)

```powershell
# Navigate to project
cd C:\Users\markr\Downloads\Daltonization

# Login to Heroku
heroku login

# Create app (choose your own name if recolor-api is taken)
heroku create recolor-api

# Encode and set Firebase credentials
$fileContent = Get-Content -Path "firebase-admin.json" -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
$base64 = [Convert]::ToBase64String($bytes)
heroku config:set FIREBASE_CREDENTIALS_BASE64=$base64

# Set CORS (use "*" for testing, change to your domain for production)
heroku config:set ALLOWED_ORIGINS="*"

# Deploy
git add .
git commit -m "Add Heroku deployment configuration"
git push heroku feature/kmeans-firebase-auth:main

# Wait for deployment (2-3 minutes)

# Test deployment
curl https://recolor-api.herokuapp.com/health
curl https://recolor-api.herokuapp.com/ishihara/info
```

### Step 3: Update Mobile App & Build (15 minutes)

```powershell
# Update API URL in mobile app
# Edit: mobile/src/services/api.ts
# Change API_BASE_URL to: "https://recolor-api.herokuapp.com"

# Build Android app
cd mobile
npx react-native run-android

# If successful, build release
cd android
.\gradlew bundleRelease

# Output will be at: app/build/outputs/bundle/release/app-release.aab
```

## 🔍 Verification Checklist

After each step, verify:

### After Step 1 (google-services.json)
- [ ] File exists at `mobile/android/app/google-services.json`
- [ ] File contains `"package_name": "com.recolor"`
- [ ] File contains `"project_id": "recolor-7d7fd"`

### After Step 2 (Heroku Deployment)
- [ ] `curl https://YOUR-APP.herokuapp.com/health` returns `{"status":"healthy"}`
- [ ] `curl https://YOUR-APP.herokuapp.com/ishihara/plates?mode=quick` returns 14 plates
- [ ] Heroku logs show no errors: `heroku logs --tail`

### After Step 3 (Mobile Build)
- [ ] App installs on emulator/device
- [ ] Ishihara test loads plate images
- [ ] Can submit test and see results
- [ ] Results are saved to Firebase (check Firestore console)

## 🐛 Quick Troubleshooting

### Issue: google-services.json not found
**Solution**: Ensure file is at exact path `mobile/android/app/google-services.json`

### Issue: Heroku deployment fails
**Solution**: 
```powershell
heroku logs --tail
# Look for error messages
# Common fixes:
# - Re-set Firebase credentials
# - Check requirements.txt has all dependencies
```

### Issue: Android build fails
**Solution**:
```powershell
# Refresh Java PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Check Java
java -version  # Should show 17.0.16

# Clean and rebuild
cd mobile/android
.\gradlew clean
.\gradlew assembleDebug
```

### Issue: Mobile app can't connect to API
**Solution**:
- Check API_BASE_URL in `mobile/src/services/api.ts`
- Verify Heroku app is running: `heroku ps`
- Check CORS settings allow your origin

## 📱 Testing on Physical Device

If using physical Android device instead of emulator:

1. Enable USB debugging on device
2. Connect via USB
3. Check connection: `adb devices`
4. Update API_BASE_URL in `api.ts`:
   ```typescript
   // For physical device on same network
   const API_BASE_URL = "http://192.168.1.9:8000";  // Your computer's IP
   
   // For production
   const API_BASE_URL = "https://recolor-api.herokuapp.com";
   ```

## 🎯 Success Criteria

You'll know everything is working when:
1. ✅ Backend responds to health check on Heroku
2. ✅ Mobile app loads Ishihara plates from backend
3. ✅ Test submission returns diagnosis results
4. ✅ Results save to Firestore (visible in Firebase Console)
5. ✅ No errors in Heroku logs or Android logcat

## 📞 Need Help?

### Check Logs
```powershell
# Backend logs
heroku logs --tail

# Mobile logs
npx react-native log-android
```

### Verify Configuration
```powershell
# Check Heroku env vars
heroku config

# Check Android build config
cd mobile/android
.\gradlew properties | Select-String -Pattern "android"
```

### Test Endpoints Manually
```powershell
# Health check
curl https://YOUR-APP.herokuapp.com/health

# Ishihara plates
curl https://YOUR-APP.herokuapp.com/ishihara/plates?mode=quick

# Test info
curl https://YOUR-APP.herokuapp.com/ishihara/info
```

---

**Estimated Total Time**: 40-60 minutes  
**Difficulty**: Medium  
**Prerequisites**: Heroku CLI installed, Firebase account access

**Last Updated**: October 27, 2025
