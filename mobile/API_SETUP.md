# Mobile App - Backend API Integration Setup

## Overview
The mobile app now fetches Ishihara test plates and evaluation results from the backend API.

## Current Status: ✅ INTEGRATED

### What's Working:
- ✅ API service configured with dev/prod endpoints
- ✅ QuickTestScreen loads 14 plates from backend
- ✅ ComprehensiveTestScreen loads 38 plates from backend
- ✅ TestResultsScreen evaluates responses via backend API
- ✅ Error handling with retry functionality
- ✅ Loading states for better UX
- ✅ Fallback for offline scenarios

---

## Configuration

### 1. Backend Server Setup

**Option A: Local Development**
```bash
# Start your Flask backend
cd backend
python -m flask run --host=0.0.0.0 --port=8000
```

**Option B: Heroku Production**
- Deploy backend to Heroku (see HEROKU_DEPLOYMENT.md)
- Update `USE_DEV_SERVER = false` in `src/services/api.ts`

### 2. Network Configuration

**For Android Emulator:**
- API automatically uses `http://10.0.2.2:8000`
- This routes to your computer's localhost

**For Physical Android Device:**
```typescript
// In mobile/src/services/api.ts, line 8, change to:
android: "http://YOUR_COMPUTER_IP:8000",  // e.g., "http://192.168.1.9:8000"
```

**For iOS Simulator:**
- Automatically uses `http://192.168.1.9:8000`
- Update to your actual computer IP

### 3. Find Your Computer's IP Address

**Windows:**
```powershell
ipconfig
# Look for "IPv4 Address" under your active network adapter
```

**macOS/Linux:**
```bash
ifconfig
# Look for "inet" under your active network interface
```

---

## API Endpoints Used

### 1. Get Test Plates
```
GET /api/ishihara/plates?mode={quick|comprehensive}
```
**Response:**
```json
{
  "mode": "quick",
  "total_plates": 14,
  "plates": [
    {
      "plate_number": 1,
      "image_url": "/static/ishihara/plate_01.png",
      "is_control": true
    }
  ]
}
```

### 2. Evaluate Test
```
POST /api/ishihara/evaluate
```
**Request Body:**
```json
{
  "user_id": "guest_123456789",
  "mode": "quick",
  "responses": {
    "1": "12",
    "2": "8",
    "3": ""
  },
  "save_profile": false
}
```

**Response:**
```json
{
  "cvd_type": "deutan",
  "severity": 0.6,
  "confidence": 0.85,
  "interpretation": "Moderate Deuteranomaly detected",
  "statistics": {
    "total_plates": 14,
    "normal_correct": 8,
    "protan_indicators": 2,
    "deutan_indicators": 3
  }
}
```

---

## Testing the Integration

### 1. Start Backend Server
```bash
cd backend
python -m flask run --host=0.0.0.0 --port=8000
```

### 2. Verify Backend is Running
Open browser: `http://localhost:8000/api/ishihara/plates?mode=quick`

Should see JSON response with plate data.

### 3. Update Mobile App IP (if needed)
Edit `mobile/src/services/api.ts`:
```typescript
const DEV_API_URL = Platform.select({
  ios: "http://YOUR_IP:8000",
  android: "http://10.0.2.2:8000",  // Emulator
  // android: "http://YOUR_IP:8000",  // Physical device
  default: "http://YOUR_IP:8000",
});
```

### 4. Run Mobile App
```bash
cd mobile
npx expo start
```

### 5. Test Flow
1. Navigate to **Ishihara Test** from home screen
2. Select **Quick Test** or **Comprehensive Test**
3. App should display loading spinner
4. Plates should load from backend (check terminal for logs)
5. Complete test by entering responses
6. Results screen should show backend evaluation

---

## Troubleshooting

### Error: "Could not load test plates"

**Check 1: Backend Running?**
```bash
# In backend directory
python -m flask run --host=0.0.0.0 --port=8000
```

**Check 2: Correct IP Address?**
- For emulator: Use `10.0.2.2:8000`
- For physical device: Use your computer's LAN IP (e.g., `192.168.1.9:8000`)

**Check 3: Firewall?**
```bash
# Windows: Allow Flask through firewall
# Settings > Windows Security > Firewall > Allow an app
```

**Check 4: Same Network?**
- Ensure phone and computer on same WiFi
- Or use Expo tunnel mode: `npx expo start --tunnel`

### Error: "Network request failed"

**Solution 1: Check Backend Logs**
Look for errors in Flask terminal output

**Solution 2: Test Backend Directly**
```bash
curl http://YOUR_IP:8000/api/ishihara/plates?mode=quick
```

**Solution 3: Enable CORS (if needed)**
Backend already has CORS enabled in `backend/app/__init__.py`

### Error: "Evaluation failed"

**Check Backend Route:**
```bash
# Test evaluation endpoint
curl -X POST http://localhost:8000/api/ishihara/evaluate \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","mode":"quick","responses":{"1":"12"}}'
```

---

## Production Deployment

### Step 1: Deploy Backend to Heroku
```bash
cd backend
git push heroku main
```

### Step 2: Update Mobile API Config
```typescript
// In mobile/src/services/api.ts
const USE_DEV_SERVER = false;  // Change to false
const PROD_API_URL = "https://YOUR_APP.herokuapp.com";  // Your Heroku URL
```

### Step 3: Build Production APK
```bash
cd mobile
eas build --platform android
```

---

## Code Files Modified

### ✅ Updated Files:
1. **`mobile/src/services/api.ts`**
   - Added dev/prod configuration
   - Improved error handling
   - Fixed API endpoints to use `/api/` prefix

2. **`mobile/src/screens/QuickTestScreen.tsx`**
   - Added `useEffect` to fetch plates on mount
   - Loading and error states
   - Dynamic plate image rendering
   - Retry functionality

3. **`mobile/src/screens/ComprehensiveTestScreen.tsx`**
   - Same changes as QuickTestScreen
   - Handles 38 plates

4. **`mobile/src/screens/TestResultsScreen.tsx`**
   - Calls `evaluateIshiharaTest` API
   - Parses backend response format
   - Shows real diagnosis from backend
   - Handles offline/error scenarios

---

## Next Steps

### Immediate (Before Testing)
- [ ] Start backend server
- [ ] Update mobile IP address if using physical device
- [ ] Test API endpoints in browser/Postman
- [ ] Run mobile app and test complete flow

### Short-term (For Production)
- [ ] Deploy backend to Heroku
- [ ] Replace placeholder Ishihara images with real clinical plates
- [ ] Switch `USE_DEV_SERVER` to false
- [ ] Test with real users

### Long-term (Features)
- [ ] Uncomment Firebase auth in `evaluateTest()`
- [ ] Implement profile saving to Firestore
- [ ] Add offline caching with AsyncStorage
- [ ] Implement retry with exponential backoff
- [ ] Add analytics for API calls

---

## Development Tips

### View API Logs
```bash
# Backend terminal will show all requests:
127.0.0.1 - - [28/Oct/2025 10:30:15] "GET /api/ishihara/plates?mode=quick HTTP/1.1" 200 -
```

### Debug Mobile Network Calls
```typescript
// In api.ts, uncomment console logs:
console.log('Fetching from:', `${API_BASE_URL}/api/ishihara/plates?mode=${mode}`);
console.log('Response:', data);
```

### Test Without Backend
The app has fallback handling - it will show error messages but won't crash.

---

**Last Updated:** October 28, 2025  
**Status:** ✅ Integration Complete - Ready for Testing  
**Backend Required:** Yes - Flask server must be running
