# Production Deployment Guide

## 🚀 Quick Deployment Options

### Option 1: Heroku (Easiest)

1. **Install Heroku CLI**:
   ```powershell
   winget install Heroku.HerokuCLI
   ```

2. **Create Procfile**:
   ```
   web: gunicorn --chdir backend app.main:app
   ```

3. **Deploy**:
   ```powershell
   heroku login
   heroku create daltonization-api
   git push heroku main
   heroku config:set FIREBASE_CREDENTIAL_PATH=/app/firebase-admin.json
   ```

### Option 2: Google Cloud Run (Serverless)

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY backend backend/
   COPY models models/
   ENV FIREBASE_CREDENTIAL_PATH=/app/secrets/firebase-admin.json
   CMD ["gunicorn", "--bind", "0.0.0.0:8080", "backend.app.main:app"]
   ```

2. **Deploy**:
   ```powershell
   gcloud run deploy daltonization-api --source .
   ```

### Option 3: AWS Elastic Beanstalk

1. **Install EB CLI**:
   ```powershell
   pip install awsebcli
   ```

2. **Initialize and deploy**:
   ```powershell
   eb init -p python-3.11 daltonization-api
   eb create daltonization-prod
   eb deploy
   ```

## 📋 Pre-Deployment Checklist

- [ ] Install gunicorn: `pip install gunicorn`
- [ ] Update requirements.txt
- [ ] Set environment variables securely
- [ ] Configure CORS origins (change from `*` to specific domains)
- [ ] Set up production Firebase security rules
- [ ] Enable HTTPS/SSL
- [ ] Set up monitoring (e.g., Sentry)
- [ ] Configure logging aggregation
- [ ] Set up CI/CD pipeline
- [ ] Add rate limiting
- [ ] Configure database backups
- [ ] Document API endpoints (Swagger/OpenAPI)

## 🔒 Security Hardening

1. **Update CORS** in `backend/app/__init__.py`:
   ```python
   CORS(app, resources={r"/*": {"origins": ["https://yourdomain.com"]}})
   ```

2. **Add rate limiting**:
   ```powershell
   pip install Flask-Limiter
   ```

3. **Enable authentication**:
   - Verify Firebase ID tokens
   - Add API key authentication
   - Implement request signing

4. **Firestore Security Rules**:
   ```javascript
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /visionProfiles/{userId} {
         allow read, write: if request.auth != null && request.auth.uid == userId;
       }
     }
   }
   ```

## 📊 Monitoring Setup

1. **Add Sentry for error tracking**:
   ```python
   import sentry_sdk
   sentry_sdk.init(dsn="YOUR_SENTRY_DSN")
   ```

2. **Add health check monitoring** (already implemented):
   - `/health/` - Basic status
   - `/health/ready` - Dependency checks

3. **Set up alerts**:
   - Response time > 2s
   - Error rate > 1%
   - CPU usage > 80%

## 🔧 Performance Optimization

- [ ] Enable response compression
- [ ] Add Redis caching for profiles
- [ ] Optimize image processing (resize before processing)
- [ ] Use CDN for static assets
- [ ] Implement async processing for large images
- [ ] Add request queuing for high load

## 📱 Mobile App Deployment

1. **Update API_BASE_URL** in mobile app to production URL
2. **Build release APK/AAB**:
   ```powershell
   cd mobile/android
   ./gradlew assembleRelease
   ```
3. **Submit to Google Play Store**
4. **Build iOS app with Xcode**
5. **Submit to Apple App Store**

## 🧪 Testing in Production

```powershell
# Test health endpoint
Invoke-RestMethod https://your-api.com/health/

# Test calibration
$profile = @{user_id='test'; responses=@{p1='incorrect'; p2='correct'; p3='incorrect'}} | ConvertTo-Json
Invoke-RestMethod -Method Post https://your-api.com/calibration/ -ContentType 'application/json' -Body $profile

# Test image processing
$img = [Convert]::ToBase64String([IO.File]::ReadAllBytes('test.png'))
$payload = @{user_id='test'; image_base64=$img} | ConvertTo-Json
Invoke-RestMethod -Method Post https://your-api.com/process/ -ContentType 'application/json' -Body $payload
```
