# Heroku Deployment Guide for ReColor Backend

## Prerequisites
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Have Firebase service account credentials ready (firebase-admin.json)

## Deployment Steps

### 1. Login to Heroku
```powershell
heroku login
```

### 2. Create Heroku App
```powershell
cd C:\Users\markr\OneDrive\Desktop\Daltonization
heroku create recolor-api
```

### 3. Add Python Buildpack
```powershell
heroku buildpacks:set heroku/python
```

### 4. Set Environment Variables

#### Firebase Credentials
```powershell
# Base64 encode your firebase-admin.json file
$fileContent = Get-Content -Path "firebase-admin.json" -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
$base64 = [Convert]::ToBase64String($bytes)
heroku config:set FIREBASE_CREDENTIALS_BASE64=$base64
```

#### CORS Configuration (IMPORTANT)
```powershell
# For development - allows all origins
heroku config:set ALLOWED_ORIGINS="*"

# For production - restrict to your domain(s)
heroku config:set ALLOWED_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"
```

#### Other Configuration
```powershell
heroku config:set FLASK_ENV=production
heroku config:set LOG_LEVEL=INFO
```

### 5. Create Buildpack Script to Decode Firebase Credentials

Create `.profile` file in project root (Heroku will run this on startup):

```bash
#!/bin/bash
if [ -n "$FIREBASE_CREDENTIALS_BASE64" ]; then
    echo $FIREBASE_CREDENTIALS_BASE64 | base64 --decode > /app/firebase-admin.json
    export FIREBASE_CREDENTIAL_PATH=/app/firebase-admin.json
fi
```

### 6. Deploy to Heroku
```powershell
git add .
git commit -m "Add Heroku deployment configuration"
git push heroku feature/kmeans-firebase-auth:main
```

### 7. Verify Deployment
```powershell
# Check logs
heroku logs --tail

# Test health endpoint
curl https://recolor-api.herokuapp.com/health
```

## Post-Deployment Configuration

### Scale Dynos
```powershell
# Use free tier
heroku ps:scale web=1

# Or upgrade for better performance
heroku ps:type hobby
```

### Monitor Application
```powershell
# View logs
heroku logs --tail

# Open app in browser
heroku open

# Check dyno status
heroku ps
```

## Update Mobile App

After deployment, update the mobile app API URL:

1. Open `mobile/src/services/api.ts`
2. Change `baseURL` from `http://localhost:8000` to `https://recolor-api.herokuapp.com`

## Testing Production API

```powershell
# Test health check
curl https://recolor-api.herokuapp.com/health

# Test Ishihara plates
curl https://recolor-api.herokuapp.com/ishihara/plates?mode=quick

# Test with authentication (requires Firebase ID token)
curl -X POST https://recolor-api.herokuapp.com/process `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer YOUR_FIREBASE_ID_TOKEN" `
  -d '{"user_id":"test","image_base64":"..."}'
```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `FIREBASE_CREDENTIALS_BASE64` | Base64-encoded Firebase credentials | (encoded JSON) |
| `ALLOWED_ORIGINS` | CORS allowed origins | `https://yourdomain.com` |
| `FLASK_ENV` | Flask environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Troubleshooting

### Application Error on Startup
```powershell
heroku logs --tail
# Check for missing dependencies or configuration errors
```

### Firebase Initialization Error
```powershell
# Verify credentials are set correctly
heroku config:get FIREBASE_CREDENTIALS_BASE64

# Re-encode and re-set if needed
```

### CORS Errors
```powershell
# Update allowed origins
heroku config:set ALLOWED_ORIGINS="https://your-actual-domain.com"
```

### Rate Limiting Issues
```powershell
# Increase rate limits by modifying backend/app/__init__.py
# Then redeploy:
git push heroku feature/kmeans-firebase-auth:main
```

## Cost Considerations

- **Free Tier**: 550-1000 dyno hours/month (app sleeps after 30 min inactivity)
- **Hobby Tier** ($7/month): Never sleeps, better for production
- **Professional Tier** ($25-$250/month): Autoscaling, metrics, multiple dynos

## Security Checklist

- ✅ Rate limiting enabled (flask-limiter)
- ✅ Authentication required on sensitive endpoints
- ⚠️ Set ALLOWED_ORIGINS to your domain (not "*")
- ✅ Firebase credentials stored as environment variable
- ⚠️ Consider adding SSL/TLS certificate for custom domain
- ⚠️ Monitor logs for suspicious activity

## Next Steps

1. Deploy backend to Heroku
2. Test all endpoints on production URL
3. Update mobile app with production API URL
4. Configure custom domain (optional)
5. Set up monitoring and alerts
6. Create backup strategy for Firestore data
