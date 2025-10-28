# Deploy ReColor Backend to Heroku

## Prerequisites Status

✅ Heroku CLI installed (v7.53.0)
✅ Logged into Heroku as caparasmarkrisen.mrc@gmail.com
✅ Git repository ready with deployment files committed
⚠️ **Account verification required** - Add payment info at https://heroku.com/verify

## Step 1: Verify Your Heroku Account

**Before proceeding, you must verify your Heroku account:**

1. Go to: https://heroku.com/verify
2. Add a credit card (free tier requires verification)
3. No charges will be made for free dyno usage
4. This is a one-time requirement

## Step 2: Prepare Firebase Credentials

You need to get your Firebase Admin SDK credentials:

### Option A: If you have firebase-admin.json locally

```powershell
# Encode to base64
$fileContent = Get-Content -Path "backend/firebase-admin.json" -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
$base64 = [Convert]::ToBase64String($bytes)
$base64 | Set-Content -Path "firebase-credentials-base64.txt"
Write-Host "Saved to firebase-credentials-base64.txt"
```

### Option B: Download from Firebase Console

1. Go to: https://console.firebase.google.com/project/recolor-7d7fd/settings/serviceaccounts/adminsdk
2. Click "Generate New Private Key"
3. Save the JSON file
4. Encode it using the PowerShell command above

## Step 3: Create Heroku App

```powershell
# Create the app (after account verification)
heroku create recolor-api

# This will output:
# Creating ⬢ recolor-api... done
# https://recolor-api.herokuapp.com/ | https://git.heroku.com/recolor-api.git
```

## Step 4: Set Environment Variables

```powershell
# Set Firebase credentials (use the base64 string from Step 2)
$base64Creds = Get-Content -Path "firebase-credentials-base64.txt" -Raw
heroku config:set FIREBASE_CREDENTIALS_BASE64="$base64Creds" --app recolor-api

# Set CORS allowed origins (update with your actual domain after deployment)
heroku config:set ALLOWED_ORIGINS="https://recolor-api.herokuapp.com" --app recolor-api

# Optional: Set Flask secret key for sessions
heroku config:set SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" --app recolor-api

# Verify config
heroku config --app recolor-api
```

## Step 5: Deploy to Heroku

```powershell
# Push current branch to Heroku main
git push heroku feature/kmeans-firebase-auth:main

# If you get an error about the remote not existing:
heroku git:remote -a recolor-api
git push heroku feature/kmeans-firebase-auth:main
```

**Expected output:**
```
Enumerating objects: ...
Counting objects: ...
remote: Compressing source files... done.
remote: Building source:
remote: -----> Building on the Heroku-22 stack
remote: -----> Using buildpack: heroku/python
remote: -----> Python app detected
remote: -----> Installing python-3.13.1
remote: -----> Installing pip 24.0, setuptools 70.0.0 and wheel 0.43.0
remote: -----> Installing SQLite3
remote: -----> Installing requirements with pip
remote:        Collecting flask==3.1.2
remote:        ...
remote: -----> Discovering process types
remote:        Procfile declares types -> web
remote: -----> Compressing...
remote: -----> Launching...
remote:        Released v1
remote:        https://recolor-api.herokuapp.com/ deployed to Heroku
```

## Step 6: Verify Deployment

```powershell
# Check if app is running
heroku ps --app recolor-api

# Open app in browser
heroku open --app recolor-api

# Check logs
heroku logs --tail --app recolor-api

# Test API endpoints
Invoke-WebRequest -Uri "https://recolor-api.herokuapp.com/health" -UseBasicParsing
Invoke-WebRequest -Uri "https://recolor-api.herokuapp.com/ishihara/plates?mode=quick" -UseBasicParsing
```

## Step 7: Test Ishihara API

```powershell
# Get plates
$response = Invoke-WebRequest -Uri "https://recolor-api.herokuapp.com/ishihara/plates?mode=quick" -UseBasicParsing
$json = $response.Content | ConvertFrom-Json
Write-Host "Total plates: $($json.total_plates)"

# Test evaluation (example)
$body = @{
    mode = "quick"
    responses = @{
        "1" = "12"
        "3" = "6"
        "4" = "29"
    }
} | ConvertTo-Json

$evalResponse = Invoke-WebRequest -Uri "https://recolor-api.herokuapp.com/ishihara/evaluate" `
    -Method POST `
    -Body $body `
    -ContentType "application/json" `
    -UseBasicParsing

$evalJson = $evalResponse.Content | ConvertFrom-Json
Write-Host "Diagnosis: $($evalJson.diagnosis.cvd_type)"
Write-Host "Confidence: $($evalJson.diagnosis.confidence)"
```

## Step 8: Update Mobile App API URL

After successful deployment, update the mobile app to use the production URL:

**File**: `mobile/src/services/api.ts`

```typescript
// Change from:
const API_BASE_URL = Platform.select({
  ios: 'http://localhost:8000',
  android: 'http://10.0.2.2:8000',
});

// To:
const API_BASE_URL = 'https://recolor-api.herokuapp.com';
```

## Common Issues & Solutions

### Issue 1: Build fails with Python version error
**Solution**: Heroku supports Python 3.13.1 (specified in runtime.txt). If there's an issue, check Heroku's supported versions:
```powershell
heroku buildpacks:versions heroku/python
```

### Issue 2: Firebase credentials not working
**Solution**: Verify the base64 encoding is correct:
```powershell
# Test decoding locally
$base64 = Get-Content "firebase-credentials-base64.txt" -Raw
$bytes = [Convert]::FromBase64String($base64)
$decoded = [System.Text.Encoding]::UTF8.GetString($bytes)
$decoded | ConvertFrom-Json  # Should show valid JSON
```

### Issue 3: App crashes on startup
**Solution**: Check logs for errors:
```powershell
heroku logs --tail --app recolor-api
```

Common fixes:
- Ensure `FIREBASE_CREDENTIALS_BASE64` is set correctly
- Check that all dependencies in requirements.txt are available
- Verify Procfile syntax is correct

### Issue 4: CORS errors in mobile app
**Solution**: Update ALLOWED_ORIGINS:
```powershell
heroku config:set ALLOWED_ORIGINS="*" --app recolor-api
# Or specify exact domains:
heroku config:set ALLOWED_ORIGINS="https://recolor-api.herokuapp.com,https://yourdomain.com" --app recolor-api
```

### Issue 5: Static files (Ishihara plates) not loading
**Solution**: Heroku serves static files through Flask. Verify:
- Files are committed to git (not in .gitignore)
- Flask is configured to serve /static/ directory
- Check: https://recolor-api.herokuapp.com/static/ishihara/plate_01.png

## Monitoring & Maintenance

### View logs
```powershell
heroku logs --tail --app recolor-api
```

### Restart dyno
```powershell
heroku restart --app recolor-api
```

### Scale dynos (if needed for production)
```powershell
# Current status
heroku ps --app recolor-api

# Scale to hobby tier (7/month, never sleeps)
heroku ps:scale web=1:hobby --app recolor-api

# Scale to free tier
heroku ps:scale web=1:free --app recolor-api
```

### Update environment variables
```powershell
heroku config:set VARIABLE_NAME="value" --app recolor-api
heroku config --app recolor-api  # View all
heroku config:unset VARIABLE_NAME --app recolor-api  # Remove one
```

## Post-Deployment Checklist

- [ ] Account verified with payment information
- [ ] Firebase credentials encoded and set as env var
- [ ] App created and deployed successfully
- [ ] Health endpoint responding: `/health`
- [ ] Ishihara API working: `/ishihara/plates?mode=quick`
- [ ] Logs show no errors: `heroku logs --tail`
- [ ] CORS configured correctly
- [ ] Mobile app updated with production URL
- [ ] Test color correction endpoint: `/process`
- [ ] Firebase authentication working
- [ ] Static files (Ishihara plates) accessible

## Next Steps After Deployment

1. **Update mobile app** with production URL
2. **Test all endpoints** from mobile app
3. **Monitor logs** for any issues
4. **Set up custom domain** (optional): https://devcenter.heroku.com/articles/custom-domains
5. **Enable HTTPS** (automatic with Heroku)
6. **Set up CI/CD** for automatic deployments on git push

## Resources

- Heroku Python Support: https://devcenter.heroku.com/articles/python-support
- Heroku Config Vars: https://devcenter.heroku.com/articles/config-vars
- Heroku Logs: https://devcenter.heroku.com/articles/logging
- Account Verification: https://devcenter.heroku.com/articles/account-verification

---

## Quick Deployment Script (After Account Verification)

Copy and run this after verifying your account:

```powershell
# 1. Encode Firebase credentials (if you have the file)
$fileContent = Get-Content -Path "backend/firebase-admin.json" -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
$base64 = [Convert]::ToBase64String($bytes)
$base64 | Set-Content -Path "firebase-credentials-base64.txt"

# 2. Create app
heroku create recolor-api

# 3. Set environment variables
$base64Creds = Get-Content -Path "firebase-credentials-base64.txt" -Raw
heroku config:set FIREBASE_CREDENTIALS_BASE64="$base64Creds" --app recolor-api
heroku config:set ALLOWED_ORIGINS="*" --app recolor-api

# 4. Deploy
git push heroku feature/kmeans-firebase-auth:main

# 5. Test
heroku open --app recolor-api
heroku logs --tail --app recolor-api

Write-Host "✅ Deployment complete! Test at: https://recolor-api.herokuapp.com"
```
