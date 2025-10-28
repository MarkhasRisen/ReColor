# ReColor Android Build Verification
# Run this to check if everything is configured correctly

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ReColor Android Build Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Check 1: Java JDK
Write-Host "[1/8] Checking Java JDK..." -NoNewline
try {
    $javaVersion = java -version 2>&1 | Select-Object -First 1
    if ($javaVersion -match "(\d+)\.") {
        $majorVersion = [int]$matches[1]
        if ($majorVersion -ge 11) {
            Write-Host " ✅ PASS" -ForegroundColor Green
            Write-Host "      Version: $javaVersion" -ForegroundColor Gray
        } else {
            Write-Host " ⚠️  WARNING" -ForegroundColor Yellow
            Write-Host "      Java $majorVersion found, but Java 17+ recommended" -ForegroundColor Yellow
            $warnings++
        }
    }
} catch {
    Write-Host " ❌ FAIL" -ForegroundColor Red
    Write-Host "      Java not found. Install with: winget install Microsoft.OpenJDK.17" -ForegroundColor Red
    $errors++
}

# Check 2: Android folder structure
Write-Host "[2/8] Checking Android folder..." -NoNewline
if (Test-Path "mobile\android\app\build.gradle") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL" -ForegroundColor Red
    Write-Host "      Android folder missing or incomplete" -ForegroundColor Red
    $errors++
}

# Check 3: google-services.json
Write-Host "[3/8] Checking Firebase config..." -NoNewline
if (Test-Path "mobile\android\app\google-services.json") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ❌ FAIL" -ForegroundColor Red
    Write-Host "      google-services.json not found in mobile/android/app/" -ForegroundColor Red
    $errors++
}

# Check 4: Package name
Write-Host "[4/8] Checking package name..." -NoNewline
$buildGradle = Get-Content "mobile\android\app\build.gradle" -Raw
if ($buildGradle -match 'applicationId "com\.recolor"') {
    Write-Host " ✅ PASS" -ForegroundColor Green
    Write-Host "      Package: com.recolor" -ForegroundColor Gray
} else {
    Write-Host " ⚠️  WARNING" -ForegroundColor Yellow
    Write-Host "      Package name may not match Firebase configuration" -ForegroundColor Yellow
    $warnings++
}

# Check 5: Node modules
Write-Host "[5/8] Checking node_modules..." -NoNewline
if (Test-Path "mobile\node_modules") {
    Write-Host " ✅ PASS" -ForegroundColor Green
} else {
    Write-Host " ⚠️  WARNING" -ForegroundColor Yellow
    Write-Host "      Run: cd mobile; npm install" -ForegroundColor Yellow
    $warnings++
}

# Check 6: Android SDK
Write-Host "[6/8] Checking Android SDK..." -NoNewline
if ($env:ANDROID_HOME) {
    Write-Host " ✅ PASS" -ForegroundColor Green
    Write-Host "      ANDROID_HOME: $env:ANDROID_HOME" -ForegroundColor Gray
} else {
    Write-Host " ⚠️  WARNING" -ForegroundColor Yellow
    Write-Host "      ANDROID_HOME not set. Install Android Studio." -ForegroundColor Yellow
    $warnings++
}

# Check 7: ADB
Write-Host "[7/8] Checking ADB..." -NoNewline
try {
    $adbVersion = adb version 2>&1 | Select-Object -First 1
    Write-Host " ✅ PASS" -ForegroundColor Green
    Write-Host "      $adbVersion" -ForegroundColor Gray
} catch {
    Write-Host " ⚠️  WARNING" -ForegroundColor Yellow
    Write-Host "      ADB not found. Install Android Studio." -ForegroundColor Yellow
    $warnings++
}

# Check 8: Connected devices
Write-Host "[8/8] Checking connected devices..." -NoNewline
try {
    $devices = adb devices 2>&1 | Select-String -Pattern "device$"
    if ($devices.Count -gt 0) {
        Write-Host " ✅ PASS" -ForegroundColor Green
        Write-Host "      Found $($devices.Count) device(s)" -ForegroundColor Gray
    } else {
        Write-Host " ⚠️  WARNING" -ForegroundColor Yellow
        Write-Host "      No devices connected. Connect a device or start an emulator." -ForegroundColor Yellow
        $warnings++
    }
} catch {
    Write-Host " ⚠️  WARNING" -ForegroundColor Yellow
    Write-Host "      Cannot check devices (ADB not available)" -ForegroundColor Yellow
    $warnings++
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "✅ All checks passed! Ready to build." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. cd mobile" -ForegroundColor White
    Write-Host "  2. npx react-native run-android" -ForegroundColor White
    Write-Host ""
    Write-Host "Or simply run: .\build-android.bat" -ForegroundColor White
} elseif ($errors -eq 0) {
    Write-Host "⚠️  $warnings warning(s) found. Build may work but check warnings above." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Try building anyway:" -ForegroundColor Cyan
    Write-Host "  cd mobile; npx react-native run-android" -ForegroundColor White
} else {
    Write-Host "❌ $errors error(s) and $warnings warning(s) found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix the errors above before building." -ForegroundColor Red
    Write-Host "See ANDROID_BUILD_FIXED.md for detailed instructions." -ForegroundColor Yellow
}

Write-Host ""
