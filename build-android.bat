@echo off
echo ========================================
echo ReColor Android Build - Quick Setup
echo ========================================
echo.

REM Check if Java is installed
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Java JDK not found!
    echo.
    echo Please install Java JDK 17 first:
    echo   Option 1: winget install Microsoft.OpenJDK.17
    echo   Option 2: Download from https://www.oracle.com/java/technologies/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Java JDK found
java -version
echo.

REM Check if Android SDK exists
if not defined ANDROID_HOME (
    echo [WARNING] ANDROID_HOME not set. Android Studio may be required.
    echo.
)

REM Navigate to mobile directory
cd /d "%~dp0mobile"
if %errorlevel% neq 0 (
    echo [ERROR] mobile directory not found
    pause
    exit /b 1
)

echo [1/4] Installing npm dependencies...
call npm install
if %errorlevel% neq 0 (
    echo [ERROR] npm install failed
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo [2/4] Cleaning Android build...
cd android
call gradlew clean
if %errorlevel% neq 0 (
    echo [ERROR] Gradle clean failed
    cd ..
    pause
    exit /b 1
)
cd ..
echo [OK] Build cleaned
echo.

echo [3/4] Checking for connected devices...
adb devices
echo.

echo [4/4] Building and installing app...
echo This may take 5-10 minutes on first build...
call npx react-native run-android
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed. Common fixes:
    echo   1. Make sure a device/emulator is connected: adb devices
    echo   2. Try: cd android ; gradlew clean ; cd ..
    echo   3. Delete android/app/build folder
    echo   4. Check ANDROID_BUILD_FIXED.md for troubleshooting
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] App built and installed!
echo ========================================
echo.
echo The app should now be running on your device.
echo To start Metro bundler separately, run: npm start
echo.
pause
