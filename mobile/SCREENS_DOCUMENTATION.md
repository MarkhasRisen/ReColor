# ReColor Mobile App - Screen Structure

## 📱 Complete App Navigation

### Authentication Flow
- **SplashScreen** - Initial loading screen with logo
- **AuthScreen** - Login/Register with guest mode option

### Main Features

#### 1. Home (HomeScreen)
Main dashboard with 4 feature cards:
- Ishihara Test
- Quick Survey
- Real-time Camera
- Awareness & Education

#### 2. Ishihara Test Module
- **IshiharaTestScreen** - Choose between Quick (14) or Comprehensive (38) test
- **QuickTestScreen** - 14-plate standard screening
- **ComprehensiveTestScreen** - 38-plate clinical-grade test
- **TestResultsScreen** - Detailed diagnosis with severity, confidence, and statistics

#### 3. Quick Survey Module
- **QuickSurveyScreen** - 5-question color perception survey
- **SurveyResultsScreen** - Risk assessment with recommendations

#### 4. Camera Module
- **CameraScreen** - Main menu for camera features:
  - **ColorEnhancementScreen** - Real-time color correction (placeholder)
  - **ColorIdentifierScreen** - Point-and-identify colors (placeholder)
  - **CVDSimulationScreen** - See how others see (placeholder)
  - **GalleryScreen** - Apply enhancement to photos (placeholder)

#### 5. Education Module
- **EducationScreen** - 6 educational topics menu
- **LearnDetailsScreen** - Detailed information on:
  - What is Color Blindness?
  - Types of CVD
  - Statistics & Demographics
  - Living with CVD
  - Genetics & Inheritance
  - Technology & Solutions

## 🎨 Design System

### Colors
- Primary Blue: `#4A90E2`
- Success Green: `#50C878`
- Warning Orange: `#FFA500`
- Error Red: `#FF6B6B`
- Purple: `#9B59B6`
- Gold: `#FFD700`

### Typography
- Title: 28-32px, Bold
- Subtitle: 14-16px, Regular
- Body: 16px, Regular
- Small: 12-14px, Regular

### Card Style
- Background: White
- Border Radius: 15px
- Padding: 20px
- Elevation/Shadow for depth
- Colored left border for categories

## 📂 File Structure

```
mobile/src/
├── screens/
│   ├── SplashScreen.tsx
│   ├── AuthScreen.tsx
│   ├── HomeScreen.tsx
│   ├── IshiharaTestScreen.tsx
│   ├── QuickTestScreen.tsx
│   ├── ComprehensiveTestScreen.tsx
│   ├── TestResultsScreen.tsx
│   ├── QuickSurveyScreen.tsx
│   ├── SurveyResultsScreen.tsx
│   ├── CameraScreen.tsx
│   ├── ColorEnhancementScreen.tsx
│   ├── ColorIdentifierScreen.tsx
│   ├── CVDSimulationScreen.tsx
│   ├── GalleryScreen.tsx
│   ├── EducationScreen.tsx
│   └── LearnDetailsScreen.tsx
├── components/
│   └── (reusable components)
├── services/
│   └── api.ts
└── assets/
    └── (images, fonts)
```

## 🚀 Next Steps

### Phase 1: Core Functionality
1. ✅ Screen structure created
2. ⏳ Connect Ishihara test to backend API
3. ⏳ Implement Firebase Auth in AuthScreen
4. ⏳ Add actual Ishihara plate images
5. ⏳ Connect TestResultsScreen to save profiles

### Phase 2: Camera Features
1. ⏳ Implement camera permissions
2. ⏳ Add react-native-vision-camera
3. ⏳ Real-time color enhancement pipeline
4. ⏳ Color identifier with RGB/HEX display
5. ⏳ CVD simulation filters
6. ⏳ Gallery image picker & enhancement

### Phase 3: Polish & Enhancement
1. ⏳ Add loading states
2. ⏳ Error handling & user feedback
3. ⏳ Offline mode support
4. ⏳ Profile management
5. ⏳ Settings screen
6. ⏳ Onboarding tutorial

### Phase 4: Advanced Features
1. ⏳ Push notifications
2. ⏳ Progress tracking
3. ⏳ Export/share results
4. ⏳ Community features
5. ⏳ Gamification elements

## 🔧 Running the App

```bash
# Install dependencies
cd mobile
npm install

# Run on Android
npx react-native run-android

# Run on iOS
npx react-native run-ios
```

## 📝 Notes

- All camera screens are currently placeholders
- Ishihara plate images need to be added to `backend/static/ishihara/`
- Firebase Auth needs to be enabled in Firebase Console
- Backend API should be deployed for full functionality

## 🎯 Current Status

✅ Complete UI/UX structure
✅ Navigation flow implemented
✅ All placeholder screens created
✅ Design system established
⏳ Awaiting backend integration
⏳ Camera features pending implementation
⏳ Real Ishihara images needed
