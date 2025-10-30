import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
// import firebase from './src/services/firebase'; // Uncomment when using native build

// Import all screens
import SplashScreen from './src/screens/SplashScreen';
import AuthScreen from './src/screens/AuthScreen';
import HomeScreen from './src/screens/HomeScreen';
import IshiharaTestScreen from './src/screens/IshiharaTestScreen';
import QuickTestScreen from './src/screens/QuickTestScreen';
import ComprehensiveTestScreen from './src/screens/ComprehensiveTestScreen';
import TestResultsScreen from './src/screens/TestResultsScreen';
import QuickSurveyScreen from './src/screens/QuickSurveyScreen';
import SurveyResultsScreen from './src/screens/SurveyResultsScreen';
import CameraScreen from './src/screens/CameraScreen';
import ColorEnhancementScreen from './src/screens/ColorEnhancementScreen';
import ColorIdentifierScreen from './src/screens/ColorIdentifierScreen';
import CVDSimulationScreen from './src/screens/CVDSimulationScreen';
import GalleryScreen from './src/screens/GalleryScreen';
import EducationScreen from './src/screens/EducationScreen';
import LearnDetailsScreen from './src/screens/LearnDetailsScreen';
import TestRunnerScreen from './src/screens/TestRunnerScreen';

const Stack = createStackNavigator();

const App = () => {
  useEffect(() => {
    // Initialize Firebase when component mounts
    // Note: @react-native-firebase requires native build, not compatible with Expo Go
    // For Expo Go demo, Firebase features are commented out
    console.log('ReColor App Initialized');
    // console.log('Firebase initialized:', firebase.apps.length > 0);
  }, []);

  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Splash"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#4A90E2',
          },
          headerTintColor: '#FFF',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        {/* Auth Flow */}
        <Stack.Screen 
          name="Splash" 
          component={SplashScreen} 
          options={{ headerShown: false }}
        />
        <Stack.Screen 
          name="Auth" 
          component={AuthScreen} 
          options={{ headerShown: false }}
        />

        {/* Main App */}
        <Stack.Screen 
          name="Main" 
          component={HomeScreen} 
          options={{ 
            title: 'ReColor',
            headerLeft: () => null,
          }}
        />

        {/* Ishihara Test Flow */}
        <Stack.Screen 
          name="IshiharaTest" 
          component={IshiharaTestScreen} 
          options={{ title: 'Ishihara Test' }}
        />
        <Stack.Screen 
          name="QuickTest" 
          component={QuickTestScreen} 
          options={{ title: 'Quick Test (14 Plates)' }}
        />
        <Stack.Screen 
          name="ComprehensiveTest" 
          component={ComprehensiveTestScreen} 
          options={{ title: 'Comprehensive Test (38 Plates)' }}
        />
        <Stack.Screen 
          name="TestResults" 
          component={TestResultsScreen} 
          options={{ title: 'Test Results' }}
        />

        {/* Survey Flow */}
        <Stack.Screen 
          name="QuickSurvey" 
          component={QuickSurveyScreen} 
          options={{ title: 'Quick Survey' }}
        />
        <Stack.Screen 
          name="SurveyResults" 
          component={SurveyResultsScreen} 
          options={{ title: 'Survey Results' }}
        />

        {/* Camera Features */}
        <Stack.Screen 
          name="Camera" 
          component={CameraScreen} 
          options={{ title: 'Camera Features' }}
        />
        <Stack.Screen 
          name="ColorEnhancement" 
          component={ColorEnhancementScreen} 
          options={{ title: 'Color Enhancement' }}
        />
        <Stack.Screen 
          name="ColorIdentifier" 
          component={ColorIdentifierScreen} 
          options={{ title: 'Color Identifier' }}
        />
        <Stack.Screen 
          name="CVDSimulation" 
          component={CVDSimulationScreen} 
          options={{ title: 'CVD Simulation' }}
        />
        <Stack.Screen 
          name="Gallery" 
          component={GalleryScreen} 
          options={{ title: 'Gallery' }}
        />

        {/* Education */}
        <Stack.Screen 
          name="Education" 
          component={EducationScreen} 
          options={{ title: 'Awareness & Education' }}
        />
        <Stack.Screen 
          name="LearnDetails" 
          component={LearnDetailsScreen} 
          options={{ title: 'Learn More' }}
        />

        {/* Testing & Debug */}
        <Stack.Screen 
          name="TestRunner" 
          component={TestRunnerScreen} 
          options={{ title: 'Algorithm Tests' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;

