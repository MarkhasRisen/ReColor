import {
  createNavigationContainerRef,
  NavigationContainer,
} from "@react-navigation/native";
import {
  CardStyleInterpolators,
  createStackNavigator,
} from "@react-navigation/stack";
import { useEffect, useRef } from "react";
import { View } from "react-native";

import { auth, onAuthStateChanged } from "./firebaseConfig";
import {
  installGlobalErrorHandlers,
  ScreenErrorBoundary,
} from "./src/utils/logger";

const navigationRef = createNavigationContainerRef();

// Screens
import AdminHubScreen from "./src/screens/AdminHubScreen";
import AdminLoginScreen from "./src/screens/AdminLoginScreen";
import AppOnboarding from "./src/screens/AppOnboarding";
import ArticleScreen from "./src/screens/ArticleScreen";
import CameraEnhanceScreen from "./src/screens/CameraEnhanceScreen";
import CareerDetailScreen from "./src/screens/CareerDetail"; // Added for Career Awareness
import ColorIdentifierScreen from "./src/screens/ColorIdentifierScreen";
import CVDGalleryScreen from "./src/screens/CVDGalleryScreen";
import CVDSimulationScreen from "./src/screens/CVDSimulationScreen";
import EducationListScreen from "./src/screens/EducationListScreen";
import IshiharaIntroScreen from "./src/screens/IshiharaIntroScreen";
import IshiharaOnboarding from "./src/screens/IshiharaOnboarding";
import LoginScreen from "./src/screens/LoginScreen";
import ResearchDashboardScreen from "./src/screens/ResearchDashboardScreen";
import ResultsScreen from "./src/screens/ResultsScreen";
import SettingsScreen from "./src/screens/SettingsScreen";
import SignUp from "./src/screens/SignUp";
import SplashScreen from "./src/screens/SplashScreen";
import SurveyScreen from "./src/screens/SurveyScreen";
import SurveySuccessScreen from "./src/screens/SurveySuccessScreen";
import TestScreen from "./src/screens/TestScreen";

// Navigation
import MainTabNavigator from "./src/navigation/MainTabNavigator";

// Theme
import { COLORS } from "./src/theme/colors";

installGlobalErrorHandlers();

const Stack = createStackNavigator();

export default function App() {
  const isFirstAuthEvent = useRef(true);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (user) => {
      if (isFirstAuthEvent.current) {
        isFirstAuthEvent.current = false;
        return;
      }
      if (!user && navigationRef.isReady()) {
        navigationRef.reset({ index: 0, routes: [{ name: "Login" }] });
      }
    });
    return unsub;
  }, []);

  return (
    <View style={{ flex: 1, backgroundColor: COLORS.background }}>
      <ScreenErrorBoundary>
        <NavigationContainer ref={navigationRef}>
          <Stack.Navigator
            initialRouteName="Splash"
            screenOptions={{
              headerShown: false,
              cardStyleInterpolator: CardStyleInterpolators.forFadeFromCenter,
            }}
          >
            {/* ── Core ── */}
            <Stack.Screen name="Splash" component={SplashScreen} />
            <Stack.Screen name="AppOnboarding" component={AppOnboarding} />
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="SignUp" component={SignUp} />
            <Stack.Screen name="AdminLogin" component={AdminLoginScreen} />
            <Stack.Screen name="AdminHub" component={AdminHubScreen} />
            <Stack.Screen
              name="ResearchDashboard"
              component={ResearchDashboardScreen}
            />
            <Stack.Screen name="Settings" component={SettingsScreen} />
            <Stack.Screen name="MainTabs" component={MainTabNavigator} />
            {/* ── Ishihara Test Flow ── */}
            <Stack.Screen
              name="IshiharaIntro"
              component={IshiharaIntroScreen}
            />
            <Stack.Screen
              name="IshiharaOnboarding"
              component={IshiharaOnboarding}
            />
            <Stack.Screen name="IshiharaTest" component={TestScreen} />
            <Stack.Screen name="IshiharaResult" component={ResultsScreen} />
            {/* ── Feature Screens ── */}
            <Stack.Screen name="Survey" component={SurveyScreen} />
            <Stack.Screen
              name="SurveySuccess"
              component={SurveySuccessScreen}
            />
            <Stack.Screen
              name="CameraEnhance"
              component={CameraEnhanceScreen}
            />
            <Stack.Screen
              name="ColorIdentifier"
              component={ColorIdentifierScreen}
            />
            <Stack.Screen
              name="CVDSimulation"
              component={CVDSimulationScreen}
            />
            <Stack.Screen name="CVDGallery" component={CVDGalleryScreen} />
            <Stack.Screen
              name="EducationList"
              component={EducationListScreen}
            />
            <Stack.Screen name="Article" component={ArticleScreen} />
            <Stack.Screen name="CareerDetail" component={CareerDetailScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </ScreenErrorBoundary>
    </View>
  );
}
