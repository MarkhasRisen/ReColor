import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { useEffect, useRef } from 'react';
import { auth, db, onAuthStateChanged } from '../../firebaseConfig';
import { doc, getDoc } from 'firebase/firestore';
import { Animated, Easing, Text, View } from 'react-native';
import BackgroundBubbles from '../components/BackgroundBubbles';
import { COLORS } from '../theme/colors';
import { styles as sharedStyles } from '../theme/styles';

export default function SplashScreen({ navigation }) {
  const logoFade = useRef(new Animated.Value(0)).current;
  const logoScale = useRef(new Animated.Value(0.75)).current;
  const logoSlide = useRef(new Animated.Value(40)).current;

  const textFade = useRef(new Animated.Value(0)).current;
  const textSlide = useRef(new Animated.Value(10)).current;

  const loaderFade = useRef(new Animated.Value(0)).current;
  const progressVal = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // 1. Logo pop & slide
    Animated.parallel([
      Animated.timing(logoFade, { toValue: 1, duration: 700, useNativeDriver: true }),
      Animated.timing(logoScale, { toValue: 1, duration: 700, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
      Animated.timing(logoSlide, { toValue: 0, duration: 700, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();

    // 2. Tagline staggers in
    Animated.parallel([
      Animated.timing(textFade, { toValue: 1, duration: 800, delay: 400, useNativeDriver: true }),
      Animated.timing(textSlide, { toValue: 0, duration: 800, delay: 400, useNativeDriver: true }),
    ]).start();

    // 3. Loader & progress bar fill over 5 seconds
    Animated.timing(loaderFade, { toValue: 1, duration: 600, delay: 800, useNativeDriver: true }).start();
    Animated.timing(progressVal, { toValue: 1, duration: 4600, delay: 200, useNativeDriver: false }).start();

    let navigated = false;
    const go = async (user) => {
      if (navigated) return;
      navigated = true;
      if (user) {
        try {
          const userDocRef = doc(db, "users", user.uid);
          const userSnap = await getDoc(userDocRef);
          if (userSnap.exists()) {
            const role = userSnap.data()?.role;
            if (role === "admin" || role === "researcher") {
              navigation.replace("AdminHub");
              return;
            }
          }
        } catch (error) {
          console.error("Error fetching user profile in splash:", error);
        }
        navigation.replace("MainTabs");
        return;
      }
      try {
        const onboarded = await AsyncStorage.getItem("@recolor_onboarded");
        navigation.replace(onboarded ? "Login" : "AppOnboarding");
      } catch (_) {
        navigation.replace("Login");
      }
    };

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      unsubscribe();
      setTimeout(() => go(user), 5000);
    });

    // Hard fallback if Firebase auth hangs
    const fallback = setTimeout(() => go(null), 8000);
    return () => clearTimeout(fallback);
  }, []);

  return (
    <View style={[sharedStyles.splashContainer, { backgroundColor: COLORS.background }]}>
      <BackgroundBubbles heavy />

      <View style={{ alignItems: 'center' }}>
        <Animated.Image
          source={require('../../assets/logo_stack.png')}
          style={{
            width: 220,
            height: 220,
            resizeMode: 'contain',
            marginBottom: 20,
            opacity: logoFade,
            transform: [{ scale: logoScale }, { translateY: logoSlide }]
          }}
        />
        <Animated.View style={{ alignItems: 'center', opacity: textFade, transform: [{ translateY: textSlide }] }}>
          <Text style={[sharedStyles.splashSub, { fontSize: 15, fontWeight: '700', color: COLORS.text, lineHeight: 22 }]}>
            Enhancing Color Perception,{'\n'}
            <Text style={{ color: COLORS.textLight, fontWeight: '600' }}>One Shade at a Time</Text>
          </Text>
        </Animated.View>
      </View>

      <Animated.View style={{ position: 'absolute', bottom: 65, opacity: loaderFade, alignItems: 'center' }}>
        <Text style={{ color: COLORS.primary, fontWeight: '950', fontSize: 10, letterSpacing: 4, marginBottom: 12 }}>
          INITIALIZING SYSTEM
        </Text>
        
        {/* Horizontal Progress Bar */}
        <View style={{ width: 140, height: 3, backgroundColor: '#E2E8F0', borderRadius: 1.5, overflow: 'hidden' }}>
          <Animated.View
            style={{
              height: '100%',
              backgroundColor: COLORS.primary,
              width: progressVal.interpolate({
                inputRange: [0, 1],
                outputRange: ['0%', '100%']
              })
            }}
          />
        </View>

        <Text style={{ fontSize: 9, color: COLORS.textLight, marginTop: 12, fontWeight: '700', opacity: 0.5, letterSpacing: 1 }}>
          RECOLOR APP • v1.0.0
        </Text>
      </Animated.View>
    </View>
  );
}
