import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { useEffect, useRef } from 'react';
import { auth, onAuthStateChanged } from '../../firebaseConfig';
import { ActivityIndicator, Animated, Image, StyleSheet, Text, View } from 'react-native';
import { COLORS } from '../theme/colors';
import { styles as sharedStyles } from '../theme/styles';

export default function SplashScreen({ navigation }) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 1000, useNativeDriver: true }),
      Animated.spring(slideAnim, { toValue: 0, friction: 6, useNativeDriver: true }),
    ]).start();

    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.1, duration: 3000, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1.0, duration: 3000, useNativeDriver: true }),
      ]),
    ).start();

    let navigated = false;
    const go = async (user) => {
      if (navigated) return;
      navigated = true;
      if (user) { navigation.replace('MainTabs'); return; }
      try {
        const onboarded = await AsyncStorage.getItem('@recolor_onboarded');
        navigation.replace(onboarded ? 'Login' : 'AppOnboarding');
      } catch (_) {
        navigation.replace('Login');
      }
    };

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      unsubscribe();
      setTimeout(() => go(user), 1800);
    });

    // Hard fallback if Firebase auth hangs
    const fallback = setTimeout(() => go(null), 5000);
    return () => clearTimeout(fallback);
  }, []);

  return (
    <View style={sharedStyles.splashContainer}>
      <Animated.View style={[StyleSheet.absoluteFill, { transform: [{ scale: pulseAnim }] }]}>
        <View style={[sharedStyles.bubble, { top: -50, left: -50, width: 220, height: 220, backgroundColor: '#FFCDD2', opacity: 0.5 }]} />
        <View style={[sharedStyles.bubble, { top: 120, left: -90, width: 140, height: 140, backgroundColor: '#E1BEE7', opacity: 0.5 }]} />
        <View style={[sharedStyles.bubble, { bottom: -60, right: -60, width: 280, height: 280, backgroundColor: '#BBDEFB', opacity: 0.5 }]} />
        <View style={[sharedStyles.bubble, { bottom: 220, right: -70, width: 160, height: 160, backgroundColor: '#C8E6C9', opacity: 0.5 }]} />
        <View style={[sharedStyles.bubble, { top: '35%', left: '15%', width: 60, height: 60, backgroundColor: '#FFF9C4', opacity: 0.6 }]} />
        <View style={[sharedStyles.bubble, { top: '60%', right: '10%', width: 40, height: 40, backgroundColor: '#FFCCBC', opacity: 0.6 }]} />
      </Animated.View>

      <Animated.View style={{ alignItems: 'center', opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
        <Image
          source={require('../../assets/logo_stack.png')}
          style={{ width: 220, height: 220, resizeMode: 'contain', marginBottom: 10 }}
        />
        <Text style={[sharedStyles.splashSub, { fontSize: 16, fontWeight: '500' }]}>
          Enhancing Color Perception,{'\n'}One Shade at a Time
        </Text>
      </Animated.View>

      <Animated.View style={{ position: 'absolute', bottom: 50, opacity: fadeAnim }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <ActivityIndicator size="small" color={COLORS.primary} />
          <Text style={{ marginLeft: 10, color: COLORS.primary, fontWeight: 'bold', letterSpacing: 1 }}>
            INITIALIZING...
          </Text>
        </View>
      </Animated.View>
    </View>
  );
}
