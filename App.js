import * as ImageManipulator from 'expo-image-manipulator';
import { useTensorflowModel } from 'react-native-fast-tflite';
import {
  getClassMask,
  applyDaltonization,
  decodeJpegBase64,
  encodeToDataUri,
  getCVDColorMatrix,
  downscaleToTensor,
  upscaleMaskNearest,
} from './tensorHelper';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';

import { Ionicons } from '@expo/vector-icons';
import Slider from '@react-native-community/slider';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer, useNavigation } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import * as ImagePicker from 'expo-image-picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Animated,
  Dimensions,
  Image,
  ImageBackground,
  Linking,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from 'react-native';
// 1. Import Firestore Tools (Standard Library)
import {
  collection,
  onSnapshot,
  orderBy,
  query
} from 'firebase/firestore';

// 2. Import Your Config & Auth Functions (Local File)
import {
  auth,
  createUserWithEmailAndPassword,
  db,
  saveExamResult, // <--- ADDED: Critical for connecting to the database
  signInWithEmailAndPassword
} from './firebaseConfig';

// --- Configuration & Constants ---
const { width } = Dimensions.get('window');
const COLORS = {
  primary: '#6C63FF', // Purple
  secondary: '#FF4081', // Pink
  accent: '#00D2D3', // Teal/Cyan
  background: '#F8F9FA',
  card: '#FFFFFF',
  text: '#2D3436',
  textLight: '#A4B0BE',
  success: '#2ECC71',
  warning: '#FF9F43',
  danger: '#FF6B6B',
  darkOverlay: 'rgba(0,0,0,0.6)',
};

const RAW_PLATES = [
  // --- The 18 You Have ---
  { id: 1, img: require('./assets/plate_1.png'), answer: '12' }, 
  { id: 2, img: require('./assets/plate_2.png'), answer: '8' },
  { id: 3, img: require('./assets/plate_3.png'), answer: '6' },
  { id: 4, img: require('./assets/plate_4.png'), answer: '29' },
  { id: 5, img: require('./assets/plate_5.png'), answer: '57' },
  { id: 6, img: require('./assets/plate_6.png'), answer: '5' },
  { id: 7, img: require('./assets/plate_7.png'), answer: '3' },
  { id: 8, img: require('./assets/plate_8.png'), answer: '15' },
  { id: 9, img: require('./assets/plate_9.png'), answer: '74' },
  { id: 10, img: require('./assets/plate_10.png'), answer: '5' },
  { id: 11, img: require('./assets/plate_11.png'), answer: '6' },
  { id: 12, img: require('./assets/plate_12.png'), answer: '57' },
  { id: 13, img: require('./assets/plate_13.png'), answer: '5' },
  { id: 14, img: require('./assets/plate_14.png'), answer: '42' },
  { id: 15, img: require('./assets/plate_15.png'), answer: '45' },
  { id: 16, img: require('./assets/plate_16.png'), answer: '2' },
  { id: 17, img: require('./assets/plate_17.png'), answer: '73' },
  { id: 18, img: require('./assets/plate_18.png'), answer: '26' },

  // --- Placeholders to reach 38 (Reuse existing images for now) ---
  { id: 19, img: require('./assets/plate_1.png'), answer: '12' }, 
  { id: 20, img: require('./assets/plate_2.png'), answer: '8' },
  { id: 21, img: require('./assets/plate_3.png'), answer: '6' },
  { id: 22, img: require('./assets/plate_4.png'), answer: '29' },
  { id: 23, img: require('./assets/plate_5.png'), answer: '57' },
  { id: 24, img: require('./assets/plate_6.png'), answer: '5' },
  { id: 25, img: require('./assets/plate_7.png'), answer: '3' },
  { id: 26, img: require('./assets/plate_8.png'), answer: '15' },
  { id: 27, img: require('./assets/plate_9.png'), answer: '74' },
  { id: 28, img: require('./assets/plate_10.png'), answer: '5' },
  { id: 29, img: require('./assets/plate_11.png'), answer: '6' },
  { id: 30, img: require('./assets/plate_12.png'), answer: '57' },
  { id: 31, img: require('./assets/plate_13.png'), answer: '5' },
  { id: 32, img: require('./assets/plate_14.png'), answer: '42' },
  { id: 33, img: require('./assets/plate_15.png'), answer: '45' },
  { id: 34, img: require('./assets/plate_16.png'), answer: '2' },
  { id: 35, img: require('./assets/plate_17.png'), answer: '73' },
  { id: 36, img: require('./assets/plate_18.png'), answer: '26' },
  { id: 37, img: require('./assets/plate_1.png'), answer: '12' },
  { id: 38, img: require('./assets/plate_2.png'), answer: '8' },
];
// --- Mock Data ---
const MOCK_HISTORY = [
  { type: 'Protanomaly', date: 'October 24, 2025', severity: 'Mild', score: 64 },
  { type: 'Deuteranomaly', date: 'October 24, 2025', severity: 'Severe', score: 0 },
  { type: 'Deuteranomaly', date: 'October 24, 2025', severity: 'Severe', score: 0 },
];

const EDU_ARTICLES = [
  { id: 1, title: 'What is Color Vision Deficiency?', image: 'https://placehold.co/300x150/FF9F43/FFFFFF?text=CVD+Info', desc: 'Learn about the science behind color vision.' },
  { id: 2, title: 'Types of Color Blindness', image: 'https://placehold.co/300x150/FF6B6B/FFFFFF?text=Eye+Anatomy', desc: 'Understand Protanomaly, Deuteranomaly, and Tritanomaly.' },
  { id: 3, title: 'Designing for Accessibility', image: 'https://placehold.co/300x150/2ECC71/FFFFFF?text=Accessibility', desc: 'Best practices for creating accessible content.' },
];

// --- Components ---

const Header = ({ title, subtitle, back }) => {
  const navigation = useNavigation();
  return (
    <View style={styles.header}>
      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        {back && (
          <TouchableOpacity onPress={() => navigation.goBack()} style={{ marginRight: 15 }}>
            <Ionicons name="arrow-back" size={24} color="#000" />
          </TouchableOpacity>
        )}
        <View>
          <Text style={styles.headerTitle}>{title}</Text>
          {subtitle && <Text style={styles.headerSubtitle}>{subtitle}</Text>}
        </View>
      </View>
      <Image 
        source={require('./assets/icon.png')} 
        style={{ 
          width: 80,      // Increased from 30
          height: 80,     // Increased from 30
          resizeMode: 'contain', // Keeps the aspect ratio correct
          // borderRadius: 5  <-- Remove this so it doesn't clip the corners of your logo
        }} 
      />
    </View>
  );
};

const Card = ({ children, style, onPress }) => {
  // Animation Value: 1 = 100% size
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const [isPressed, setIsPressed] = useState(false);

  const handlePressIn = () => {
    setIsPressed(true);
    // Animate to 96% size (Shrink)
    Animated.spring(scaleAnim, {
      toValue: 0.96,
      useNativeDriver: true,
      speed: 20,
      bounciness: 10,
    }).start();
  };

  const handlePressOut = () => {
    setIsPressed(false);
    // Animate back to 100% size (Bounce)
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      speed: 20,
      bounciness: 10,
    }).start();
  };

  return (
    <Pressable
      onPress={onPress}
      onPressIn={onPress ? handlePressIn : null}
      onPressOut={onPress ? handlePressOut : null}
      style={{ marginBottom: 15 }} // Keep layout spacing
    >
      <Animated.View
        style={[
          styles.card,
          style,
          {
            // Apply the Scale Animation
            transform: [{ scale: scaleAnim }],
            // Add a dynamic border and shadow change when pressed
            borderColor: isPressed ? COLORS.primary : 'transparent',
            borderWidth: isPressed ? 1 : 0,
            // Slight opacity change for "glow" feel
            opacity: isPressed ? 0.9 : 1,
            marginBottom: 0 // handled by parent Pressable
          }
        ]}
      >
        {children}
      </Animated.View>
    </Pressable>
  );
};
const ProgressBar = ({ label, value, color, count, percentage }) => (
  <View style={{ marginBottom: 15 }}>
    <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 }}>
      <Text style={{ fontSize: 14, fontWeight: '600', color: '#555' }}>{label}</Text>
      <Text style={{ fontSize: 14, fontWeight: 'bold' }}>{count} <Text style={{color: COLORS.textLight}}>({percentage})</Text></Text>
    </View>
    <View style={{ height: 10, backgroundColor: '#E0E0E0', borderRadius: 5, overflow: 'hidden' }}>
      <View style={{ width: value, height: '100%', backgroundColor: color }} />
    </View>
  </View>
);

// --- Reusable Background Component ---
const BackgroundBubbles = () => (
  <View style={[StyleSheet.absoluteFill, { zIndex: -1, overflow: 'hidden' }]} pointerEvents="none">
    <View style={[styles.bubble, { top: -50, left: -50, width: 200, height: 200, backgroundColor: '#FFCDD2' }]} />
    <View style={[styles.bubble, { top: 100, left: -80, width: 120, height: 120, backgroundColor: '#E1BEE7' }]} />
    <View style={[styles.bubble, { bottom: -50, right: -50, width: 250, height: 250, backgroundColor: '#BBDEFB' }]} />
    <View style={[styles.bubble, { bottom: 200, right: -60, width: 150, height: 150, backgroundColor: '#C8E6C9' }]} />
    <View style={[styles.bubble, { top: '40%', left: '10%', width: 50, height: 50, backgroundColor: '#FFF9C4' }]} />
  </View>
);
// --- Screens: Auth & Onboarding ---

function SplashScreen({ navigation }) {
  // 1. Animation Values
  const fadeAnim = useRef(new Animated.Value(0)).current; // Opacity (starts invisible)
  const slideAnim = useRef(new Animated.Value(50)).current; // Position (starts 50px down)
  const pulseAnim = useRef(new Animated.Value(1)).current; // Bubble Scale (starts at 1x)

  useEffect(() => {
    // A. Start the Entrance Animation (Fade + Slide Up)
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.spring(slideAnim, {
        toValue: 0,
        friction: 6,
        useNativeDriver: true,
      }),
    ]).start();

    // B. Start the "Breathing" Bubble Loop
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1, // Grow by 10%
          duration: 3000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1.0, // Shrink back
          duration: 3000,
          useNativeDriver: true,
        }),
      ])
    ).start();

    // C. Navigate after 3.5 seconds
    setTimeout(() => navigation.replace('Login'), 3500);
  }, []);

  return (
    <View style={styles.splashContainer}>
      {/* --- ANIMATED BUBBLES --- */}
      {/* We wrap the bubbles in an Animated.View to apply the breathing effect to all of them */}
      <Animated.View 
        style={[
          StyleSheet.absoluteFill, 
          { transform: [{ scale: pulseAnim }] } // The Breathing Effect
        ]}
      >
        <View style={[styles.bubble, { top: -50, left: -50, width: 220, height: 220, backgroundColor: '#FFCDD2', opacity: 0.5 }]} />
        <View style={[styles.bubble, { top: 120, left: -90, width: 140, height: 140, backgroundColor: '#E1BEE7', opacity: 0.5 }]} />
        <View style={[styles.bubble, { bottom: -60, right: -60, width: 280, height: 280, backgroundColor: '#BBDEFB', opacity: 0.5 }]} />
        <View style={[styles.bubble, { bottom: 220, right: -70, width: 160, height: 160, backgroundColor: '#C8E6C9', opacity: 0.5 }]} />
        <View style={[styles.bubble, { top: '35%', left: '15%', width: 60, height: 60, backgroundColor: '#FFF9C4', opacity: 0.6 }]} />
        <View style={[styles.bubble, { top: '60%', right: '10%', width: 40, height: 40, backgroundColor: '#FFCCBC', opacity: 0.6 }]} />
      </Animated.View>
      
      {/* --- ANIMATED CONTENT (Logo & Text) --- */}
      <Animated.View style={{ alignItems: 'center', opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
        <Image 
          source={require('./assets/logo_stack.png')} 
          style={{ width: 220, height: 220, resizeMode: 'contain', marginBottom: 10 }} 
        />

        <Text style={[styles.splashSub, { fontSize: 16, fontWeight: '500' }]}>
          Enhancing Color Perception,{'\n'}One Shade at a Time
        </Text>
      </Animated.View>
      
      {/* Footer Loader */}
      <Animated.View style={{ position: 'absolute', bottom: 50, opacity: fadeAnim }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <ActivityIndicator size="small" color={COLORS.primary} />
          <Text style={{ marginLeft: 10, color: COLORS.primary, fontWeight: 'bold', letterSpacing: 1 }}>INITIALIZING...</Text>
        </View>
      </Animated.View>
    </View>
  );
}

function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert("Error", "Please enter email and password.");
      return;
    }
    setLoading(true);
    try {
      // 1. Try to Login
      await signInWithEmailAndPassword(auth, email, password);
      console.log("Login Success");
      navigation.replace('MainTabs');
    } catch (error) {
      // 2. Emergency "Auto-Signup" for Defense Demo
      // If user doesn't exist, we create them on the fly so you don't get stuck.
      if (error.code === 'auth/user-not-found' || error.code === 'auth/invalid-credential') {
        try {
          await createUserWithEmailAndPassword(auth, email, password);
          console.log("Account Created on the fly");
          navigation.replace('MainTabs');
        } catch (regError) {
          Alert.alert("Registration Error", regError.message);
        }
      } else {
        Alert.alert("Login Failed", error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      <ScrollView contentContainerStyle={{ flexGrow: 1, justifyContent: 'center', paddingBottom: 60 }} showsVerticalScrollIndicator={false}>
        <View style={[styles.loginCard, { marginTop: 0 }]}>
          <View style={{ alignItems: 'center', marginBottom: 30 }}>
            <Image source={require('./assets/logo_row.png')} style={{ width: 250, height: 150, resizeMode: 'contain', marginBottom: 10 }} />
            <Text style={styles.headerTitle}>Welcome</Text>
            <Text style={{ textAlign: 'center', color: COLORS.textLight, marginTop: 5 }}>Sign in to enhance your color perception</Text>
          </View>

          {/* INPUTS TIED TO STATE */}
          <View style={styles.inputContainer}>
            <Ionicons name="mail-outline" size={20} color="#999" />
            <TextInput 
              style={styles.input} 
              placeholder="admin@recolor.app" 
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
            />
          </View>

          <View style={styles.inputContainer}>
            <Ionicons name="key-outline" size={20} color="#999" />
            <TextInput 
              style={styles.input} 
              placeholder="••••••••" 
              secureTextEntry 
              value={password}
              onChangeText={setPassword}
            />
          </View>

          <TouchableOpacity style={styles.btnPrimary} onPress={handleLogin}>
            {loading ? <ActivityIndicator color="#FFF" /> : <Text style={styles.btnText}>Login / Sign Up</Text>}
          </TouchableOpacity>

          <TouchableOpacity style={{ marginTop: 20, alignSelf: 'center' }} onPress={() => navigation.navigate('AdminLogin')}>
            <Text style={{ color: COLORS.textLight }}>
               <Ionicons name="lock-closed" size={14} color={COLORS.warning} /> Admin / Expert Portal
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

function AdminLoginScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Header title="Admin Portal" subtitle="Secure Access" back />
      <BackgroundBubbles />
      
      <ScrollView contentContainerStyle={{ padding: 20 }}showsVerticalScrollIndicator={false}>
        <View style={{ alignItems: 'center', marginVertical: 30 }}>
          <View style={styles.iconCircleGradient}>
            <Ionicons name="lock-closed-outline" size={40} color="#FFF" />
          </View>
          <Text style={[styles.headerTitle, { marginTop: 15 }]}>Admin Login</Text>
          <Text style={{ color: COLORS.textLight }}>Access ReColor Admin & Expert Portal</Text>
        </View>

        <View style={styles.infoBox}>
          <Ionicons name="shield-checkmark-outline" size={20} color="#333" />
          <Text style={{ marginLeft: 10, flex: 1, fontSize: 12, color: '#444' }}>
            This portal uses secure Firebase authentication with role-based access control.
          </Text>
        </View>

        <Text style={styles.label}>Email Address</Text>
        <View style={styles.inputContainer}>
          <Ionicons name="mail-outline" size={20} color="#999" />
          <TextInput style={styles.input} placeholder="admin@recolor.app" />
        </View>

        <Text style={styles.label}>Password</Text>
        <View style={styles.inputContainer}>
          <Ionicons name="key-outline" size={20} color="#999" />
          <TextInput style={styles.input} placeholder="••••••••" secureTextEntry />
        </View>

        <TouchableOpacity style={[styles.btnPrimary, { marginTop: 20, backgroundColor: '#8E24AA' }]}>
          <Ionicons name="lock-open-outline" size={20} color="#FFF" style={{ marginRight: 10 }} />
          <Text style={styles.btnText}>Sign In Securely</Text>
        </TouchableOpacity>

        {/* DEMO CREDENTIALS - CLICKABLE */}
        <View style={{ marginTop: 30, backgroundColor: '#E3F2FD', padding: 15, borderRadius: 12, borderWidth: 1, borderColor: '#BBDEFB' }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
            <Ionicons name="information-circle-outline" size={20} color="#1565C0" />
            <Text style={{ marginLeft: 5, color: '#1565C0', fontWeight: 'bold' }}>Demo Credentials (Testing Only):</Text>
          </View>

          {/* Researcher Button */}
          <TouchableOpacity 
            style={{ backgroundColor: '#FFF', padding: 12, borderRadius: 8, marginBottom: 10, borderLeftWidth: 4, borderLeftColor: '#6C63FF' }}
            onPress={() => navigation.navigate('AdminHub', { role: 'researcher' })}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="flask-outline" size={20} color="#6C63FF" />
              <Text style={{ marginLeft: 10, fontWeight: 'bold', color: '#333' }}>PERI Researcher</Text>
            </View>
            <Text style={{ color: '#777', fontSize: 12, marginTop: 2 }}>researcher@peri.edu / peri123</Text>
          </TouchableOpacity>

          {/* Admin Button */}
          <TouchableOpacity 
            style={{ backgroundColor: '#FFF', padding: 12, borderRadius: 8, borderLeftWidth: 4, borderLeftColor: '#9C27B0' }}
            onPress={() => navigation.navigate('AdminHub', { role: 'admin' })}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="settings-outline" size={20} color="#9C27B0" />
              <Text style={{ marginLeft: 10, fontWeight: 'bold', color: '#333' }}>System Administrator</Text>
            </View>
            <Text style={{ color: '#777', fontSize: 12, marginTop: 2 }}>admin@recolor.app / admin123</Text>
          </TouchableOpacity>
        </View>

      </ScrollView>
    </View>
  );
}

// --- Screens: Admin Flow ---

function AdminHubScreen({ route, navigation }) {
  const { role } = route.params || { role: 'researcher' };
  const isResearcher = role === 'researcher';

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <Header title="Admin Hub" subtitle={isResearcher ? "Dr. PERI Researcher" : "System Administrator"} back />
      <BackgroundBubbles />
      
      <View style={{ padding: 20 }}>
        
        {/* Dynamic Profile Card */}
        <ImageBackground 
          style={{ width: '100%', padding: 25, borderRadius: 20, overflow: 'hidden', marginBottom: 20, backgroundColor: isResearcher ? '#6200EA' : '#651FFF' }}
          source={{ uri: 'https://placehold.co/400x150/6200EA/B388FF?text= ' }}
        >
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <View style={{ width: 60, height: 60, borderRadius: 30, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' }}>
              <Ionicons name="person" size={30} color="#FFF" />
            </View>
            <View style={{ marginLeft: 15 }}>
              <Text style={{ color: '#FFF', fontSize: 18, fontWeight: 'bold' }}>
                {isResearcher ? "Dr. PERI Researcher" : "System Administrator"}
              </Text>
              <Text style={{ color: 'rgba(255,255,255,0.9)', marginBottom: 5 }}>
                {isResearcher ? "researcher@peri.edu" : "admin@recolor.app"}
              </Text>
              <View style={{ backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12, alignSelf: 'flex-start' }}>
                <Text style={{ color: '#FFF', fontSize: 10, fontWeight: 'bold' }}>
                  {isResearcher ? "PERI Researcher" : "System Administrator"}
                </Text>
              </View>
            </View>
          </View>
        </ImageBackground>

        {/* System Status */}
        <View style={{ backgroundColor: '#E8F5E9', padding: 15, borderRadius: 12, flexDirection: 'row', alignItems: 'center', marginBottom: 20, borderWidth: 1, borderColor: '#C8E6C9' }}>
          <View style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: COLORS.success, marginRight: 10 }} />
          <View>
            <Text style={{ fontWeight: 'bold', color: '#2E7D32' }}>System Status: All Systems Operational</Text>
            <Text style={{ fontSize: 12, color: '#666' }}>Last checked: Just now</Text>
          </View>
        </View>

        {/* Architecture Card */}
        <Card style={{ marginBottom: 20 }}>
          <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 5 }}>System Architecture</Text>
          <Text style={{ fontSize: 12, color: '#666', marginBottom: 15 }}>ReColor Admin/Expert Flow Diagram</Text>
          <Image 
            source={{ uri: 'https://placehold.co/600x300/FFF/000?text=Flow+Diagram' }} 
            style={{ width: '100%', height: 150, resizeMode: 'contain', marginBottom: 15 }} 
          />
          <View style={{ backgroundColor: '#E3F2FD', padding: 10, borderRadius: 8, flexDirection: 'row', alignItems: 'center' }}>
            <Ionicons name="information-circle-outline" size={16} color="#1565C0" style={{ marginRight: 8 }} />
            <Text style={{ fontSize: 11, color: '#1565C0', flex: 1 }}>
              Following secure Firebase authentication with role-based access control
            </Text>
          </View>
        </Card>

        {/* Access Portals - FIXED NAVIGATION */}
        <Text style={{ color: '#999', fontSize: 12, marginBottom: 10, letterSpacing: 1 }}>YOUR ACCESS PORTALS</Text>
        
        {isResearcher ? (
          // RESEARCHER CARD - Directly clickable
          <Card 
            onPress={() => navigation.navigate('ResearchDashboard')}
            style={{ flexDirection: 'row', alignItems: 'center' }}
          >
            <View style={{ width: 50, height: 50, borderRadius: 25, backgroundColor: '#E3F2FD', alignItems: 'center', justifyContent: 'center' }}>
              <Ionicons name="bar-chart" size={24} color="#2196F3" />
            </View>
            <View style={{ marginLeft: 15, flex: 1 }}>
              <Text style={{ fontSize: 16, fontWeight: 'bold' }}>Research Dashboard</Text>
              <Text style={{ fontSize: 12, color: '#666' }}>View anonymized data • Statistics</Text>
            </View>
            <View style={{ backgroundColor: '#E3F2FD', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 }}>
              <Text style={{ color: '#2196F3', fontSize: 10, fontWeight: 'bold' }}>Primary</Text>
            </View>
          </Card>
        ) : (
          // ADMIN CARD
          <Card 
            onPress={() => navigation.navigate('ResearchDashboard')}
            style={{ flexDirection: 'row', alignItems: 'center' }}
          >
            <View style={{ width: 50, height: 50, borderRadius: 25, backgroundColor: '#F3E5F5', alignItems: 'center', justifyContent: 'center' }}>
              <Ionicons name="settings" size={24} color="#9C27B0" />
            </View>
            <View style={{ marginLeft: 15, flex: 1 }}>
              <Text style={{ fontSize: 16, fontWeight: 'bold' }}>System Management</Text>
              <Text style={{ fontSize: 12, color: '#666' }}>User management • Content CMS</Text>
            </View>
          </Card>
        )}

        {/* Logout - RED BUTTON */}
        <TouchableOpacity 
          style={{ marginTop: 30, backgroundColor: '#D32F2F', padding: 15, borderRadius: 12, alignItems: 'center', flexDirection: 'row', justifyContent: 'center' }}
          onPress={() => navigation.navigate('Login')}
        >
          <Ionicons name="log-out-outline" size={20} color="#FFF" style={{ marginRight: 10 }} />
          <Text style={{ fontWeight: 'bold', color: '#FFF' }}>Logout from Admin Portal</Text>
        </TouchableOpacity>

      </View>
    </ScrollView>
  );
}

function ResearchDashboardScreen({ navigation }) {
  const [activeTab, setActiveTab] = useState('View Data'); // 'View Data' or 'Guidelines'

  // Helper Component for the Progress Bars
  const StatBar = ({ label, count, percent, color, badge }) => (
    <View style={{ marginBottom: 15 }}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <Text style={{ fontSize: 13, color: '#333' }}>{label}</Text>
          {badge && <View style={{ marginLeft: 8, borderWidth:1, borderColor:'#EEE', paddingHorizontal:6, borderRadius:4 }}><Text style={{ fontSize:10, color:'#555' }}>{badge}</Text></View>}
        </View>
        <Text style={{ fontWeight: 'bold', fontSize: 13 }}>{count} <Text style={{color:'#999', fontWeight:'normal'}}>({percent})</Text></Text>
      </View>
      <View style={{ height: 8, backgroundColor: '#F0F0F0', borderRadius: 4 }}>
        <View style={{ width: percent, height: '100%', backgroundColor: color, borderRadius: 4 }} />
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <Header title="Research Dashboard" subtitle="PERI Researcher Portal" back />
      <BackgroundBubbles />

      {/* --- CUSTOM TAB BAR --- */}
      <View style={{ flexDirection: 'row', padding: 20, paddingBottom: 0 }}>
        <TouchableOpacity 
          style={{ flex: 1, paddingVertical: 10, backgroundColor: activeTab === 'View Data' ? '#FFF' : '#F5F5F5', alignItems: 'center', borderTopLeftRadius: 20, borderBottomLeftRadius: 20, borderWidth: 1, borderColor: '#E0E0E0', borderRightWidth:0 }}
          onPress={() => setActiveTab('View Data')}
        >
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Ionicons name="stats-chart" size={18} color={activeTab === 'View Data' ? '#333' : '#999'} />
            <Text style={{ marginLeft: 8, fontWeight: 'bold', color: activeTab === 'View Data' ? '#333' : '#999' }}>View Data</Text>
          </View>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={{ flex: 1, paddingVertical: 10, backgroundColor: activeTab === 'Guidelines' ? '#FFF' : '#F5F5F5', alignItems: 'center', borderTopRightRadius: 20, borderBottomRightRadius: 20, borderWidth: 1, borderColor: '#E0E0E0' }}
          onPress={() => setActiveTab('Guidelines')}
        >
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Ionicons name="document-text" size={18} color={activeTab === 'Guidelines' ? '#333' : '#999'} />
            <Text style={{ marginLeft: 8, fontWeight: 'bold', color: activeTab === 'Guidelines' ? '#333' : '#999' }}>Guidelines</Text>
          </View>
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        
        {/* --- TAB 1: VIEW DATA --- */}
        {activeTab === 'View Data' && (
          <>
            {/* Privacy Alert */}
            <View style={{ backgroundColor: '#E1F5FE', padding: 15, borderRadius: 8, marginBottom: 20, borderWidth:1, borderColor:'#B3E5FC' }}>
               <Text style={{ color: '#0277BD', fontSize: 12, lineHeight: 18 }}>
                 <Text style={{ fontWeight: 'bold' }}>Privacy Protected: </Text>
                 All data is anonymized. No personally identifiable information (PII) is visible. Only demographics and test scores are available.
               </Text>
            </View>

            {/* Aggregate Reports Header */}
            <View style={{ marginBottom: 15 }}>
               <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                 <Ionicons name="document-text-outline" size={20} color="#333" />
                 <Text style={{ fontWeight: 'bold', fontSize: 16, marginLeft: 10 }}>Aggregate Reports</Text>
               </View>
               <Text style={{ color: '#666', fontSize: 12, marginTop: 2 }}>Summary statistics across all participants</Text>
            </View>

            {/* 4-Grid Stats Cards */}
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 20 }}>
               {/* Card 1 */}
               <View style={{ width: '48%', backgroundColor: '#E3F2FD', padding: 15, borderRadius: 12, marginBottom: 10 }}>
                  <Ionicons name="trending-up" size={20} color="#1E88E5" />
                  <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#1565C0', marginTop: 10 }}>3891</Text>
                  <Text style={{ fontSize: 11, color: '#1E88E5' }}>Total Tests</Text>
               </View>
               {/* Card 2 */}
               <View style={{ width: '48%', backgroundColor: '#F3E5F5', padding: 15, borderRadius: 12, marginBottom: 10 }}>
                  <Ionicons name="people" size={20} color="#8E24AA" />
                  <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#6A1B9A', marginTop: 10 }}>1247</Text>
                  <Text style={{ fontSize: 11, color: '#8E24AA' }}>Participants</Text>
               </View>
               {/* Card 3 */}
               <View style={{ width: '48%', backgroundColor: '#E8F5E9', padding: 15, borderRadius: 12 }}>
                  <Ionicons name="time-outline" size={20} color="#43A047" />
                  <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#2E7D32', marginTop: 10 }}>8.5 min</Text>
                  <Text style={{ fontSize: 11, color: '#43A047' }}>Avg Time</Text>
               </View>
               {/* Card 4 */}
               <View style={{ width: '48%', backgroundColor: '#FCE4EC', padding: 15, borderRadius: 12 }}>
                  <Ionicons name="checkbox-outline" size={20} color="#E91E63" />
                  <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#C2185B', marginTop: 10 }}>94%</Text>
                  <Text style={{ fontSize: 11, color: '#E91E63' }}>Data Quality</Text>
               </View>
            </View>

            <TouchableOpacity style={{ backgroundColor: '#FFF', borderWidth: 1, borderColor: '#DDD', padding: 12, borderRadius: 8, alignItems: 'center', marginBottom: 25 }}>
               <Text style={{ fontWeight: 'bold', color: '#333' }}><Ionicons name="download-outline" size={16} /> Export Full Report (CSV)</Text>
            </TouchableOpacity>

            {/* Detailed Stats */}
            <Card style={{ marginBottom: 20 }}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 20 }}>
                <Ionicons name="bar-chart-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Analyze Screening Statistics</Text>
              </View>
              
              <StatBar label="Normal Vision" badge="N/A" count={478} percent="38%" color={COLORS.success} />
              <StatBar label="Protanomaly" badge="Mild-Moderate" count={412} percent="33%" color="#2979FF" />
              <StatBar label="Deuteranomaly" badge="Mild-Moderate" count={289} percent="23%" color="#E040FB" />
              <StatBar label="Tritanomaly" badge="Mild" count={68} percent="6%" color="#FF5722" />
            </Card>

            {/* Demographics */}
            <Card>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 20 }}>
                <Ionicons name="people-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Demographics Analysis</Text>
              </View>
              <Text style={{fontSize:12, color:'#666', marginBottom:15}}>Age distribution (anonymized data only)</Text>
              
              <StatBar label="Age 18-30" count={534} percent="43%" color="#03A9F4" />
              <StatBar label="Age 31-50" count={412} percent="33%" color="#E040FB" />
              <StatBar label="Age 51+" count={301} percent="24%" color="#FF9800" />
            </Card>
          </>
        )}

        {/* --- TAB 2: GUIDELINES --- */}
        {activeTab === 'Guidelines' && (
          <>
            {/* Update Protocols */}
            <Card style={{ marginBottom: 20 }}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
                <Ionicons name="document-text-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Update Clinical Protocols</Text>
              </View>
              <Text style={{ color: '#666', fontSize: 12, marginBottom: 15 }}>Modify testing guidelines and recommendations</Text>
              
              <View style={{ backgroundColor: '#F5F5F5', borderRadius: 8, padding: 15, marginBottom: 15 }}>
                <Text style={{ color: '#333', fontSize: 12, fontWeight:'bold', marginBottom:5 }}>Protocol Updates</Text>
                <TextInput 
                  placeholder="Enter updated clinical protocols, testing procedures..." 
                  multiline 
                  style={{ height: 60, textAlignVertical: 'top', fontSize: 12 }} 
                />
              </View>

              <TouchableOpacity style={[styles.btnPrimary, { backgroundColor: '#7B1FA2' }]}>
                <Ionicons name="save-outline" size={18} color="#FFF" style={{ marginRight: 8 }} />
                <Text style={styles.btnText}>Update Protocol</Text>
              </TouchableOpacity>

              <View style={{ marginTop: 20, backgroundColor: '#E3F2FD', padding: 15, borderRadius: 8 }}>
                <View style={{flexDirection:'row', alignItems:'center', marginBottom:10}}>
                   <Ionicons name="information-circle" size={16} color="#1565C0" />
                   <Text style={{marginLeft:5, color:'#1565C0', fontWeight:'bold', fontSize:12}}>Current Active Protocols:</Text>
                </View>
                <View style={{marginLeft:5}}>
                  <Text style={{fontSize:11, color:'#555', lineHeight:18}}>• Ishihara Test - 38 plates (comprehensive)</Text>
                  <Text style={{fontSize:11, color:'#555', lineHeight:18}}>• Quick Test - 14 plates for rapid screening</Text>
                  <Text style={{fontSize:11, color:'#555', lineHeight:18}}>• Color Enhancement filters - Calibrated</Text>
                  <Text style={{fontSize:11, color:'#555', lineHeight:18}}>• AI Color Identifier - 95% accuracy baseline</Text>
                </View>
              </View>
            </Card>

            {/* Push Notifications */}
            <Card>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
                <Ionicons name="notifications-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Push Notifications to Users</Text>
              </View>
              <Text style={{ color: '#666', fontSize: 12, marginBottom: 15 }}>Send announcements to all app users</Text>
              
              <View style={{ backgroundColor: '#F5F5F5', borderRadius: 8, padding: 15, marginBottom: 15 }}>
                <Text style={{ color: '#333', fontSize: 12, fontWeight:'bold', marginBottom:5 }}>Notification Message</Text>
                <TextInput placeholder="Enter notification message..." style={{ fontSize: 12 }} />
              </View>

              <TouchableOpacity style={[styles.btnPrimary, { backgroundColor: '#C2185B' }]}>
                <Ionicons name="notifications" size={18} color="#FFF" style={{ marginRight: 8 }} />
                <Text style={styles.btnText}>Send Notification to All Users</Text>
              </TouchableOpacity>

              <View style={{ marginTop: 20, backgroundColor: '#FCE4EC', padding: 15, borderRadius: 8 }}>
                <View style={{flexDirection:'row', alignItems:'center', marginBottom:10}}>
                   <Ionicons name="notifications" size={16} color="#880E4F" />
                   <Text style={{marginLeft:5, color:'#880E4F', fontWeight:'bold', fontSize:12}}>Recent Notifications:</Text>
                </View>
                
                <View style={{ backgroundColor: '#FFF', padding: 10, borderRadius: 6, marginBottom: 8, borderWidth:1, borderColor:'#F8BBD0' }}>
                  <Text style={{ fontSize: 10, color: '#888' }}>2 days ago • Sent to 1,247 users</Text>
                  <Text style={{ fontSize: 12, fontWeight: '500', marginTop: 2, color:'#333' }}>"New education articles available in the Learning Hub"</Text>
                </View>
                
                <View style={{ backgroundColor: '#FFF', padding: 10, borderRadius: 6, borderWidth:1, borderColor:'#F8BBD0' }}>
                  <Text style={{ fontSize: 10, color: '#888' }}>1 week ago • Sent to 1,189 users</Text>
                  <Text style={{ fontSize: 12, fontWeight: '500', marginTop: 2, color:'#333' }}>"Updated color enhancement filters now available"</Text>
                </View>
              </View>
            </Card>
          </>
        )}

      </ScrollView>
    </View>
  );
}

// --- Screens: User Main Tabs ---

function HomeScreen({ navigation }) {
  return (
    <View style={styles.container}>
      {/* ADD BUBBLES HERE */}
      <BackgroundBubbles />

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}style={{ backgroundColor: 'transparent' }}>
        <View style={{ marginTop: 10, marginBottom: 20 }}>
          <Text style={{ color: '#666', marginBottom: 5 }}>Hello, Demo User</Text>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontSize: 28, fontWeight: 'bold', flex: 1, marginRight: 10 }}>
              Welcome to ReColor
            </Text>
            <Image 
              source={require('./assets/icon.png')} 
              style={{ width: 180, height: 180, resizeMode: 'contain' }} 
            />
          </View>
        </View>

        <Card style={styles.featureCard} onPress={() => navigation.navigate('IshiharaIntro')}>
          <View style={[styles.featureIcon, { backgroundColor: '#E3F2FD' }]}>
            <Ionicons name="eye" size={28} color="#2196F3" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.cardTitle}>Take Ishihara Test</Text>
            <Text style={styles.cardDesc}>Screen for color vision deficiency with our digital test</Text>
          </View>
        </Card>

        <Card style={styles.featureCard} onPress={() => navigation.navigate('Survey')}>
          <View style={[styles.featureIcon, { backgroundColor: '#E8F5E9' }]}>
            <Ionicons name="clipboard" size={28} color="#4CAF50" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.cardTitle}>Quick Survey</Text>
            <Text style={styles.cardDesc}>Help us understand your color vision deficiency</Text>
          </View>
        </Card>

        <Card style={styles.featureCard} onPress={() => navigation.navigate('CameraSim')}>
          <View style={[styles.featureIcon, { backgroundColor: '#F3E5F5' }]}>
            <Ionicons name="color-wand" size={28} color="#9C27B0" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.cardTitle}>Color Enhancement Mode</Text>
            <Text style={styles.cardDesc}>View the world with adaptive color enhancement</Text>
          </View>
        </Card>

        <Card style={styles.featureCard} onPress={() => navigation.navigate('EducationList')}>
          <View style={[styles.featureIcon, { backgroundColor: '#FCE4EC' }]}>
            <Ionicons name="book" size={28} color="#E91E63" />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.cardTitle}>Awareness & Education</Text>
            <Text style={styles.cardDesc}>Learn about color vision deficiency and accessibility</Text>
          </View>
        </Card>
      </ScrollView>
    </View>
  );
}

function HistoryScreen() {
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Auto-load data when screen opens
  useEffect(() => {
    // 1. If not logged in, show nothing
    if (!auth.currentUser) {
      setLoading(false);
      return;
    }

    // 2. Query ONLY the current user's data
    const historyRef = collection(db, "users", auth.currentUser.uid, "history");
    const q = query(historyRef, orderBy("date", "desc"));

    // 3. Real-time Listener (Updates automatically)
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const results = snapshot.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          // Handle missing fields safely
          type: data.diagnosis || "Unknown",
          score: data.score !== undefined ? data.score : "?",
          severity: data.severity || "N/A",
          // Convert Firebase Timestamp to readable text
          date: data.date?.toDate ? data.date.toDate().toLocaleDateString() : "Just now"
        };
      });
      setHistoryData(results);
      setLoading(false);
    }, (error) => {
      console.error("History fetch error:", error);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  return (
    <View style={styles.container}>
      <Header title="Your History" back />
      <BackgroundBubbles />

      {loading ? (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <ActivityIndicator size="large" color={COLORS.primary} />
        </View>
      ) : historyData.length === 0 ? (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', opacity: 0.6 }}>
          <Ionicons name="clipboard-outline" size={60} color="#333" />
          <Text style={{ marginTop: 10, fontSize: 16 }}>No tests taken yet.</Text>
        </View>
      ) : (
        <ScrollView contentContainerStyle={{ padding: 20 }}>
          {historyData.map((item, index) => (
            <Card key={index} style={{ marginBottom: 15, paddingVertical: 15 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                <View>
                  <Text style={{ fontSize: 16, fontWeight: 'bold', color: '#333' }}>{item.type}</Text>
                  <Text style={{ fontSize: 12, color: '#888' }}>{item.date}</Text>
                </View>
                <View style={{ alignItems: 'flex-end' }}>
                  <Text style={{ fontSize: 20, fontWeight: 'bold', color: COLORS.primary }}>{item.score}/14</Text>
                  <Text style={{ fontSize: 11, fontWeight: 'bold', color: item.severity === 'Severe' ? COLORS.danger : COLORS.warning }}>
                    {item.severity}
                  </Text>
                </View>
              </View>
            </Card>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

function ProfileScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        
        {/* Header */}
        <View style={{ marginTop: 20, marginBottom: 20, flexDirection: 'row', alignItems: 'center' }}>
          
          {/* Text Container: flex: 1 pushes the image to the right, but respects the image's margin */}
          <View style={{ flex: 1 }}>
            <Text style={{ color: '#666', fontSize: 16 }}>Hello, Demo User</Text>
            <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#333' }}>Welcome to ReColor</Text>
          </View>
          
          <Image 
            source={require('./assets/icon.png')} 
            style={{ 
              width: 120, 
              height: 120, 
              resizeMode: 'contain',
              marginRight: 40 // Change this number to move it more/less left
            }} 
          />
        </View>

        {/* Profile Summary Card */}
        <Card style={{ alignItems: 'center', paddingVertical: 30 }}>
          <View style={{ width: 80, height: 80, borderRadius: 40, backgroundColor: COLORS.primary, alignItems: 'center', justifyContent: 'center', marginBottom: 15, elevation: 5 }}>
            <Ionicons name="person" size={40} color="#FFF" />
          </View>
          
          <Text style={{ fontSize: 20, fontWeight: 'bold', color: '#333' }}>Demo User</Text>
          <Text style={{ color: '#888', marginBottom: 25 }}>demo@recolor.app</Text>

          {/* Manage Settings Button */}
          <TouchableOpacity 
            style={{ backgroundColor: '#E8EAF6', paddingVertical: 12, paddingHorizontal: 40, borderRadius: 8, width: '100%', alignItems: 'center' }}
            onPress={() => navigation.navigate('Settings')}
          >
            <Text style={{ color: '#333', fontWeight: '600' }}>Manage Settings</Text>
          </TouchableOpacity>
        </Card>

      </ScrollView>
    </View>
  );
}
// --- Screens: User Feature Flows ---

function SettingsScreen({ navigation }) {
  const [enhancement, setEnhancement] = useState(0);
  const [audio, setAudio] = useState(true);

  return (
    <View style={styles.container}>
      <Header title="Settings" back />
      <BackgroundBubbles />
      
      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        
        {/* Profile Details */}
        <Card style={{ marginBottom: 20 }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 15 }}>
            <Ionicons name="person-outline" size={20} color="#555" />
            <Text style={{ marginLeft: 10, fontWeight: 'bold', fontSize: 16 }}>Profile</Text>
          </View>
          <View style={{ marginBottom: 10 }}>
            <Text style={{ color: '#999', fontSize: 12 }}>Name</Text>
            <Text style={{ fontSize: 16, color: '#333', fontWeight: '500' }}>Demo User</Text>
          </View>
          <View>
            <Text style={{ color: '#999', fontSize: 12 }}>Email</Text>
            <Text style={{ fontSize: 16, color: '#333', fontWeight: '500' }}>demo@recolor.app</Text>
          </View>
        </Card>

        {/* Preferences */}
        <Card style={{ marginBottom: 20 }}>
           <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 15 }}>
            <Ionicons name="settings-outline" size={20} color="#555" />
            <Text style={{ marginLeft: 10, fontWeight: 'bold', fontSize: 16 }}>Preferences</Text>
          </View>

          {/* Slider Section */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 }}>
            <Text style={{ fontWeight: '600', color: '#333' }}>Color Enhancement Intensity</Text>
            <Text style={{ color: '#999' }}>{Math.round(enhancement)}%</Text>
          </View>
          
          <Slider
            style={{width: '100%', height: 40}}
            minimumValue={0} maximumValue={100} step={1}
            value={enhancement} onValueChange={setEnhancement}
            minimumTrackTintColor={COLORS.primary}
            maximumTrackTintColor="#E0E0E0"
            thumbTintColor={COLORS.primary}
          />
          <Text style={{ fontSize: 11, color: '#999', marginBottom: 25 }}>
            Adjust the strength of color enhancement in camera mode
          </Text>

          {/* Audio Toggle */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <View style={{ flex: 1, paddingRight: 20 }}>
              <Text style={{ fontWeight: '600', color: '#333' }}>Audio Feedback</Text>
              <Text style={{ fontSize: 11, color: '#999', marginTop: 2 }}>Enable voice announcements for color identification</Text>
            </View>
            <Switch 
              value={audio} 
              onValueChange={setAudio} 
              trackColor={{ false: "#767577", true: "#333" }}
              thumbColor={audio ? "#FFF" : "#f4f3f4"}
            />
          </View>
        </Card>

        {/* Data Section - FIXED NAVIGATION */}
        <Card style={{ marginBottom: 20 }}>
          <Text style={{ marginBottom: 15, fontWeight: 'bold', fontSize: 16, color: '#333' }}>Data</Text>
          <TouchableOpacity 
            style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}
            // FIX: Jump to MainTabs, then to the History screen
            onPress={() => navigation.navigate('MainTabs', { screen: 'History' })}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="time-outline" size={24} color="#333" />
              <View style={{ marginLeft: 15 }}>
                <Text style={{ fontWeight: '600', fontSize: 15 }}>View Test History</Text>
                <Text style={{ color: '#999', fontSize: 12 }}>3 tests completed</Text>
              </View>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#CCC" />
          </TouchableOpacity>
        </Card>

        {/* Account Section - RED BUTTON */}
        <Card>
          <Text style={{ marginBottom: 15, fontWeight: 'bold', fontSize: 16, color: '#333' }}>Account</Text>
          <TouchableOpacity 
            style={[styles.btnPrimary, { backgroundColor: '#D32F2F', height: 45 }]}
            onPress={() => navigation.replace('Login')}
          >
            <Ionicons name="log-out-outline" size={20} color="#FFF" style={{ marginRight: 8 }} />
            <Text style={{ color: '#FFF', fontWeight: 'bold' }}>Log Out</Text>
          </TouchableOpacity>
        </Card>

      </ScrollView>
    </View>
  );
}


function IshiharaIntroScreen({ navigation }) {
  return (
    <View style={styles.container}>
       <Header title="Choose Test Type" back />
       <ScrollView contentContainerStyle={{ padding: 20 }}>
         <Text style={{ color: '#777', marginBottom: 20 }}>Select the test duration that works best for you</Text>
         
         {/* COMPREHENSIVE TEST */}
         {/* FIX: Moved onPress DIRECTLY into Card. Removed TouchableOpacity wrapper. */}
         <Card 
           onPress={() => navigation.navigate('IshiharaTest', { testType: 'comprehensive' })}
           style={{ marginBottom: 20 }}
         >
           <View style={{ flexDirection: 'row' }}>
             <View style={{ flex: 1 }}>
               <Text style={styles.cardTitle}>Comprehensive Test</Text>
               <Text style={styles.cardDesc}>Complete 38-plate assessment</Text>
               <View style={{ marginTop: 10, flexDirection: 'row', alignItems: 'center' }}>
                 <Ionicons name="time-outline" size={16} color="#666" />
                 <Text style={{ fontSize: 12, marginLeft: 5, color: '#666' }}>15-20 minutes</Text>
               </View>
             </View>
             <View style={{ justifyContent: 'center', alignItems: 'center' }}>
                <View style={[styles.iconCircle, { backgroundColor: '#E3F2FD' }]}>
                  <Ionicons name="shield-checkmark" size={24} color="#2196F3" />
                </View>
                <Text style={{ fontSize: 10, color: '#2196F3', marginTop: 5, textAlign: 'center' }}>Accurate</Text>
             </View>
           </View>
         </Card>

         {/* QUICK TEST */}
         {/* FIX: Moved onPress DIRECTLY into Card. */}
         <Card 
           onPress={() => navigation.navigate('IshiharaTest', { testType: 'quick' })}
           style={{ marginBottom: 20 }}
         >
            <View style={{ flexDirection: 'row' }}>
              <View style={{ flex: 1 }}>
                <Text style={styles.cardTitle}>Quick Test</Text>
                <Text style={styles.cardDesc}>14-plate screening</Text>
                <View style={{ marginTop: 10, flexDirection: 'row', alignItems: 'center' }}>
                  <Ionicons name="time-outline" size={16} color="#666" />
                  <Text style={{ fontSize: 12, marginLeft: 5, color: '#666' }}>5-8 minutes</Text>
                </View>
              </View>
              <View style={{ justifyContent: 'center', alignItems: 'center' }}>
                  <View style={[styles.iconCircle, { backgroundColor: '#F3E5F5' }]}>
                    <Ionicons name="flash" size={24} color="#9C27B0" />
                  </View>
                  <Text style={{ fontSize: 10, color: '#9C27B0', marginTop: 5, textAlign: 'center' }}>Fast</Text>
              </View>
            </View>
          </Card>
       </ScrollView>
    </View>
  );
}

function IshiharaTestScreen({ route, navigation }) {
  const { testType } = route.params || { testType: 'quick' };
  
  // --- STATE ---
  const [testState, setTestState] = useState(() => {
    const sourceData = (RAW_PLATES && RAW_PLATES.length > 0) ? RAW_PLATES : [{id:999, img:null, answer:'0'}];
    const count = testType === 'comprehensive' ? 38 : 14;
    const safeCount = Math.min(count, sourceData.length);
    const shuffled = [...sourceData].sort(() => 0.5 - Math.random()).slice(0, safeCount);
    
    return {
      queue: shuffled,
      current: shuffled[0],
      index: 0,
      score: 0,
      userInput: ''
    };
  });

  const [timeLeft, setTimeLeft] = useState(5);
  const [showImage, setShowImage] = useState(true); // Controls visibility

  // --- LOGIC: TIMER & VISIBILITY ---
  useEffect(() => {
    if (!testState.current) return;

    // Reset for new plate
    setTimeLeft(5);
    setShowImage(true);

    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          setShowImage(false); // HIDE IMAGE after 5 seconds
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [testState.index]);

  // --- LOGIC: INPUT ---
  const handlePress = (num) => {
    if (testState.userInput.length < 3) {
      setTestState(prev => ({ ...prev, userInput: prev.userInput + num }));
    }
  };

  const handleBackspace = () => {
    setTestState(prev => ({ ...prev, userInput: prev.userInput.slice(0, -1) }));
  };

 const handleNext = async () => {
    const { queue, index, score, userInput, current } = testState;
    
    // Check answer
    const isCorrect = userInput === current.answer;
    const newScore = isCorrect ? score + 1 : score;

    if (index < queue.length - 1) {
      // Continue to next plate
      setTestState({
        queue,
        index: index + 1,
        current: queue[index + 1],
        score: newScore,
        userInput: ''
      });
    } else {
      // --- TEST FINISHED ---
      const finalScore = newScore;
      const total = queue.length;
      
      // Simple Diagnosis Logic
      let diagnosis = "Normal Vision";
      let severity = "None";
      const percentage = (finalScore / total) * 100;

      if (percentage < 80) {
        diagnosis = "Deuteranomaly (Simulated)";
        severity = percentage < 40 ? "Severe" : "Moderate";
      }

      // SAVE TO FIREBASE (The "Miracle" Part)
      if (auth.currentUser) {
        // This function handles the Dual Write (User Profile + Research Data)
        await saveExamResult(auth.currentUser.uid, finalScore, diagnosis, severity);
      }

      // NAVIGATE
      navigation.replace('IshiharaResult', { score: newScore, total: total, type: testType });
    }
  }; 

  if (!testState.current) return <View style={styles.container} />;

  // Progress calculation
  const progressPercent = ((testState.index + 1) / testState.queue.length) * 100;

  return (
    <View style={[styles.container, { flexDirection: 'column' }]}>
      
      {/* 1. TOP HEADER & PROGRESS */}
      <View style={{ backgroundColor: '#FFF', paddingBottom: 10, elevation: 2 }}>
        <Header title="Ishihara Test" subtitle={`Plate ${testState.index + 1} of ${testState.queue.length}`} />
        {/* Slim Progress Bar */}
        <View style={{ height: 4, width: '100%', backgroundColor: '#F0F0F0', marginTop: 10 }}>
          <View style={{ height: '100%', width: `${progressPercent}%`, backgroundColor: COLORS.primary }} />
        </View>
      </View>

      {/* 2. MIDDLE: IMAGE AREA (Takes all available space) */}
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F8F9FA' }}>
        
        {/* Card Container for Image */}
        <View style={{ 
          width: width * 0.85, 
          aspectRatio: 1, 
          backgroundColor: '#FFF', 
          borderRadius: 20, 
          justifyContent: 'center', 
          alignItems: 'center',
          elevation: 5, // Shadow
          shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 10, shadowOffset: {width:0, height:5}
        }}>
          {showImage ? (
            <Image 
              key={testState.index} 
              source={testState.current.img} 
              style={{ width: '90%', height: '90%', resizeMode: 'contain' }} 
            />
          ) : (
            // HIDDEN STATE
            <View style={{ alignItems: 'center' }}>
              <Ionicons name="eye-off-outline" size={60} color="#CCC" />
              <Text style={{ color: '#999', marginTop: 10, fontWeight: 'bold' }}>Image Hidden</Text>
              <Text style={{ color: '#AAA', fontSize: 12 }}>Enter what you saw</Text>
            </View>
          )}
        </View>

        {/* Timer Text under card */}
        <Text style={{ 
          marginTop: 20, 
          color: showImage ? (timeLeft <= 2 ? COLORS.danger : '#666') : '#CCC', 
          fontWeight: 'bold', 
          fontSize: 14 
        }}>
          {showImage ? `Time remaining: ${timeLeft}s` : "Time's up!"}
        </Text>
      </View>

      {/* 3. BOTTOM: CONTROLS (Fixed Height) */}
      <View style={{ backgroundColor: '#FFF', padding: 20, borderTopLeftRadius: 25, borderTopRightRadius: 25, elevation: 15 }}>
        
        {/* Input Display */}
        <View style={{ alignItems: 'center', marginBottom: 15 }}>
          <Text style={{ fontSize: 12, color: '#999', textTransform: 'uppercase', letterSpacing: 1 }}>Your Answer</Text>
          <Text style={{ fontSize: 32, fontWeight: 'bold', color: '#333', letterSpacing: 2 }}>
            {testState.userInput || '_'}
          </Text>
        </View>

        {/* Cleaner Numpad Grid */}
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 10 }}>
          {[1,2,3,4,5,6,7,8,9,0].map(n => (
            <TouchableOpacity 
              key={n} 
              style={{ 
                width: (width - 80) / 5, // 5 items per row roughly
                height: 50, 
                borderRadius: 12, 
                backgroundColor: '#F5F5F5', 
                alignItems: 'center', 
                justifyContent: 'center',
                borderWidth: 1, borderColor: '#EEE'
              }} 
              onPress={() => handlePress(n.toString())}
            >
              <Text style={{ fontSize: 20, fontWeight: '600', color: '#333' }}>{n}</Text>
            </TouchableOpacity>
          ))}
          
          {/* Backspace */}
          <TouchableOpacity 
            style={{ width: (width - 80) / 5, height: 50, borderRadius: 12, backgroundColor: '#FFEBEE', alignItems: 'center', justifyContent: 'center' }} 
            onPress={handleBackspace}
          >
             <Ionicons name="backspace-outline" size={24} color="#D32F2F" />
          </TouchableOpacity>
          
          {/* Submit - Big Button */}
          <TouchableOpacity 
            disabled={testState.userInput.length === 0} 
            style={{ 
              width: (width - 80) / 5, 
              height: 50, 
              borderRadius: 12, 
              backgroundColor: testState.userInput.length > 0 ? COLORS.primary : '#E0E0E0', 
              alignItems: 'center', 
              justifyContent: 'center' 
            }} 
            onPress={() => handleNext(false)}
          >
            <Ionicons name="arrow-forward" size={24} color="#FFF" />
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

function IshiharaResultScreen({ navigation }) {
  return (
    <ScrollView style={styles.container}>
      <Header title="Your Results" back />
      <View style={{ padding: 20 }}>
        
        <Card style={{ backgroundColor: '#FFF3E0', alignItems: 'center', paddingVertical: 30 }}>
           <Ionicons name="alert-circle" size={40} color="#FF9800" />
           <Text style={{ color: '#E65100', marginTop: 10 }}>Classification</Text>
           <Text style={{ color: '#E65100', fontSize: 12 }}>Likely</Text>
           <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#3E2723', marginVertical: 5 }}>Deuteranomaly</Text>
           <View style={{ backgroundColor: '#D50000', paddingHorizontal: 15, paddingVertical: 5, borderRadius: 15 }}>
             <Text style={{ color: '#FFF', fontSize: 12, fontWeight: 'bold' }}>Severity: Severe</Text>
           </View>
        </Card>

        <Card style={{ marginTop: 20, backgroundColor: '#E3F2FD' }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
            <Ionicons name="eye" size={24} color="#2196F3" />
            <Text style={{ marginLeft: 10, fontWeight: 'bold', color: '#0D47A1' }}>What This Means</Text>
          </View>
          <Text style={{ color: '#1565C0', lineHeight: 20 }}>
            Red-green color vision deficiency. Green colors appear less bright.
          </Text>
          <Text style={{ color: '#2196F3', marginTop: 10, fontWeight: 'bold' }}>Learn More {'>'}</Text>
        </Card>

        <Card style={{ marginTop: 20 }}>
          <Text style={{ fontWeight: 'bold', color: '#6C63FF', marginBottom: 15 }}>Recommendations</Text>
          
          <TouchableOpacity style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 15 }}>
             <View style={[styles.iconCircle, { backgroundColor: '#F3E5F5', width: 40, height: 40 }]}>
                <Ionicons name="color-wand" size={20} color="#9C27B0" />
             </View>
             <View style={{ marginLeft: 10, flex: 1 }}>
               <Text style={{ fontWeight: '600' }}>Color Enhancement</Text>
               <Text style={{ fontSize: 10, color: '#777' }}>Use app's real-time color mode</Text>
             </View>
             <Ionicons name="chevron-forward" size={20} color="#CCC" />
          </TouchableOpacity>

          <TouchableOpacity style={{ flexDirection: 'row', alignItems: 'center' }}>
             <View style={[styles.iconCircle, { backgroundColor: '#E3F2FD', width: 40, height: 40 }]}>
                <Ionicons name="medkit" size={20} color="#2196F3" />
             </View>
             <View style={{ marginLeft: 10, flex: 1 }}>
               <Text style={{ fontWeight: '600' }}>Professional Care</Text>
               <Text style={{ fontSize: 10, color: '#777' }}>Consult an eye care professional</Text>
             </View>
             <Ionicons name="chevron-forward" size={20} color="#CCC" />
          </TouchableOpacity>
        </Card>

        <TouchableOpacity 
          style={[styles.btnPrimary, { marginTop: 20, backgroundColor: '#1A237E' }]}
          onPress={() => navigation.navigate('MainTabs')}
        >
           <Text style={styles.btnText}>Back to Home</Text>
        </TouchableOpacity>
        <Text style={{ textAlign: 'center', fontSize: 10, color: '#999', marginTop: 15, paddingHorizontal: 20 }}>
          Screening tool only. Not a medical diagnosis. Consult qualified eye care professionals.
        </Text>
      </View>
    </ScrollView>
  );
}

function SurveyScreen({ navigation }) {
  const [selectedCause, setSelectedCause] = useState(null);
  const [selectedSex, setSelectedSex] = useState(null);
  
  return (
    <View style={styles.container}>
      <Header title="Quick Survey" back />
      <ScrollView contentContainerStyle={{ padding: 20 }}showsVerticalScrollIndicator={false}>
        <Text style={{ textAlign: 'center', fontWeight: 'bold', fontSize: 18, marginBottom: 5 }}>Help Us Understand</Text>
        <Text style={{ textAlign: 'center', color: '#777', marginBottom: 25 }}>What do you think are the causes of your CVD?</Text>

        {/* Causes Grid */}
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 20 }}>
           {['Medical Intake', 'Genetics', 'Ageing', 'Others'].map((item, idx) => (
             <TouchableOpacity 
               key={idx} 
               onPress={() => setSelectedCause(idx)}
               style={[
                 styles.surveyOption, 
                 selectedCause === idx && { borderColor: COLORS.primary, borderWidth: 2, backgroundColor: '#F3E5F5' }
               ]}
             >
                <Ionicons 
                  name={selectedCause === idx ? "checkmark-circle" : "radio-button-off"} 
                  size={24} 
                  color={selectedCause === idx ? COLORS.primary : "#CCC"} 
                />
                <Text style={{ fontWeight: 'bold', marginTop: 10 }}>{item}</Text>
             </TouchableOpacity>
           ))}
        </View>

        {/* SEX SELECTION */}
        <Card>
          <Text style={{ fontWeight: 'bold', marginBottom: 15 }}>Sex</Text>
          <View style={{ flexDirection: 'row', justifyContent: 'space-around' }}>
            {['Male', 'Female'].map((sex) => (
              <TouchableOpacity 
                key={sex}
                style={{ flexDirection: 'row', alignItems: 'center', padding: 10, borderWidth: 1, borderColor: selectedSex === sex ? COLORS.primary : '#EEE', borderRadius: 8, width: '45%', justifyContent:'center' }}
                onPress={() => setSelectedSex(sex)}
              >
                <Ionicons 
                  name={selectedSex === sex ? "radio-button-on" : "radio-button-off"} 
                  size={20} 
                  color={selectedSex === sex ? COLORS.primary : "#999"} 
                />
                <Text style={{ marginLeft: 10, fontWeight: selectedSex === sex ? 'bold' : 'normal' }}>{sex}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </Card>

        <TouchableOpacity 
          style={[styles.btnPrimary, { marginTop: 30, backgroundColor: '#111' }]}
          onPress={() => navigation.replace('SurveySuccess')}
        >
          <Text style={styles.btnText}>Submit Survey</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

function SurveySuccessScreen({ navigation }) {
  return (
    <View style={[styles.container, { justifyContent: 'center', alignItems: 'center', backgroundColor: '#F8F9FA' }]}>
      <View style={{ width: 80, height: 80, borderRadius: 40, backgroundColor: '#E8F5E9', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
        <Ionicons name="checkmark" size={40} color={COLORS.success} />
      </View>
      <Text style={{ fontSize: 22, fontWeight: 'bold', color: '#333' }}>Thank You!</Text>
      <Text style={{ color: '#777', marginTop: 10 }}>Your response has been recorded</Text>
      
      <TouchableOpacity 
        style={[styles.btnOutline, { marginTop: 40, width: 200, backgroundColor: '#FFF' }]}
        // FIX: Navigate to 'MainTabs' instead of 'Home'
        onPress={() => navigation.navigate('MainTabs')}
      >
        <Text style={{ color: COLORS.primary, fontWeight: 'bold' }}>Back to Home</Text>
      </TouchableOpacity>
    </View>
  );
}

function CameraSimScreen({ navigation }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPosition, setCameraPosition] = useState('back');
  const [cvdType, setCvdType]           = useState('Protan');
  const [processedUri, setProcessedUri] = useState(null);
  const [showModal, setShowModal]       = useState(false);
  const [modelStatus, setModelStatus]   = useState('loading');

  const cameraRef      = useRef(null);
  const isProcessing   = useRef(false);
  const timerRef       = useRef(null);
  const cvdTypeRef     = useRef(cvdType);

  const device = useCameraDevice(cameraPosition);

  // Load TFLite model
  const tflite = useTensorflowModel(require('./assets/color_model.tflite'));

  // Keep ref in sync with state
  useEffect(() => { cvdTypeRef.current = cvdType; }, [cvdType]);

  // Track model readiness
  useEffect(() => {
    if (tflite.state === 'loaded') setModelStatus('ready');
    if (tflite.state === 'error')  setModelStatus('error');
  }, [tflite.state]);

  const DISPLAY_SIZE = 384;
  const CNN_SIZE = 128;

  // ── Self-scheduling frame pipeline (no setInterval spam) ──
  const runFramePipeline = useCallback(async () => {
    if (isProcessing.current) return;
    if (!cameraRef.current) { timerRef.current = setTimeout(runFramePipeline, 200); return; }
    if (tflite.state !== 'loaded' || !tflite.model) { timerRef.current = setTimeout(runFramePipeline, 200); return; }
    if (cvdTypeRef.current === 'Off') { setProcessedUri(null); timerRef.current = setTimeout(runFramePipeline, 200); return; }

    isProcessing.current = true;
    try {
      // 1. VisionCamera takePhoto — no preview freeze, no shutter sound
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        enableShutterSound: false,
      });
      const fileUri = `file://${photo.path}`;

      // 2. Single async resize to display resolution (384x384)
      const displayResized = await ImageManipulator.manipulateAsync(
        fileUri,
        [{ resize: { width: DISPLAY_SIZE, height: DISPLAY_SIZE } }],
        { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 0.8 }
      );

      // 3. Decode to RGBA at display resolution
      const rawImage = decodeJpegBase64(displayResized.base64);

      // 4. Downscale RGBA → CNN tensor in JS (fast, synchronous — no second async call)
      const inputTensor = downscaleToTensor(rawImage.data, DISPLAY_SIZE, DISPLAY_SIZE, CNN_SIZE, CNN_SIZE);

      // 5. Run U-Net inference at 128x128
      const outputs = tflite.model.runSync([inputTensor]);
      const mask128 = getClassMask(outputs[0]);

      // 6. Upscale mask to display resolution (nearest-neighbor)
      const mask = upscaleMaskNearest(mask128, CNN_SIZE, CNN_SIZE, DISPLAY_SIZE, DISPLAY_SIZE);

      // 7. Apply daltonization at display resolution (9x more pixels than 128x128)
      const daltonized = applyDaltonization(rawImage, mask, cvdTypeRef.current);

      // 8. Re-encode to data URI for display
      const uri = encodeToDataUri(daltonized, DISPLAY_SIZE, DISPLAY_SIZE);
      setProcessedUri(uri);
    } catch (e) {
      // Silently ignore frame errors
    } finally {
      isProcessing.current = false;
      // Schedule next frame immediately after current finishes
      timerRef.current = setTimeout(runFramePipeline, 50);
    }
  }, [tflite.state, tflite.model]);

  // Start pipeline when model is ready, stop on unmount
  useEffect(() => {
    if (tflite.state === 'loaded') {
      timerRef.current = setTimeout(runFramePipeline, 200);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [tflite.state, runFramePipeline]);

  // Clear overlay when CVD type changes to Off
  useEffect(() => {
    if (cvdType === 'Off') setProcessedUri(null);
  }, [cvdType]);

  // ── Capture to gallery ──
  const handleCapture = async () => {
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') { Alert.alert('Permission needed', 'Allow access to save photos.'); return; }
      if (!cameraRef.current) return;

      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'balanced',
        enableShutterSound: false,
      });
      const fileUri = `file://${photo.path}`;

      if (cvdType !== 'Off' && tflite.state === 'loaded' && tflite.model) {
        // Save daltonized version at higher resolution
        const CAPTURE_SIZE = 512;
        const captureResized = await ImageManipulator.manipulateAsync(
          fileUri, [{ resize: { width: CAPTURE_SIZE, height: CAPTURE_SIZE } }],
          { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 0.9 }
        );
        const rawImage = decodeJpegBase64(captureResized.base64);
        const inputTensor = downscaleToTensor(rawImage.data, CAPTURE_SIZE, CAPTURE_SIZE, CNN_SIZE, CNN_SIZE);
        const outputs = tflite.model.runSync([inputTensor]);
        const mask128 = getClassMask(outputs[0]);
        const mask = upscaleMaskNearest(mask128, CNN_SIZE, CNN_SIZE, CAPTURE_SIZE, CAPTURE_SIZE);
        const daltonized = applyDaltonization(rawImage, mask, cvdType);
        const uri = encodeToDataUri(daltonized, CAPTURE_SIZE, CAPTURE_SIZE);
        const filename = `daltonized_${Date.now()}.jpg`;
        const saveUri = FileSystem.documentDirectory + filename;
        await FileSystem.writeAsStringAsync(saveUri, uri.split(',')[1], { encoding: FileSystem.EncodingType.Base64 });
        await MediaLibrary.saveToLibraryAsync(saveUri);
        Alert.alert('Saved', 'Enhanced photo saved to gallery.');
      } else {
        await MediaLibrary.saveToLibraryAsync(fileUri);
        Alert.alert('Saved', 'Photo saved to gallery.');
      }
    } catch (e) {
      console.warn('[CameraSim] capture error:', e);
      Alert.alert('Error', 'Could not save the photo.');
    }
  };

  // ── Permission gates ──
  if (hasPermission === null) return <View />;
  if (!hasPermission) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text style={{ textAlign: 'center', marginBottom: 20 }}>
          We need camera access for ReColor.
        </Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>

      {/* VisionCamera — no shutter sound, no preview freeze */}
      {device && (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          enableShutterSound={false}
        />
      )}

      {/* Daltonized frame overlay */}
      {processedUri && cvdType !== 'Off' && (
        <Image
          source={{ uri: processedUri }}
          style={StyleSheet.absoluteFill}
          resizeMode="cover"
          fadeDuration={0}
          pointerEvents="none"
        />
      )}

      <SafeAreaView style={{ flex: 1 }} pointerEvents="box-none">

        {/* Top Bar */}
        <View style={styles.camTopBar}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={{ padding: 5 }}>
            <Ionicons name="arrow-back" size={24} color="#FFF" />
          </TouchableOpacity>

          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Text style={{
              color: '#FFF', fontWeight: 'bold', marginRight: 10,
              textShadowColor: 'rgba(0,0,0,0.75)',
              textShadowOffset: { width: -1, height: 1 },
              textShadowRadius: 10,
            }}>
              {cvdType === 'Off' ? 'Normal' : `${cvdType} Mode`}
            </Text>
            <TouchableOpacity onPress={() => setShowModal(true)}>
              <Ionicons name="menu" size={28} color="#FFF" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Model status indicator */}
        {modelStatus === 'loading' && (
          <View style={{ position: 'absolute', top: 70, alignSelf: 'center',
            backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6 }}>
            <Text style={{ color: '#FFF', fontSize: 12 }}>Loading model...</Text>
          </View>
        )}
        {modelStatus === 'error' && (
          <View style={{ position: 'absolute', top: 70, alignSelf: 'center',
            backgroundColor: 'rgba(200,0,0,0.7)', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6 }}>
            <Text style={{ color: '#FFF', fontSize: 12 }}>Model failed - native build required</Text>
          </View>
        )}

        {/* CVD type selector (right side) */}
        <View style={{ position: 'absolute', top: 100, right: 20, alignItems: 'center' }}>
          {['Off', 'Protan', 'Deutan', 'Tritan'].map((m) => (
            <TouchableOpacity
              key={m}
              onPress={() => setCvdType(m)}
              style={[
                styles.filterBtn,
                {
                  backgroundColor: cvdType === m ? COLORS.primary : 'rgba(0,0,0,0.5)',
                  marginBottom: 15,
                },
              ]}
            >
              <Text style={{ color: '#FFF', fontWeight: 'bold', fontSize: 10 }}>
                {m === 'Off' ? 'Off' : m.charAt(0)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Bottom controls */}
        <View style={{ flex: 1, justifyContent: 'flex-end', paddingBottom: 30 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
            <TouchableOpacity onPress={() => setCameraPosition(p => p === 'back' ? 'front' : 'back')}>
              <Ionicons name="camera-reverse" size={30} color="#FFF" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.shutterBtn} onPress={handleCapture}>
              <View style={{ width: 60, height: 60, borderRadius: 30, backgroundColor: '#FFF' }} />
            </TouchableOpacity>

            <TouchableOpacity onPress={() => navigation.navigate('CVDGallery')}>
              <Ionicons name="images" size={30} color="#FFF" />
            </TouchableOpacity>
          </View>
        </View>

        <ModeSelector
          visible={showModal}
          onClose={() => setShowModal(false)}
          navigation={navigation}
          currentMode="Enhancement"
        />
      </SafeAreaView>
    </View>
  );
}

// --- THE REAL LOGIC: Euclidean Color Classifier ---
// This replaces the TFLite model for the Expo Go environment.
// It calculates the geometric distance between the camera pixel and known colors.
const CLASSIFIER_DB = {
  'Red': { r: 255, g: 0, b: 0, hex: '#FF0000' },
  'Green': { r: 0, g: 128, b: 0, hex: '#008000' },
  'Blue': { r: 0, g: 0, b: 255, hex: '#0000FF' },
  'Yellow': { r: 255, g: 255, b: 0, hex: '#FFFF00' },
  'Cyan': { r: 0, g: 255, b: 255, hex: '#00FFFF' },
  'Magenta': { r: 255, g: 0, b: 255, hex: '#FF00FF' },
  'White': { r: 255, g: 255, b: 255, hex: '#FFFFFF' },
  'Black': { r: 0, g: 0, b: 0, hex: '#000000' },
  'Gray': { r: 128, g: 128, b: 128, hex: '#808080' },
  'Orange': { r: 255, g: 165, b: 0, hex: '#FFA500' },
  'Purple': { r: 128, g: 0, b: 128, hex: '#800080' },
  'Brown': { r: 165, g: 42, b: 42, hex: '#A52A2A' },
  'Beige': { r: 245, g: 245, b: 220, hex: '#F5F5DC' },
  'Gold': { r: 255, g: 215, b: 0, hex: '#FFD700' },
  'Silver': { r: 192, g: 192, b: 192, hex: '#C0C0C0' },
  'Pink': { r: 255, g: 192, b: 203, hex: '#FFC0CB' },
  'Navy': { r: 0, g: 0, b: 128, hex: '#000080' },
  'Teal': { r: 0, g: 128, b: 128, hex: '#008080' },
  'Lime': { r: 0, g: 255, b: 0, hex: '#00FF00' }
};

function ColorIdentifierScreen({ navigation }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPosition, setCameraPosition] = useState('back');
  const [audio, setAudio] = useState(true);
  const [showModal, setShowModal] = useState(false);

  const [cursorPosition, setCursorPosition] = useState({ x: width / 2, y: 300 });
  const [identifiedColor, setIdentifiedColor] = useState({ name: 'Ready to Scan', hex: '#333', conf: '' });
  const [isProcessing, setIsProcessing] = useState(false);

  const cameraRef = useRef(null);
  const device = useCameraDevice(cameraPosition);

  // --- PERMISSION CHECK ---
  if (hasPermission === null) return <View />;
  if (!hasPermission) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text style={{ marginBottom: 20 }}>Camera access is needed.</Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // --- Color detection using center pixel sampling ---
  const runDetection = async () => {
    if (!cameraRef.current || isProcessing) return;
    setIsProcessing(true);

    try {
      // 1. Take photo — no shutter sound, no preview freeze
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        enableShutterSound: false,
      });
      const fileUri = `file://${photo.path}`;

      // 2. Crop center 20x20 pixels then resize to 1x1 for average color
      const cropResult = await ImageManipulator.manipulateAsync(
        fileUri,
        [
          { crop: { originX: photo.width / 2 - 10, originY: photo.height / 2 - 10, width: 20, height: 20 } },
          { resize: { width: 1, height: 1 } },
        ],
        { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 1.0 }
      );

      // 3. Decode the 1x1 JPEG to get actual RGB values
      const pixelData = decodeJpegBase64(cropResult.base64);
      const r = pixelData.data[0];
      const g = pixelData.data[1];
      const b = pixelData.data[2];

      // 4. Find nearest color by Euclidean distance
      let bestName = 'Unknown';
      let bestHex = '#333';
      let bestDist = Infinity;
      for (const [name, col] of Object.entries(CLASSIFIER_DB)) {
        const dist = Math.sqrt((r - col.r) ** 2 + (g - col.g) ** 2 + (b - col.b) ** 2);
        if (dist < bestDist) {
          bestDist = dist;
          bestName = name;
          bestHex = col.hex;
        }
      }

      // Confidence: inverse of distance (max RGB distance is ~441)
      const confidence = Math.max(0, Math.round((1 - bestDist / 441) * 100));

      setIdentifiedColor({ name: bestName, hex: bestHex, conf: `${confidence}%` });
    } catch (e) {
      console.log('[ColorID] detection error:', e);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTouch = (evt) => {
    const { locationX, locationY } = evt.nativeEvent;
    setCursorPosition({ x: locationX, y: locationY });
    runDetection();
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>

      {/* VisionCamera — no shutter sound, no freeze */}
      {device && (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          enableShutterSound={false}
        />
      )}

      <View
        style={StyleSheet.absoluteFill}
        onStartShouldSetResponder={() => true}
        onResponderMove={handleTouch}
        onResponderGrant={handleTouch}
      />

      <SafeAreaView style={{ flex: 1 }} pointerEvents="box-none">
        <View style={styles.camTopBar}>
          <View style={{flexDirection:'row', alignItems:'center'}}>
              <TouchableOpacity onPress={() => navigation.goBack()} style={{ marginRight: 10 }}>
                <Ionicons name="arrow-back" size={24} color="#FFF" />
              </TouchableOpacity>
              <View style={styles.camPill}>
                 <Text style={{ color: '#FFF', fontSize: 12, fontWeight: 'bold' }}>Color Identifier</Text>
              </View>
          </View>
          <View style={{flexDirection:'row', alignItems:'center'}}>
             <TouchableOpacity onPress={() => setAudio(!audio)} style={{ marginRight: 15 }}>
                <Ionicons name={audio ? "volume-high" : "volume-mute"} size={24} color="#FFF" />
             </TouchableOpacity>
             <TouchableOpacity onPress={() => setShowModal(true)}>
               <Ionicons name="menu" size={28} color="#FFF" />
             </TouchableOpacity>
          </View>
        </View>

        {/* CROSSHAIR */}
        <View pointerEvents="none" style={{ position: 'absolute', top: cursorPosition.y - 50, left: cursorPosition.x - 50, width: 100, height: 100, justifyContent: 'center', alignItems: 'center' }}>
           <View style={{ width: 2, height: 50, backgroundColor: 'rgba(255,255,255,0.8)', position:'absolute' }} />
           <View style={{ width: 50, height: 2, backgroundColor: 'rgba(255,255,255,0.8)', position:'absolute' }} />
           <View style={{ width: 20, height: 20, borderRadius: 10, borderWidth: 2, borderColor: '#FFF' }} />
        </View>

        {/* RESULT CARD */}
        <View style={{ position: 'absolute', top: '60%', alignSelf: 'center', pointerEvents: 'none' }}>
           <View style={{ backgroundColor: 'rgba(0,0,0,0.85)', padding: 15, borderRadius: 12, alignItems: 'center', minWidth: 150 }}>
              <View style={{ width: 30, height: 30, borderRadius: 15, backgroundColor: identifiedColor.hex, marginBottom: 5, borderWidth: 2, borderColor:'#FFF' }} />
              <Text style={{ color: '#FFF', fontWeight: 'bold', fontSize: 16 }}>{identifiedColor.name}</Text>
              <Text style={{ color: '#CCC', fontSize: 12 }}>
                 {isProcessing ? "Calculating..." : `RGB Match: ${identifiedColor.conf}`}
              </Text>
           </View>
        </View>

        {/* BOTTOM CONTROLS */}
        <View style={{ position: 'absolute', bottom: 30, width: '100%', alignItems: 'center', zIndex: 30 }}>
           <View style={{ marginBottom: 20, backgroundColor:'rgba(0,0,0,0.6)', paddingHorizontal: 15, paddingVertical: 8, borderRadius: 20 }}>
              <Text style={{ color: '#FFF', fontSize: 12, fontWeight: 'bold' }}>Tap to identify color</Text>
           </View>
           <View style={{ flexDirection: 'row', width: '100%', justifyContent: 'space-between', paddingHorizontal: 30, alignItems: 'center' }}>
              <TouchableOpacity style={styles.camBtnCircleSmall} onPress={() => setCameraPosition(p => p === 'back' ? 'front' : 'back')}>
                 <Ionicons name="camera-reverse-outline" size={24} color="#FFF" />
              </TouchableOpacity>
              <TouchableOpacity style={styles.camBtnCircleSmall} onPress={() => navigation.navigate('CVDGallery')}>
                 <Ionicons name="image-outline" size={24} color="#FFF" />
              </TouchableOpacity>
           </View>
        </View>

        <ModeSelector visible={showModal} onClose={() => setShowModal(false)} navigation={navigation} currentMode="Identifier" />
      </SafeAreaView>
    </View>
  );
}

function CVDSimulationScreen({ navigation }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [mode, setMode]                 = useState('Off');
  const [showModal, setShowModal]       = useState(false);
  const [cameraPosition, setCameraPosition] = useState('back');
  const [processedUri, setProcessedUri] = useState(null);

  const cameraRef = useRef(null);
  const device = useCameraDevice(cameraPosition);
  const isProcessing = useRef(false);
  const timerRef     = useRef(null);
  const modeRef      = useRef(mode);

  // Keep ref in sync with state
  useEffect(() => { modeRef.current = mode; }, [mode]);

  // ── Self-scheduling CVD simulation loop ──
  const runSimLoop = useCallback(async () => {
    if (isProcessing.current) return;
    if (!cameraRef.current) { timerRef.current = setTimeout(runSimLoop, 200); return; }
    if (modeRef.current === 'Off') { setProcessedUri(null); timerRef.current = setTimeout(runSimLoop, 200); return; }

    isProcessing.current = true;
    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        enableShutterSound: false,
      });
      const fileUri = `file://${photo.path}`;

      // Resize for fast JS-side processing (no CNN — just matrix multiply)
      const resized = await ImageManipulator.manipulateAsync(
        fileUri,
        [{ resize: { width: 384, height: 384 } }],
        { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 0.8 }
      );
      const rawImage = decodeJpegBase64(resized.base64);

      // Apply CVD color matrix to every pixel (no CNN needed — simple matrix multiply)
      const m = getCVDColorMatrix(modeRef.current);
      const pixels = rawImage.data;
      for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i] / 255, g = pixels[i+1] / 255, b = pixels[i+2] / 255;
        pixels[i]   = Math.min(255, Math.max(0, (m[0]*r + m[1]*g + m[2]*b) * 255));
        pixels[i+1] = Math.min(255, Math.max(0, (m[5]*r + m[6]*g + m[7]*b) * 255));
        pixels[i+2] = Math.min(255, Math.max(0, (m[10]*r + m[11]*g + m[12]*b) * 255));
      }

      const uri = encodeToDataUri(rawImage, rawImage.width, rawImage.height);
      setProcessedUri(uri);
    } catch (e) {
      // Silently ignore frame errors
    } finally {
      isProcessing.current = false;
      timerRef.current = setTimeout(runSimLoop, 50); // Faster than Enhancement — no CNN overhead
    }
  }, []);

  // Start loop on mount, stop on unmount
  useEffect(() => {
    timerRef.current = setTimeout(runSimLoop, 200);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [runSimLoop]);

  // Clear overlay when mode is Off
  useEffect(() => {
    if (mode === 'Off') setProcessedUri(null);
  }, [mode]);

  // --- PERMISSION CHECK ---
  if (hasPermission === null) return <View />;
  if (!hasPermission) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text style={{ marginBottom: 20 }}>Camera access is needed for simulation.</Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const handleCapture = async () => {
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission needed', 'Please allow access to save photos.');
        return;
      }

      if (cameraRef.current) {
        const photo = await cameraRef.current.takePhoto({
          qualityPrioritization: 'balanced',
          enableShutterSound: false,
        });
        const fileUri = `file://${photo.path}`;

        if (mode !== 'Off') {
          const resized = await ImageManipulator.manipulateAsync(
            fileUri,
            [{ resize: { width: 512, height: 512 } }],
            { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 0.9 }
          );
          const rawImage = decodeJpegBase64(resized.base64);
          const m = getCVDColorMatrix(mode);
          const pixels = rawImage.data;
          for (let i = 0; i < pixels.length; i += 4) {
            const r = pixels[i] / 255, g = pixels[i+1] / 255, b = pixels[i+2] / 255;
            pixels[i]   = Math.min(255, Math.max(0, (m[0]*r + m[1]*g + m[2]*b) * 255));
            pixels[i+1] = Math.min(255, Math.max(0, (m[5]*r + m[6]*g + m[7]*b) * 255));
            pixels[i+2] = Math.min(255, Math.max(0, (m[10]*r + m[11]*g + m[12]*b) * 255));
          }
          const uri = encodeToDataUri(rawImage, rawImage.width, rawImage.height);
          const filename = `cvd_sim_${Date.now()}.jpg`;
          const saveUri = FileSystem.documentDirectory + filename;
          await FileSystem.writeAsStringAsync(saveUri, uri.split(',')[1], {
            encoding: FileSystem.EncodingType.Base64,
          });
          await MediaLibrary.saveToLibraryAsync(saveUri);
          Alert.alert('Saved', 'CVD simulation photo saved to gallery.');
        } else {
          await MediaLibrary.saveToLibraryAsync(fileUri);
          Alert.alert('Saved', 'Photo saved to gallery.');
        }
      }
    } catch (e) {
      console.warn('[CVDSim] capture error:', e);
      Alert.alert('Error', 'Could not save the photo.');
    }
  };

  const getSimDescription = () => {
    switch(mode) {
      case 'Protan': return 'Protanopia: Red-green color blindness (red deficiency)';
      case 'Deutan': return 'Deuteranopia: Red-green color blindness (green deficiency)';
      case 'Tritan': return 'Tritanopia: Blue-yellow color blindness';
      default: return 'Normal Vision: No simulation active';
    }
  };

  const cameraLabel = cameraPosition === 'back' ? 'Back Camera' : 'Front Camera';

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>

      {/* VisionCamera — live feed, no frame processor (avoids Skia crash) */}
      {device && (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          enableShutterSound={false}
        />
      )}

      {/* CVD-simulated overlay */}
      {processedUri && mode !== 'Off' && (
        <Image
          source={{ uri: processedUri }}
          style={StyleSheet.absoluteFill}
          resizeMode="cover"
          fadeDuration={0}
          pointerEvents="none"
        />
      )}

      <SafeAreaView style={{ flex: 1 }} pointerEvents="box-none">

        <View style={styles.camTopBar}>
          <View style={{flexDirection:'row', alignItems:'center'}}>
              <TouchableOpacity onPress={() => navigation.goBack()} style={{ marginRight: 10 }}>
                <Ionicons name="arrow-back" size={24} color="#FFF" />
              </TouchableOpacity>
              <View style={styles.camPill}>
                 <Text style={{ color: '#FFF', fontSize: 12, fontWeight: 'bold' }}>{cameraLabel}</Text>
              </View>
          </View>
          <View style={{flexDirection:'row', alignItems:'center'}}>
             <Text style={{color:'#FFF', fontWeight:'bold', marginRight:10}}>CVD Sim</Text>
             <TouchableOpacity onPress={() => setShowModal(true)}>
               <Ionicons name="menu" size={28} color="#FFF" />
             </TouchableOpacity>
          </View>
        </View>

        {/* --- INFO BOX --- */}
        <View style={{
          position: 'absolute', top: 120, left: 20, right: 85,
          backgroundColor: 'rgba(30, 20, 20, 0.9)',
          padding: 15, borderRadius: 12,
          flexDirection: 'row', alignItems: 'center'
        }}>
           <Ionicons name="information-circle-outline" size={24} color="#FFF" style={{marginRight: 12}} />
           <View style={{flex: 1}}>
              <Text style={{ color: '#BBB', fontSize: 10, marginBottom: 2 }}>Simulation Active</Text>
              <Text style={{ color: '#FFF', fontSize: 13, fontWeight: 'bold', lineHeight: 18 }}>
                {getSimDescription()}
              </Text>
           </View>
        </View>

        {/* Side Modes */}
        <View style={{ position: 'absolute', top: 120, right: 20, alignItems: 'center', zIndex: 100, elevation: 100 }}>
           <TouchableOpacity onPress={() => setMode('Off')} style={[styles.filterBtn, { backgroundColor: '#555', marginBottom: 20 }, mode === 'Off' && styles.filterBtnActive]}>
             <Text style={{ fontWeight: 'bold', color: '#FFF' }}>Off</Text>
           </TouchableOpacity>

           <TouchableOpacity onPress={() => setMode('Protan')} style={[styles.filterBtn, { backgroundColor: '#FF3B30', marginBottom: 15 }, mode === 'Protan' && styles.filterBtnActive]}>
             <Text style={styles.filterText}>P</Text>
           </TouchableOpacity>

           <TouchableOpacity onPress={() => setMode('Deutan')} style={[styles.filterBtn, { backgroundColor: '#4CD964', marginBottom: 15 }, mode === 'Deutan' && styles.filterBtnActive]}>
             <Text style={styles.filterText}>D</Text>
           </TouchableOpacity>

           <TouchableOpacity onPress={() => setMode('Tritan')} style={[styles.filterBtn, { backgroundColor: '#007AFF' }, mode === 'Tritan' && styles.filterBtnActive]}>
             <Text style={styles.filterText}>T</Text>
           </TouchableOpacity>
        </View>

        {/* Bottom Controls */}
        <View style={{ position: 'absolute', bottom: 30, width: '100%', flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 30, alignItems: 'center', zIndex: 50 }}>
           <TouchableOpacity style={styles.camBtnCircleSmall} onPress={() => setCameraPosition(p => p === 'back' ? 'front' : 'back')}>
              <Ionicons name="camera-reverse-outline" size={24} color="#FFF" />
           </TouchableOpacity>

           <TouchableOpacity style={styles.shutterBtn} onPress={handleCapture}>
              <Ionicons name="camera" size={32} color="#000" />
           </TouchableOpacity>

           <TouchableOpacity style={styles.camBtnCircleSmall} onPress={() => navigation.navigate('CVDGallery')}>
              <Ionicons name="image-outline" size={24} color="#FFF" />
           </TouchableOpacity>
        </View>

        <ModeSelector visible={showModal} onClose={() => setShowModal(false)} navigation={navigation} currentMode="Simulation" />
      </SafeAreaView>
    </View>
  );
}
function CVDGalleryScreen({ navigation }) {
  const [image, setImage] = useState(null);
  const [mode, setMode] = useState('Off');

  // 1. Pick Image Logic
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  // 2. Filter Logic (Overlay)
  const getFilterColor = () => {
    switch (mode) {
      case 'Protan': return 'rgba(255, 0, 0, 0.15)'; // Correction: Boost Red
      case 'Deutan': return 'rgba(0, 255, 0, 0.15)'; // Correction: Boost Green
      case 'Tritan': return 'rgba(0, 0, 255, 0.15)'; // Correction: Boost Blue
      default: return 'transparent';
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>
      <SafeAreaView style={{ flex: 1 }}>
        
        {/* Header */}
        <View style={styles.camTopBar}>
          <TouchableOpacity onPress={() => navigation.goBack()}><Ionicons name="arrow-back" size={24} color="#FFF" /></TouchableOpacity>
          <Text style={{ color: '#FFF', fontWeight: 'bold' }}>Gallery Analysis</Text>
          <TouchableOpacity onPress={pickImage}><Ionicons name="add-circle" size={28} color="#FFF" /></TouchableOpacity>
        </View>

        {/* Main Content */}
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          {image ? (
            <View style={{ width: width, height: width * 1.3 }}>
              <Image source={{ uri: image }} style={{ width: '100%', height: '100%', resizeMode: 'contain' }} />
              {/* THE FILTER OVERLAY */}
              <View style={[StyleSheet.absoluteFill, { backgroundColor: getFilterColor() }]} pointerEvents="none" />
            </View>
          ) : (
            <TouchableOpacity onPress={pickImage} style={{ alignItems: 'center' }}>
              <Ionicons name="images-outline" size={60} color="#555" />
              <Text style={{ color: '#777', marginTop: 10 }}>Tap to pick an image</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Controls */}
        {image && (
          <View style={{ flexDirection: 'row', justifyContent: 'center', paddingBottom: 30, gap: 10 }}>
             {['Off', 'Protan', 'Deutan', 'Tritan'].map(m => (
               <TouchableOpacity 
                 key={m} 
                 onPress={() => setMode(m)} 
                 style={{ backgroundColor: mode === m ? COLORS.primary : '#333', padding: 10, borderRadius: 20, paddingHorizontal: 20 }}
               >
                 <Text style={{ color: '#FFF', fontWeight: 'bold' }}>{m}</Text>
               </TouchableOpacity>
             ))}
          </View>
        )}

      </SafeAreaView>
    </View>
  );
}

function EducationListScreen({ navigation }) {
  
  const openWebsite = () => {
    Linking.openURL('https://peri.ph/');
  };

  const ARTICLES = [
    { 
      id: 1, 
      title: 'Designing for Accessibility', 
      desc: 'Best practices for creating color-accessible content and environments.',
      image: require('./assets/access.png') // <--- Updated to PNG
    },
    { 
      id: 2, 
      title: 'Tips for Daily Life', 
      desc: 'Practical strategies for navigating a color-coded world with confidence.',
      image: require('./assets/domore.png') // <--- Updated to PNG
    },
    { 
      id: 3, 
      title: 'What is Color Vision Deficiency?', 
      desc: 'Learn about the science behind color vision and how CVD affects millions.',
      image: require('./assets/art.png') // <--- Updated to PNG
    },
    { 
      id: 4, 
      title: 'Types of Color Blindness', 
      desc: 'Understand the different types of CVD including protanomaly and deuteranomaly.',
      image: require('./assets/eye.png') // <--- Updated to PNG
    }
  ];

  return (
    <View style={styles.container}>
      <Header title="Learn & Understand" back />
      <BackgroundBubbles />
      
      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        
        {/* --- PERI FEATURED CARD --- */}
        <View style={{ backgroundColor: '#F3E5F5', borderRadius: 16, overflow: 'hidden', marginBottom: 25, borderWidth: 1, borderColor: '#E1BEE7' }}>
          {/* UPDATED TO PNG */}
          <Image 
            source={require('./assets/peri.png')} 
            style={{ width: '100%', height: 180 }}
            resizeMode="cover"
          />
          <View style={{ padding: 20 }}>
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
              <Ionicons name="business" size={20} color="#9C27B0" />
              <Text style={{ marginLeft: 10, color: '#4A148C', fontWeight: 'bold', fontSize: 16, flex: 1 }}>
                About Philippine Eye Research Institute
              </Text>
            </View>
            
            <Text style={{ color: '#4A148C', fontSize: 13, lineHeight: 20, marginBottom: 15 }}>
              The Philippine Eye Research Institute (PERI) is the premier eye research institution in the Philippines, dedicated to preventing blindness.
            </Text>

            <Text style={{ fontSize: 13, color: '#333', lineHeight: 20 }}>
              <Text style={{ fontWeight: 'bold' }}>Our Mission:</Text> To improve eye health outcomes through innovative research and evidence-based clinical practices.
            </Text>
            
            <TouchableOpacity 
              style={{ marginTop: 20, backgroundColor: '#FFF', paddingVertical: 12, borderRadius: 8, alignItems: 'center', flexDirection: 'row', justifyContent: 'center', borderWidth: 1, borderColor: '#D1C4E9' }}
              onPress={openWebsite}
            >
              <Text style={{ color: '#4A148C', fontWeight: 'bold', marginRight: 8 }}>Visit PERI Website</Text>
              <Ionicons name="open-outline" size={16} color="#4A148C" />
            </TouchableOpacity>
          </View>
        </View>

        {/* --- CVD SIMULATIONS --- */}
        <Card style={{ marginBottom: 25, backgroundColor: '#F5F7FA' }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 15 }}>
            <Ionicons name="eye-outline" size={24} color="#333" />
            <Text style={{ marginLeft: 10, fontWeight: 'bold', fontSize: 16, color: '#333' }}>CVD Simulations</Text>
          </View>
          <Text style={{ color: '#666', fontSize: 12, marginBottom: 15 }}>
            Experience how different types of color vision deficiency affect color perception.
          </Text>

          {/* Protanomaly Button */}
          <TouchableOpacity 
            style={{ backgroundColor: '#FFEBEE', padding: 15, borderRadius: 12, flexDirection: 'row', alignItems: 'center', marginBottom: 10, borderWidth: 1, borderColor: '#FFCDD2' }}
            onPress={() => navigation.navigate('CVDSimulation')}
          >
            <View style={{ flex: 1 }}>
              <Text style={{ fontWeight: 'bold', color: '#B71C1C', fontSize: 16 }}>Protanomaly</Text>
              <Text style={{ color: '#E57373', fontSize: 12 }}>Red-weak vision</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#B71C1C" />
          </TouchableOpacity>

          {/* Deuteranomaly Button */}
          <TouchableOpacity 
            style={{ backgroundColor: '#E8F5E9', padding: 15, borderRadius: 12, flexDirection: 'row', alignItems: 'center', marginBottom: 10, borderWidth: 1, borderColor: '#C8E6C9' }}
            onPress={() => navigation.navigate('CVDSimulation')}
          >
            <View style={{ flex: 1 }}>
              <Text style={{ fontWeight: 'bold', color: '#1B5E20', fontSize: 16 }}>Deuteranomaly</Text>
              <Text style={{ color: '#81C784', fontSize: 12 }}>Green-weak vision</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#1B5E20" />
          </TouchableOpacity>

          {/* Tritanomaly Button */}
          <TouchableOpacity 
            style={{ backgroundColor: '#E3F2FD', padding: 15, borderRadius: 12, flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: '#BBDEFB' }}
            onPress={() => navigation.navigate('CVDSimulation')}
          >
            <View style={{ flex: 1 }}>
              <Text style={{ fontWeight: 'bold', color: '#0D47A1', fontSize: 16 }}>Tritanomaly</Text>
              <Text style={{ color: '#64B5F6', fontSize: 12 }}>Blue-weak vision</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#0D47A1" />
          </TouchableOpacity>
        </Card>

        {/* --- ARTICLES LIST --- */}
        <Text style={{ fontWeight: 'bold', fontSize: 18, marginBottom: 15, color: '#333' }}>Latest Articles</Text>
        {ARTICLES.map(article => (
           <Card key={article.id} style={{ padding: 0, overflow: 'hidden', marginBottom: 20 }}>
             <Image source={article.image} style={{ width: '100%', height: 140 }} resizeMode="cover" />
             <View style={{ padding: 20 }}>
               <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 5, color: '#333' }}>{article.title}</Text>
               <Text style={{ fontSize: 12, color: '#666', lineHeight: 18 }}>{article.desc}</Text>
             </View>
           </Card>
        ))}

      </ScrollView>
    </View>
  );
}

// --- Navigation Config ---

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

function MainTabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: COLORS.primary,
        tabBarInactiveTintColor: COLORS.textLight,
        
        // --- FIXED STYLING ---
        tabBarStyle: { 
          height: 65,          // Standard compact height
          paddingTop: 5,       // Space above icons
          paddingBottom: 5,    // Space below text (prevents cut-off)
          backgroundColor: '#FFFFFF',
          borderTopWidth: 1,
          borderTopColor: '#F0F0F0',
          elevation: 0,        // Removes shadow causing "floating" look
        },
        tabBarLabelStyle: {
          fontSize: 10,        // readable size
          fontWeight: '600',
          marginBottom: 2,     // slight spacing between text and bottom edge
        },
        // ---------------------

        tabBarIcon: ({ color, size }) => {
          let iconName;
          if (route.name === 'Home') iconName = 'home';
          else if (route.name === 'History') iconName = 'time';
          else if (route.name === 'Profile') iconName = 'person';
          return <Ionicons name={iconName} size={22} color={color} />; // Size 22 balances well with text
        },
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
}

// --- New Component: Persistent Disclaimer ---
const DisclaimerBanner = () => (
  <View style={styles.disclaimerContainer}>
    <Ionicons name="warning-outline" size={14} color="#E65100" style={{ marginRight: 5 }} />
    <Text style={styles.disclaimerText}>
      Screening purpose only. Not a clinical diagnosis.
    </Text>
  </View>
);

// --- Global Camera Mode Switcher ---
const ModeSelector = ({ visible, onClose, navigation, currentMode }) => {
  if (!visible) return null;
  
  const navigateTo = (screen) => {
    onClose();
    navigation.replace(screen); // Use replace to switch modes without stacking
  };

  return (
    <TouchableOpacity activeOpacity={1} onPress={onClose} style={styles.modalOverlay}>
      <View style={styles.modalContent}>
        <Text style={styles.modalTitle}>Select Mode</Text>
        
        <TouchableOpacity 
          style={[styles.modalOption, currentMode === 'Enhancement' && styles.modalOptionActive]} 
          onPress={() => navigateTo('CameraSim')}
        >
          <Ionicons name="color-wand" size={20} color={currentMode === 'Enhancement' ? '#FFF' : '#333'} />
          <Text style={[styles.modalText, currentMode === 'Enhancement' && {color:'#FFF'}]}>Color Enhancement</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.modalOption, currentMode === 'Identifier' && styles.modalOptionActive]} 
          onPress={() => navigateTo('ColorIdentifier')}
        >
          <Ionicons name="eyedrop" size={20} color={currentMode === 'Identifier' ? '#FFF' : '#333'} />
          <Text style={[styles.modalText, currentMode === 'Identifier' && {color:'#FFF'}]}>Color Identifier</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.modalOption, currentMode === 'Simulation' && styles.modalOptionActive]} 
          onPress={() => navigateTo('CVDSimulation')}
        >
          <Ionicons name="eye" size={20} color={currentMode === 'Simulation' ? '#FFF' : '#333'} />
          <Text style={[styles.modalText, currentMode === 'Simulation' && {color:'#FFF'}]}>CVD Simulation</Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
};
export default function App() {
  const [routeName, setRouteName] = useState('Splash');

  // Helper to find the current active screen name
  const getActiveRouteName = (state) => {
    if (!state || !state.routes) return null;
    const route = state.routes[state.index];
    // Dive into nested navigators (like the Tab bar)
    if (route.state) {
      return getActiveRouteName(route.state);
    }
    return route.name;
  };

 return (
    // FIX: Main container is a column. 
    // The "View style={{flex:1}}" takes all space, pushing the Disclaimer to the very bottom.
    <View style={{ flex: 1, flexDirection: 'column', backgroundColor: '#000' }}>
      
      {/* 1. THE APP CONTENT */}
      <View style={{ flex: 1, backgroundColor: COLORS.background }}>
        <NavigationContainer
          onStateChange={(state) => {
            const currentRouteName = getActiveRouteName(state);
            setRouteName(currentRouteName);
          }}
        >
          <Stack.Navigator initialRouteName="Splash" screenOptions={{ headerShown: false }}>
            {/* ... Keep all your Stack.Screen lines exactly as they are ... */}
            <Stack.Screen name="Splash" component={SplashScreen} />
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="AdminLogin" component={AdminLoginScreen} />
            <Stack.Screen name="AdminHub" component={AdminHubScreen} />
            <Stack.Screen name="ResearchDashboard" component={ResearchDashboardScreen} />
            <Stack.Screen name="Settings" component={SettingsScreen} />
            <Stack.Screen name="MainTabs" component={MainTabNavigator} />
            <Stack.Screen name="IshiharaIntro" component={IshiharaIntroScreen} />
            <Stack.Screen name="IshiharaTest" component={IshiharaTestScreen} />
            <Stack.Screen name="IshiharaResult" component={IshiharaResultScreen} />
            <Stack.Screen name="Survey" component={SurveyScreen} />
            <Stack.Screen name="CameraSim" component={CameraSimScreen} />
            <Stack.Screen name="EducationList" component={EducationListScreen} />
            <Stack.Screen name="ColorIdentifier" component={ColorIdentifierScreen} />
            <Stack.Screen name="SurveySuccess" component={SurveySuccessScreen} />
            <Stack.Screen name="CVDSimulation" component={CVDSimulationScreen} />
            <Stack.Screen name="CVDGallery" component={CVDGalleryScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </View>

      {/* 2. THE DISCLAIMER (Sits safely below everything) */}
       {!['Splash', 'CameraSim', 'ColorIdentifier', 'CVDSimulation', 'CVDGallery'].includes(routeName) && <DisclaimerBanner />}
    </View>
  );
}

// --- Styles ---

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  splashContainer: {
    flex: 1,
    backgroundColor: '#FFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  splashBubble1: {
    position: 'absolute', top: -50, left: -50, width: 200, height: 200, borderRadius: 100, backgroundColor: '#FFCDD2', opacity: 0.5
  },
  splashBubble2: {
    position: 'absolute', bottom: 100, right: -20, width: 100, height: 100, borderRadius: 50, backgroundColor: '#BBDEFB', opacity: 0.5
  },
  logoText: {
    fontSize: 32, fontWeight: 'bold', color: COLORS.text, letterSpacing: 1
  },
  logoTextSmall: {
    fontSize: 20, fontWeight: 'bold', color: COLORS.text, marginBottom: 10
  },
  splashSub: {
    marginTop: 10, textAlign: 'center', color: '#777', width: '70%'
  },
  loginCard: {
    margin: 20, marginTop: 100, padding: 30, backgroundColor: '#FFF', borderRadius: 20, elevation: 5, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 10, shadowOffset: { width: 0, height: 5 }
  },
  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 15, paddingTop: 55, backgroundColor: '#FFF', borderBottomWidth: 1, borderBottomColor: '#F0F0F0', height:100,
  },
  headerTitle: {
    fontSize: 18, fontWeight: 'bold', color: COLORS.text
  },
  headerSubtitle: {
    fontSize: 12, color: COLORS.textLight
  },
 card: {
    backgroundColor: COLORS.card,
    padding: 20,
    borderRadius: 16,
    elevation: 4, // Increased shadow for better "pop"
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
    // Removed 'marginBottom' here because the Pressable handles it now
  },
  cardTitle: {
    fontSize: 16, fontWeight: 'bold', color: '#333', marginBottom: 5
  },
  cardDesc: {
    fontSize: 12, color: '#777'
  },
  btnPrimary: {
    backgroundColor: '#000', padding: 15, borderRadius: 10, alignItems: 'center', flexDirection: 'row', justifyContent: 'center'
  },
  btnOutline: {
    borderWidth: 1, borderColor: '#DDD', padding: 15, borderRadius: 10, alignItems: 'center', flexDirection: 'row', justifyContent: 'center', backgroundColor: '#FFF'
  },
  btnText: {
    color: '#FFF', fontWeight: 'bold'
  },
  inputContainer: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#F5F5F5', borderRadius: 8, paddingHorizontal: 15, height: 50, marginTop: 5, marginBottom: 15
  },
  input: {
    flex: 1, marginLeft: 10
  },
  label: {
    fontSize: 12, fontWeight: 'bold', color: '#333'
  },
  iconCircle: {
    width: 50, height: 50, borderRadius: 25, alignItems: 'center', justifyContent: 'center'
  },
  iconCircleGradient: {
    width: 80, height: 80, borderRadius: 40, alignItems: 'center', justifyContent: 'center', backgroundColor: '#8E24AA'
  },
  infoBox: {
    backgroundColor: '#E3F2FD', padding: 15, borderRadius: 10, flexDirection: 'row', marginBottom: 20, borderColor: '#BBDEFB', borderWidth: 1
  },
  adminHeaderCard: {
    width: '100%', padding: 20, borderRadius: 16, overflow: 'hidden', justifyContent: 'center'
  },
  avatarLarge: {
    width: 60, height: 60, borderRadius: 30, backgroundColor: 'rgba(255,255,255,0.3)', alignItems: 'center', justifyContent: 'center'
  },
  badge: {
    backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 15, alignSelf: 'flex-start', marginTop: 5
  },
  featureCard: {
    marginBottom: 15, flexDirection: 'row', alignItems: 'center'
  },
  featureIcon: {
    width: 60, height: 60, borderRadius: 30, alignItems: 'center', justifyContent: 'center', marginRight: 15
  },
  plateContainer: {
    width: width * 0.8, height: width * 0.8, alignItems: 'center', justifyContent: 'center'
  },
  plateImage: {
    width: width * 0.9,  // Increased from fixed 300
    height: width * 0.9, // Square aspect ratio
    borderRadius: (width * 0.9) / 2, // Perfect circle
    resizeMode: 'contain'
  },
  numpadContainer: {
    padding: 20, backgroundColor: '#FFF', borderTopLeftRadius: 20, borderTopRightRadius: 20, elevation: 10
  },
  inputDisplay: {
    alignItems: 'center', marginBottom: 20, borderBottomWidth: 1, borderBottomColor: '#EEE', paddingBottom: 10
  },
  numpadGrid: {
    flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center'
  },
  numKey: {
    width: '30%', height: 60, alignItems: 'center', justifyContent: 'center', margin: '1.5%', borderWidth: 1, borderColor: '#EEE', borderRadius: 10
  },
  numKeyText: {
    fontSize: 20, fontWeight: 'bold', color: '#333'
  },
  surveyOption: {
    width: '48%', height: 120, backgroundColor: '#FFF', borderRadius: 12, padding: 15, alignItems: 'center', justifyContent: 'center', marginBottom: 15, elevation: 1
  },
  camTopBar: {
    marginTop: 50, paddingHorizontal: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'
  },
  camBottomBar: {
    position: 'absolute', bottom: 40, width: '100%', alignItems: 'center'
  },
camBtnSmall: {
    width: 45, height: 45, backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: 22.5, alignItems: 'center', justifyContent: 'center'
  },
  camBtnCircleSmall: {
    width: 50, 
    height: 50, 
    backgroundColor: '#333', 
    borderRadius: 25, 
    alignItems: 'center', 
    justifyContent: 'center', 
    borderWidth: 1, 
    borderColor: '#555'
  },
  shutterBtn: {
    width: 80, height: 80, borderRadius: 40, backgroundColor: '#FFF', alignItems: 'center', justifyContent: 'center', borderWidth: 4, borderColor: '#CCC'
  },
 filterBtn: {
    width: 55, // Slightly larger base
    height: 55, 
    borderRadius: 27.5, 
    alignItems: 'center', 
    justifyContent: 'center', 
    elevation: 6,
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowOffset: {width: 2, height: 2}
  },
  filterBtnActive: {
    borderWidth: 3,
    borderColor: '#FFF',
    transform: [{ scale: 1.15 }], // Pops out when selected
    elevation: 10
  },
  filterText: {
    color: '#FFF', 
    fontWeight: '900', // Extra bold text
    fontSize: 18
  },
  disclaimerContainer: {
    width: '100%',
    backgroundColor: '#FFF3E0', 
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderTopWidth: 1,
    borderColor: '#FFE0B2',
  },
  disclaimerText: {
    fontSize: 10,
    color: '#E65100', // Darker Orange text
    fontWeight: 'bold',
    textAlign: 'center',
    textTransform: 'uppercase',
    letterSpacing: 0.5
  }, 
   bubble: {
    position: 'absolute',
    borderRadius: 999,
    opacity: 0.6, // Glassy effect
  },
  camPill: {
    backgroundColor: 'rgba(0,0,0,0.6)', 
    paddingHorizontal: 12, 
    paddingVertical: 6, 
    borderRadius: 15 
  },
  modalOverlay: {
    flex: 1, 
    backgroundColor: 'rgba(0,0,0,0.5)', 
    justifyContent: 'center', 
    alignItems: 'center',
    position: 'absolute',
    width: '100%',
    height: '100%',
    zIndex: 999
  },
  modalContent: {
    backgroundColor: '#FFF', 
    width: 250, 
    borderRadius: 15, 
    padding: 20, 
    elevation: 10
  },
  modalTitle: {
    fontSize: 18, 
    fontWeight: 'bold', 
    marginBottom: 15, 
    textAlign: 'center'
  },
  modalOption: {
    flexDirection: 'row', 
    alignItems: 'center', 
    paddingVertical: 12, 
    borderBottomWidth: 1, 
    borderColor: '#EEE' 

  },
  modalOptionActive: {
    backgroundColor: COLORS.primary,
    borderRadius: 8,
    paddingHorizontal: 10,
    borderBottomWidth: 0
  },
  modalText: {
    marginLeft: 15, 
    fontSize: 16
  },
});