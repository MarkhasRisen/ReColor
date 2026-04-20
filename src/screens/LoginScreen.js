import { Ionicons } from '@expo/vector-icons';
import * as Google from 'expo-auth-session/providers/google';
import * as WebBrowser from 'expo-web-browser';
import { MotiView } from 'moti';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  GoogleAuthProvider,
  auth,
  createUserWithEmailAndPassword,
  signInWithCredential,
  signInWithEmailAndPassword,
} from '../../firebaseConfig';
import { COLORS, RADIUS, SHADOW, SPACING } from '../theme/colors';

WebBrowser.maybeCompleteAuthSession();

// To enable Google Sign-In:
// 1. Firebase Console → Project Settings → General → Your Apps → Add Android app
// 2. Register SHA-1 fingerprint (from: eas credentials or keytool)
// 3. Download google-services.json and place in project root
// 4. Replace placeholder below with your OAuth Web Client ID
const GOOGLE_WEB_CLIENT_ID = '912342727884-YOUR_OAUTH_WEB_CLIENT_ID.apps.googleusercontent.com';
const GOOGLE_CONFIGURED = !GOOGLE_WEB_CLIENT_ID.includes('YOUR_OAUTH');

const APP_VERSION = 'v1.0.4-thesis';

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPass, setShowPass] = useState(false);

  const [request, response, promptAsync] = Google.useAuthRequest({
    webClientId: GOOGLE_WEB_CLIENT_ID,
    androidClientId: GOOGLE_WEB_CLIENT_ID,
  });

  useEffect(() => {
    if (response?.type === 'success') {
      const { id_token } = response.params;
      if (!id_token) { Alert.alert('Sign-in failed', 'Google did not return a token.'); return; }
      setLoading(true);
      const credential = GoogleAuthProvider.credential(id_token);
      signInWithCredential(auth, credential)
        .then(() => navigation.replace('MainTabs'))
        .catch((err) => Alert.alert('Google Sign-in Failed', err.message))
        .finally(() => setLoading(false));
    } else if (response?.type === 'error') {
      Alert.alert('Sign-in Failed', response.error?.message || 'Google Sign-in was unsuccessful.');
    }
  }, [response]);

  const handleLogin = async () => {
    if (!email.trim() || !password) {
      Alert.alert('Missing Fields', 'Please enter your email and password.');
      return;
    }
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email.trim(), password);
      navigation.replace('MainTabs');
    } catch (err) {
      if (err.code === 'auth/user-not-found') {
        try {
          await createUserWithEmailAndPassword(auth, email.trim(), password);
          navigation.replace('MainTabs');
        } catch (regErr) {
          Alert.alert('Registration Error', regErr.message);
        }
      } else if (
        err.code === 'auth/invalid-credential' ||
        err.code === 'auth/wrong-password'
      ) {
        Alert.alert('Login Failed', 'Incorrect email or password.');
      } else {
        Alert.alert('Error', err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      {/* Background blobs */}
      <View style={styles.blob1} />
      <View style={styles.blob2} />
      <View style={styles.blob3} />

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Logo */}
        <MotiView
          from={{ opacity: 0, translateY: -20 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: 'spring', damping: 18 }}
          style={styles.logoWrap}
        >
          <Image
            source={require('../../assets/logo_stack.png')}
            style={styles.logo}
            resizeMode="contain"
          />
        </MotiView>

        {/* Card */}
        <MotiView
          from={{ opacity: 0, translateY: 30 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: 'spring', damping: 18, delay: 150 }}
          style={styles.card}
        >
          <Text style={styles.welcomeTitle}>Welcome Back</Text>
          <Text style={styles.welcomeSub}>Sign in to enhance your colour perception</Text>

          {/* Email */}
          <View style={styles.inputRow}>
            <Ionicons name="mail-outline" size={18} color={COLORS.textLight} />
            <TextInput
              style={styles.input}
              placeholder="you@example.com"
              placeholderTextColor={COLORS.textLight}
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
            />
          </View>

          {/* Password */}
          <View style={styles.inputRow}>
            <Ionicons name="key-outline" size={18} color={COLORS.textLight} />
            <TextInput
              style={styles.input}
              placeholder="••••••••"
              placeholderTextColor={COLORS.textLight}
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPass}
            />
            <TouchableOpacity onPress={() => setShowPass(!showPass)}>
              <Ionicons
                name={showPass ? 'eye-off-outline' : 'eye-outline'}
                size={18}
                color={COLORS.textLight}
              />
            </TouchableOpacity>
          </View>

          {/* Login button */}
          <TouchableOpacity
            style={styles.loginBtn}
            onPress={handleLogin}
            disabled={loading}
            activeOpacity={0.85}
          >
            {loading ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <Text style={styles.loginBtnText}>Login / Sign Up</Text>
            )}
          </TouchableOpacity>

          {/* Divider */}
          <View style={styles.dividerRow}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>or</Text>
            <View style={styles.dividerLine} />
          </View>

          {/* Google Sign-in */}
          <TouchableOpacity
            style={[styles.googleBtn, !GOOGLE_CONFIGURED && { opacity: 0.5 }]}
            onPress={() => {
              if (!GOOGLE_CONFIGURED) {
                Alert.alert('Not Configured', 'Google Sign-In requires SHA-1 fingerprint registration in Firebase Console. Use email login for now.');
                return;
              }
              promptAsync();
            }}
            disabled={loading}
            activeOpacity={0.85}
          >
            <Ionicons name="logo-google" size={20} color="#DB4437" />
            <Text style={styles.googleBtnText}>Continue with Google</Text>
          </TouchableOpacity>

          {/* Admin link */}
          <TouchableOpacity
            style={styles.adminLink}
            onPress={() => navigation.navigate('AdminLogin')}
          >
            <Ionicons name="lock-closed" size={13} color={COLORS.warning} />
            <Text style={styles.adminLinkText}> Admin / Expert Portal</Text>
          </TouchableOpacity>
        </MotiView>

        {/* Version badge */}
        <MotiView
          from={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 500 }}
          style={styles.versionBadge}
        >
          <Text style={styles.versionText}>{APP_VERSION}</Text>
        </MotiView>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },
  blob1: {
    position: 'absolute', top: -60, left: -60, width: 220, height: 220,
    borderRadius: 110, backgroundColor: '#FFCDD2', opacity: 0.45,
  },
  blob2: {
    position: 'absolute', bottom: -80, right: -60, width: 280, height: 280,
    borderRadius: 140, backgroundColor: '#BBDEFB', opacity: 0.4,
  },
  blob3: {
    position: 'absolute', top: '40%', right: -40, width: 150, height: 150,
    borderRadius: 75, backgroundColor: '#E1BEE7', opacity: 0.35,
  },
  scroll: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.xl,
  },
  logoWrap: { alignItems: 'center', marginBottom: SPACING.lg },
  logo: { width: 180, height: 100 },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.xl,
    padding: SPACING.lg,
    ...SHADOW.lg,
  },
  welcomeTitle: {
    fontSize: 24,
    fontWeight: '800',
    color: COLORS.text,
    textAlign: 'center',
    marginBottom: 4,
  },
  welcomeSub: {
    fontSize: 13,
    color: COLORS.textLight,
    textAlign: 'center',
    marginBottom: SPACING.lg,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: 12,
    marginBottom: SPACING.sm,
    backgroundColor: COLORS.background,
    gap: SPACING.sm,
  },
  input: {
    flex: 1,
    fontSize: 15,
    color: COLORS.text,
  },
  loginBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.md,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: SPACING.md,
    ...SHADOW.sm,
  },
  loginBtnText: { color: '#FFF', fontWeight: '700', fontSize: 16 },
  adminLink: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: SPACING.lg,
    padding: SPACING.sm,
  },
  adminLinkText: { fontSize: 13, color: COLORS.textLight },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: SPACING.md,
    gap: SPACING.sm,
  },
  dividerLine: { flex: 1, height: 1, backgroundColor: COLORS.border },
  dividerText: { fontSize: 12, color: COLORS.textLight },
  googleBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACING.sm,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: RADIUS.md,
    paddingVertical: 13,
    backgroundColor: COLORS.background,
    marginBottom: SPACING.sm,
  },
  googleBtnText: { color: COLORS.text, fontWeight: '600', fontSize: 15 },
  versionBadge: {
    alignSelf: 'center',
    marginTop: SPACING.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: COLORS.surfaceAlt,
  },
  versionText: { fontSize: 11, color: COLORS.primary, fontWeight: '600' },
});
