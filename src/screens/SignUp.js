import { Ionicons } from "@expo/vector-icons";
import { doc, setDoc } from "firebase/firestore"; // ADDED THIS
import { MotiView } from "moti";
import BackgroundBubbles from "../components/BackgroundBubbles";
import { useState } from "react";
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
} from "react-native";
import {
  auth, // ADDED THIS
  createUserWithEmailAndPassword,
  db,
  sendEmailVerification,
} from "../../firebaseConfig";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

const APP_VERSION = "v1.0.0-beta";

export default function SignUp({ navigation }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [authKey, setAuthKey] = useState("");
  const [showAuthKey, setShowAuthKey] = useState(false);
  const [focusedField, setFocusedField] = useState(null);

  const handleSignUp = async () => {
    const targetEmail = email.trim();
    if (!targetEmail || !password || !confirmPassword || !authKey) {
      Alert.alert(
        "Missing Fields",
        "Please fill in all fields including the authorization key.",
      );
      return;
    }
    if (authKey !== "PERI_DEV_2026") {
      Alert.alert(
        "Invalid Authorization Key",
        "Registration is restricted to authorized PERI researchers and admins.",
      );
      return;
    }
    if (password !== confirmPassword) {
      Alert.alert("Password Mismatch", "Passwords do not match.");
      return;
    }

    const passRegex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{6,14}$/;
    if (!passRegex.test(password)) {
      Alert.alert(
        "Weak Password",
        "Use 6-14 characters with uppercase, lowercase, number, and special character.",
      );
      return;
    }

    setLoading(true);
    try {
      // 1. Create the Auth User
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        targetEmail,
        password,
      );

      // 2. CREATE FIRESTORE DOCUMENT (The "Badge")
      // This ensures every user has a role record indexed by their UID
      await setDoc(doc(db, "users", userCredential.user.uid), {
        email: targetEmail,
        role: "researcher",
        createdAt: new Date().toISOString(),
      });

      // 3. Send Verification (RA 10173 Compliance)
      await sendEmailVerification(userCredential.user);

      Alert.alert(
        "Verification Sent",
        `A secure link has been sent to ${targetEmail}. Please verify your email before logging in.`,
        [{ text: "Go to Login", onPress: () => navigation.replace("Login") }],
      );
    } catch (err) {
      if (err.code === "auth/email-already-in-use") {
        Alert.alert("Account Exists", "This email is already registered.");
      } else {
        Alert.alert("Sign Up Error", err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <BackgroundBubbles />
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <MotiView
          from={{ opacity: 0, translateY: -20 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 600 }}
          style={styles.logoWrap}
        >
          <Image
            source={require("../../assets/logo_stack.png")}
            style={styles.logo}
            resizeMode="contain"
          />
        </MotiView>
        <MotiView
          from={{ opacity: 0, translateY: 30 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 800, delay: 150 }}
          style={styles.card}
        >
          <Text style={styles.title}>Create Account</Text>
          <Text style={styles.subtitle}>
            Join ReColor to track your colour perception
          </Text>
          {/* EMAIL FIELD */}
          <View style={styles.inputGroup}>
            <Text style={styles.fieldLabel}>Clinical Email Address</Text>
            <View style={[
              styles.inputRow,
              focusedField === "email" && { borderColor: COLORS.primary }
            ]}>
              <Ionicons
                name="mail-outline"
                size={18}
                color={focusedField === "email" ? COLORS.primary : COLORS.textLight}
              />
              <TextInput
                style={styles.input}
                placeholder="researcher@peri.org"
                value={email}
                onChangeText={setEmail}
                autoCapitalize="none"
                keyboardType="email-address"
                onFocus={() => setFocusedField("email")}
                onBlur={() => setFocusedField(null)}
              />
            </View>
            <Text style={styles.fieldHelper}>Your registered institutional or clinical email.</Text>
          </View>

          {/* PASSWORD FIELD */}
          <View style={styles.inputGroup}>
            <Text style={styles.fieldLabel}>Security Password</Text>
            <View style={[
              styles.inputRow,
              focusedField === "password" && { borderColor: COLORS.primary }
            ]}>
              <Ionicons
                name="key-outline"
                size={18}
                color={focusedField === "password" ? COLORS.primary : COLORS.textLight}
              />
              <TextInput
                style={styles.input}
                placeholder="Enter password"
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPass}
                onFocus={() => setFocusedField("password")}
                onBlur={() => setFocusedField(null)}
              />
              <TouchableOpacity onPress={() => setShowPass(!showPass)}>
                <Ionicons
                  name={showPass ? "eye-off-outline" : "eye-outline"}
                  size={18}
                  color={focusedField === "password" ? COLORS.primary : COLORS.textLight}
                />
              </TouchableOpacity>
            </View>
            <Text style={styles.fieldHelper}>6-14 characters with uppercase, lowercase, number & symbol.</Text>
          </View>

          {/* CONFIRM PASSWORD FIELD */}
          <View style={styles.inputGroup}>
            <Text style={styles.fieldLabel}>Confirm Password</Text>
            <View style={[
              styles.inputRow,
              focusedField === "confirmPassword" && { borderColor: COLORS.primary }
            ]}>
              <Ionicons
                name="shield-checkmark-outline"
                size={18}
                color={focusedField === "confirmPassword" ? COLORS.primary : COLORS.textLight}
              />
              <TextInput
                style={styles.input}
                placeholder="Re-enter password"
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry={!showConfirm}
                onFocus={() => setFocusedField("confirmPassword")}
                onBlur={() => setFocusedField(null)}
              />
              <TouchableOpacity onPress={() => setShowConfirm(!showConfirm)}>
                <Ionicons
                  name={showConfirm ? "eye-off-outline" : "eye-outline"}
                  size={18}
                  color={focusedField === "confirmPassword" ? COLORS.primary : COLORS.textLight}
                />
              </TouchableOpacity>
            </View>
            <Text style={styles.fieldHelper}>Must match the security password entered above.</Text>
          </View>

          {/* AUTHORIZATION KEY FIELD */}
          <View style={styles.inputGroup}>
            <Text style={styles.fieldLabel}>PERI Authorization Key</Text>
            <View style={[
              styles.inputRow,
              focusedField === "authKey" && { borderColor: COLORS.primary }
            ]}>
              <Ionicons
                name="lock-closed-outline"
                size={18}
                color={focusedField === "authKey" ? COLORS.primary : COLORS.textLight}
              />
              <TextInput
                style={styles.input}
                placeholder="Enter clinical passcode"
                value={authKey}
                onChangeText={setAuthKey}
                secureTextEntry={!showAuthKey}
                autoCapitalize="none"
                onFocus={() => setFocusedField("authKey")}
                onBlur={() => setFocusedField(null)}
              />
              <TouchableOpacity onPress={() => setShowAuthKey(!showAuthKey)}>
                <Ionicons
                  name={showAuthKey ? "eye-off-outline" : "eye-outline"}
                  size={18}
                  color={focusedField === "authKey" ? COLORS.primary : COLORS.textLight}
                />
              </TouchableOpacity>
            </View>
            <Text style={styles.fieldHelper}>Secret passcode issued to assigned researchers and admins.</Text>
          </View>
          <TouchableOpacity
            style={styles.signUpBtn}
            onPress={handleSignUp}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <Text style={styles.signUpBtnText}>Create Account</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.loginLink}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.loginLinkText}>
              Not PERI Researcher/Admin?{" "}
              <Text style={{ color: COLORS.primary, fontWeight: "700" }}>
                Go Back
              </Text>
            </Text>
          </TouchableOpacity>
        </MotiView>
        <View style={styles.versionBadge}>
          <Text style={styles.versionText}>{APP_VERSION}</Text>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.xl,
  },
  logoWrap: { alignItems: "center", marginBottom: SPACING.lg },
  logo: { width: 180, height: 100 },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.xl,
    padding: SPACING.lg,
    ...SHADOW.lg,
  },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: COLORS.text,
    textAlign: "center",
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 13,
    color: COLORS.textLight,
    textAlign: "center",
    marginBottom: SPACING.lg,
  },
  inputGroup: {
    marginBottom: SPACING.md,
  },
  fieldLabel: {
    fontSize: 12,
    fontWeight: "800",
    color: COLORS.text,
    marginBottom: 6,
    letterSpacing: 0.3,
  },
  fieldHelper: {
    fontSize: 10,
    fontWeight: "600",
    color: COLORS.textLight,
    marginTop: 4,
    opacity: 0.7,
    lineHeight: 14,
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: 12,
    backgroundColor: COLORS.background,
    gap: SPACING.sm,
  },
  input: { flex: 1, fontSize: 15, color: COLORS.text },
  signUpBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.md,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: SPACING.md,
    ...SHADOW.sm,
  },
  signUpBtnText: { color: "#FFF", fontWeight: "700", fontSize: 16 },
  loginLink: {
    alignItems: "center",
    marginTop: SPACING.md,
    padding: SPACING.sm,
  },
  loginLinkText: { fontSize: 14, color: COLORS.textLight },
  versionBadge: {
    alignSelf: "center",
    marginTop: SPACING.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: COLORS.surfaceAlt,
  },
  versionText: { fontSize: 11, color: COLORS.primary, fontWeight: "600" },
});
