import { Ionicons } from "@expo/vector-icons";
import { doc, setDoc } from "firebase/firestore"; // ADDED THIS
import { MotiView } from "moti";
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

const APP_VERSION = "v1.0.5-thesis";

export default function SignUp({ navigation }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSignUp = async () => {
    const targetEmail = email.trim();
    if (!targetEmail || !password || !confirmPassword) {
      Alert.alert("Missing Fields", "Please fill in all fields.");
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
        role: "user", // Default role
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
      <View style={styles.blob1} />
      <View style={styles.blob2} />
      <View style={styles.blob3} />
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
          <View style={styles.inputRow}>
            <Ionicons name="mail-outline" size={18} color={COLORS.textLight} />
            <TextInput
              style={styles.input}
              placeholder="you@example.com"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
            />
          </View>
          <View style={styles.inputRow}>
            <Ionicons name="key-outline" size={18} color={COLORS.textLight} />
            <TextInput
              style={styles.input}
              placeholder="Password"
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!showPass}
            />
            <TouchableOpacity onPress={() => setShowPass(!showPass)}>
              <Ionicons
                name={showPass ? "eye-off-outline" : "eye-outline"}
                size={18}
                color={COLORS.textLight}
              />
            </TouchableOpacity>
          </View>
          <View style={styles.inputRow}>
            <Ionicons
              name="shield-checkmark-outline"
              size={18}
              color={COLORS.textLight}
            />
            <TextInput
              style={styles.input}
              placeholder="Confirm password"
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry={!showConfirm}
            />
            <TouchableOpacity onPress={() => setShowConfirm(!showConfirm)}>
              <Ionicons
                name={showConfirm ? "eye-off-outline" : "eye-outline"}
                size={18}
                color={COLORS.textLight}
              />
            </TouchableOpacity>
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
              Already have an account?{" "}
              <Text style={{ color: COLORS.primary, fontWeight: "700" }}>
                Log In
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
  blob1: {
    position: "absolute",
    top: -60,
    left: -60,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: "#FFCDD2",
    opacity: 0.45,
  },
  blob2: {
    position: "absolute",
    bottom: -80,
    right: -60,
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: "#BBDEFB",
    opacity: 0.4,
  },
  blob3: {
    position: "absolute",
    top: "40%",
    right: -40,
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: "#E1BEE7",
    opacity: 0.35,
  },
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
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: 12,
    marginBottom: SPACING.sm,
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
