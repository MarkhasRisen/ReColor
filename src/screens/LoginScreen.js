import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { doc, getDoc } from "firebase/firestore";
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
  auth,
  db,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
  signOut,
} from "../../firebaseConfig";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

const APP_VERSION = "v1.0.0-beta";

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPass, setShowPass] = useState(false);

  const handleForgotPassword = async () => {
    const target = email.trim();
    if (!target || !target.includes("@")) {
      Alert.alert(
        "Valid Email Required",
        "Please enter your full email address first.",
      );
      return;
    }
    try {
      await sendPasswordResetEmail(auth, target);
      Alert.alert(
        "Email Sent",
        `A reset link has been sent to ${target}. Check your Inbox and Spam.`,
      );
    } catch (err) {
      Alert.alert(
        "Request Failed",
        "Check your connection or if the email is correct.",
      );
    }
  };

  const handleLogin = async () => {
    if (!email.trim() || !password) {
      Alert.alert("Missing Fields", "Please enter your email and password.");
      return;
    }

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    setLoading(true);

    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        email.trim(),
        password,
      );
      const user = userCredential.user;

      // IDENTITY VERIFICATION BYPASS FOR DEV ACCOUNT
      const isDevAccount =
        email.trim().toLowerCase() === "recolor.dev@gmail.com";

      if (!user.emailVerified && !isDevAccount) {
        await signOut(auth);
        Alert.alert(
          "Verification Required",
          "Please verify your email before signing in.",
          [{ text: "OK" }],
        );
        setLoading(false);
        return;
      }

      const userDoc = await getDoc(doc(db, "users", user.uid));
      if (userDoc.exists()) {
        const role = userDoc.data().role;
        if (role === "admin" || role === "researcher") {
          navigation.replace("AdminHub", { role });
        } else {
          navigation.replace("MainTabs");
        }
      } else {
        navigation.replace("MainTabs");
      }
    } catch (err) {
      Alert.alert("Login Failed", "Incorrect credentials or network error.");
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
          <Text style={styles.welcomeTitle}>Welcome Back</Text>
          <Text style={styles.welcomeSub}>
            Sign in to enhance your color perception
          </Text>
          <View style={styles.inputRow}>
            <Ionicons
              name="person-outline"
              size={18}
              color={COLORS.textLight}
            />
            <TextInput
              style={styles.input}
              placeholder="Email Address"
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
              autoCapitalize="none"
            />
            <TouchableOpacity onPress={() => setShowPass(!showPass)}>
              <Ionicons
                name={showPass ? "eye-off-outline" : "eye-outline"}
                size={18}
                color={COLORS.textLight}
              />
            </TouchableOpacity>
          </View>
          <TouchableOpacity
            onPress={handleForgotPassword}
            style={styles.forgotBtn}
          >
            <Text style={styles.forgotText}>Forgot password?</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.loginBtn}
            onPress={handleLogin}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <Text style={styles.loginBtnText}>Login</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.createAccountLink}
            onPress={() => navigation.navigate("SignUp")}
          >
            <Text style={styles.createAccountText}>
              New here?{" "}
              <Text style={{ color: COLORS.primary, fontWeight: "700" }}>
                Create an account
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
  welcomeTitle: {
    fontSize: 24,
    fontWeight: "800",
    color: COLORS.text,
    textAlign: "center",
    marginBottom: 4,
  },
  welcomeSub: {
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
  forgotBtn: {
    alignSelf: "flex-end",
    marginTop: 4,
    marginBottom: 2,
    padding: 4,
  },
  forgotText: { fontSize: 13, color: COLORS.primary, fontWeight: "600" },
  loginBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.md,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: SPACING.md,
    ...SHADOW.sm,
  },
  loginBtnText: { color: "#FFF", fontWeight: "700", fontSize: 16 },
  createAccountLink: {
    alignItems: "center",
    marginTop: SPACING.md,
    padding: SPACING.sm,
  },
  createAccountText: { fontSize: 14, color: COLORS.textLight },
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
