import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Haptics from "expo-haptics";
import { doc, getDoc } from "firebase/firestore";
import { MotiView } from "moti";
import { useEffect, useState } from "react";
import BackgroundBubbles from "../components/BackgroundBubbles";
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
  const [showAdminPortal, setShowAdminPortal] = useState(false);
  const [localDiagnosis, setLocalDiagnosis] = useState("Unscreened");
  const [localCount, setLocalCount] = useState(0);
  const [focusedField, setFocusedField] = useState(null);

  useEffect(() => {
    AsyncStorage.getItem("@recolor_latest_diagnosis")
      .then((diag) => {
        if (diag) setLocalDiagnosis(diag);
      })
      .catch(() => {});

    AsyncStorage.getItem("@recolor_local_history")
      .then((historyRaw) => {
        if (historyRaw) {
          const parsed = JSON.parse(historyRaw);
          if (Array.isArray(parsed)) {
            setLocalCount(parsed.length);
          }
        }
      })
      .catch(() => {});
  }, []);

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

        {/* GUEST ACCESS CARD */}
        <MotiView
          from={{ opacity: 0, translateY: 30 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 800, delay: 150 }}
          style={styles.card}
        >
          <Text style={styles.welcomeTitle}>
            Welcome to <Text style={{ color: "#EF4444" }}>R</Text>e<Text style={{ color: "#10B981" }}>C</Text>olo<Text style={{ color: "#3B82F6" }}>r</Text>
          </Text>
          <Text style={styles.welcomeSub}>
            Take Ishihara tests, calibrate camera colors, and apply color vision enhancement filters.
          </Text>


          <View style={styles.localSessionBadge}>
            <View style={styles.pulseDot} />
            <Text style={styles.localSessionBadgeText}>
              {localCount > 0
                ? `Active Session • ${localDiagnosis} (${localCount} local logs)`
                : "New Session • Guest Mode"}
            </Text>
          </View>

          <TouchableOpacity
            activeOpacity={0.85}
            style={[styles.guestLauncherCard, { backgroundColor: COLORS.primary }]}
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
              navigation.replace("MainTabs");
            }}
          >
            <View style={styles.launcherIconWrapper}>
              <Ionicons name="color-palette" size={24} color={COLORS.primary} />
            </View>
            <View style={styles.launcherTextWrapper}>
              <Text style={styles.launcherTitle}>Begin Guest Session</Text>
              <Text style={styles.launcherDesc}>Launch Ishihara tests, camera tools & color identifier</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#FFF" />
          </TouchableOpacity>
        </MotiView>

        {/* PERI CLINICAL DRAWER TOGGLE */}
        <TouchableOpacity
          style={styles.portalToggle}
          onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
            setShowAdminPortal(!showAdminPortal);
          }}
        >
          <Text style={styles.portalToggleText}>
            PERI Clinical Portal
          </Text>
          <Ionicons
            name={showAdminPortal ? "chevron-up-outline" : "chevron-down-outline"}
            size={16}
            color={COLORS.primary}
          />
        </TouchableOpacity>

        {/* CLINICAL LOGIN FORM */}
        {showAdminPortal && (
          <MotiView
            from={{ opacity: 0, translateY: -10, scale: 0.97 }}
            animate={{ opacity: 1, translateY: 0, scale: 1 }}
            transition={{ type: "timing", duration: 250 }}
            style={[styles.card, { marginTop: SPACING.md }]}
          >
            <Text style={styles.welcomeTitle}>Clinical Sign-In</Text>
            <Text style={styles.welcomeSub}>
              Access research dashboards and collection logs.
            </Text>
            <View style={[
              styles.inputRow,
              focusedField === "email" && { borderColor: COLORS.primary }
            ]}>
              <Ionicons
                name="person-outline"
                size={18}
                color={focusedField === "email" ? COLORS.primary : COLORS.textLight}
              />
              <TextInput
                style={styles.input}
                placeholder="Email Address"
                value={email}
                onChangeText={setEmail}
                autoCapitalize="none"
                keyboardType="email-address"
                onFocus={() => setFocusedField("email")}
                onBlur={() => setFocusedField(null)}
              />
            </View>
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
                placeholder="Password"
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPass}
                autoCapitalize="none"
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
                <ActivityIndicator color={COLORS.primary} />
              ) : (
                <Text style={styles.loginBtnText}>Login</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.createAccountLink}
              onPress={() => navigation.navigate("SignUp")}
            >
              <Text style={styles.createAccountText}>
                PERI Researcher / Admin?{" "}
                <Text style={{ color: COLORS.primary, fontWeight: "700" }}>
                  Create an account
                </Text>
              </Text>
            </TouchableOpacity>
          </MotiView>
        )}

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
    lineHeight: 18,
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
    backgroundColor: COLORS.surfaceAlt,
    borderRadius: RADIUS.md,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: SPACING.md,
    ...SHADOW.sm,
  },
  loginBtnText: { color: COLORS.primary, fontWeight: "700", fontSize: 16 },
  createAccountLink: {
    alignItems: "center",
    marginTop: SPACING.md,
    padding: SPACING.sm,
  },
  createAccountText: { fontSize: 14, color: COLORS.textLight },
  guestLauncherCard: {
    flexDirection: "row",
    alignItems: "center",
    borderRadius: RADIUS.lg,
    padding: SPACING.lg,
    marginTop: SPACING.md,
    ...SHADOW.md,
  },
  launcherIconWrapper: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    marginRight: SPACING.md,
    ...SHADOW.sm,
  },
  launcherTextWrapper: {
    flex: 1,
    marginRight: 8,
  },
  launcherTitle: {
    fontSize: 16,
    fontWeight: "900",
    color: "#FFFFFF",
  },
  launcherDesc: {
    fontSize: 12,
    color: "rgba(255, 255, 255, 0.85)",
    marginTop: 4,
    lineHeight: 16,
  },
  localSessionBadge: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    backgroundColor: COLORS.surfaceAlt,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
    marginTop: SPACING.sm,
    gap: 6,
  },
  localSessionBadgeText: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.primary,
  },
  portalToggle: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    marginTop: SPACING.lg,
    paddingVertical: SPACING.sm,
  },
  portalToggleText: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  versionBadge: {
    alignSelf: "center",
    marginTop: SPACING.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: COLORS.surfaceAlt,
  },
  versionText: { fontSize: 11, color: COLORS.primary, fontWeight: "600" },
  pulseDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.success,
  },
});
