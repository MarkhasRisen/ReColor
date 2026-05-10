import Ionicons from "@expo/vector-icons/Ionicons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";
import { collection, doc, getDoc, getDocs } from "firebase/firestore";
import { useEffect, useState } from "react";
import {
  Alert,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { auth, db, signOut } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Header from "../components/Header";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";
import { loadCalibration } from "../utils/cameraCalibration";

export default function SettingsScreen({ navigation }) {
  const [intensity, setIntensity] = useState(100);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [expertRole, setExpertRole] = useState(null);
  const [testCount, setTestCount] = useState(0);
  const [calibration, setCalibration] = useState(null);
  const user = auth.currentUser;

  useEffect(() => {
    loadSettings();
    checkExpertStatus();
    fetchTestCount();
    refreshCalibration();
    const unsub = navigation.addListener("focus", refreshCalibration);
    return unsub;
  }, [navigation]);

  const refreshCalibration = () => {
    loadCalibration().then(setCalibration);
  };

  const loadSettings = async () => {
    const savedIntensity = await AsyncStorage.getItem("@recolor_intensity");
    const savedAudio = await AsyncStorage.getItem("audio_feedback_enabled");
    if (savedIntensity) setIntensity(Number(savedIntensity));
    if (savedAudio) setAudioEnabled(savedAudio === "true");
  };

  const checkExpertStatus = async () => {
    if (user) {
      const userDoc = await getDoc(doc(db, "users", user.uid));
      if (
        userDoc.exists() &&
        (userDoc.data().role === "admin" ||
          userDoc.data().role === "researcher")
      ) {
        setExpertRole(userDoc.data().role);
      }
    }
  };

  const fetchTestCount = async () => {
    if (user) {
      try {
        const snap = await getDocs(
          collection(db, "users", user.uid, "history"),
        );
        setTestCount(snap.size);
      } catch (error) {
        console.error("Could not fetch test count:", error);
      }
    }
  };

  const showInfo = (title, message) => {
    Alert.alert(title, message);
  };

  const handleResetOnboarding = async () => {
    Alert.alert(
      "Reset Experience",
      "This will log you out and restart the app from the welcome tour. Continue?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset",
          style: "destructive",
          onPress: async () => {
            try {
              await AsyncStorage.removeItem("@recolor_onboarded");
              await signOut(auth);
              navigation.replace("Splash");
            } catch (e) {
              Alert.alert("Error", "Could not reset the application state.");
            }
          },
        },
      ],
    );
  };

  return (
    <View style={styles.root}>
      <Header title="Settings" back />
      <BackgroundBubbles />
      <ScrollView contentContainerStyle={styles.scroll}>
        {/* PROFILE CARD */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PROFILE</Text>
          <View style={styles.card}>
            <Text style={styles.subtext}>Name</Text>
            <Text style={styles.label}>
              {user?.email?.split("@")[0] || "User"}
            </Text>
            <View style={{ height: 10 }} />
            <Text style={styles.subtext}>Email</Text>
            <Text style={styles.label}>{user?.email}</Text>
          </View>
        </View>

        {/* PREFERENCES CARD */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PREFERENCES</Text>
          <View style={styles.card}>
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                marginBottom: 8,
              }}
            >
              <Text style={styles.label}>
                Color Enhancement Intensity: {Math.round(intensity)}%
              </Text>
              <TouchableOpacity
                onPress={() =>
                  showInfo(
                    "Enhancement Intensity",
                    "Adjusts the strength of the color enhancement filter applied to the camera feed.",
                  )
                }
                style={{ marginLeft: 8 }}
              >
                <Ionicons
                  name="information-circle-outline"
                  size={18}
                  color={COLORS.textLight}
                />
              </TouchableOpacity>
            </View>
            <Slider
              style={styles.slider}
              minimumValue={0}
              maximumValue={100}
              value={intensity}
              onSlidingComplete={(v) => {
                setIntensity(v);
                AsyncStorage.setItem("@recolor_intensity", v.toString());
              }}
              minimumTrackTintColor={COLORS.primary}
            />
            <View style={[styles.row, { marginTop: 20 }]}>
              <View
                style={{ flexDirection: "row", alignItems: "center", flex: 1 }}
              >
                <Text style={styles.label}>Audio Feedback</Text>
                <TouchableOpacity
                  onPress={() =>
                    showInfo(
                      "Audio Feedback",
                      "Enables spoken color names in the Color Identifier mode.",
                    )
                  }
                  style={{ marginLeft: 8 }}
                >
                  <Ionicons
                    name="information-circle-outline"
                    size={18}
                    color={COLORS.textLight}
                  />
                </TouchableOpacity>
              </View>
              <Switch
                value={audioEnabled}
                onValueChange={(v) => {
                  setAudioEnabled(v);
                  AsyncStorage.setItem("audio_feedback_enabled", v.toString());
                }}
                trackColor={{ true: COLORS.primary }}
              />
            </View>

            <TouchableOpacity
              style={[styles.row, { marginTop: 20 }]}
              onPress={() => navigation.navigate("CameraCalibration")}
              activeOpacity={0.7}
            >
              <Ionicons
                name="color-wand-outline"
                size={22}
                color={COLORS.primary}
              />
              <View style={{ flex: 1, marginLeft: 12 }}>
                <View style={{ flexDirection: "row", alignItems: "center" }}>
                  <Text style={styles.label}>Camera Calibration</Text>
                  <TouchableOpacity
                    onPress={(e) => {
                      e.stopPropagation();
                      showInfo(
                        "Camera Calibration",
                        "Corrects cross-device color drift by setting a manual white-balance reference.",
                      );
                    }}
                    style={{ marginLeft: 8, padding: 4 }}
                  >
                    <Ionicons
                      name="information-circle-outline"
                      size={18}
                      color={COLORS.textLight}
                    />
                  </TouchableOpacity>
                </View>
                <Text style={styles.subtextCount}>
                  {calibration
                    ? `Calibrated · R${calibration.rScale.toFixed(2)} G${calibration.gScale.toFixed(2)} B${calibration.bScale.toFixed(2)}`
                    : "Not calibrated · using auto white-balance"}
                </Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#CCC" />
            </TouchableOpacity>
          </View>
        </View>

        {/* DATA CARD */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>DATA</Text>
          <TouchableOpacity
            style={styles.card}
            onPress={() =>
              navigation.navigate("MainTabs", { screen: "History" })
            }
          >
            <View style={styles.row}>
              <Ionicons name="time-outline" size={24} color="#333" />
              <View style={{ flex: 1, marginLeft: 12 }}>
                <Text style={styles.label}>View Test History</Text>
                <Text style={styles.subtextCount}>
                  {testCount} {testCount === 1 ? "test" : "tests"} completed
                </Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#CCC" />
            </View>
          </TouchableOpacity>
        </View>

        {/* EXPERT UTILITIES */}
        {expertRole && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>EXPERT UTILITIES</Text>

            <TouchableOpacity
              style={styles.expertBtn}
              onPress={() =>
                navigation.navigate("AdminHub", { role: expertRole })
              }
            >
              <Ionicons
                name="shield-checkmark"
                size={20}
                color="#FFF"
                style={{ marginRight: 10 }}
              />
              <Text style={styles.expertBtnText}>Launch Expert Portal</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.expertBtn, styles.resetBtn]}
              onPress={handleResetOnboarding}
            >
              <Ionicons
                name="refresh"
                size={20}
                color={COLORS.primary}
                style={{ marginRight: 10 }}
              />
              <Text style={[styles.expertBtnText, { color: COLORS.primary }]}>
                Reset Onboarding Flow
              </Text>
            </TouchableOpacity>
          </View>
        )}

        {/* ACCOUNT CARD */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>ACCOUNT</Text>
          <TouchableOpacity
            style={[styles.expertBtn, { backgroundColor: COLORS.danger }]}
            onPress={() => auth.signOut()}
          >
            <Text style={styles.expertBtnText}>Log Out</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },
  scroll: { padding: SPACING.md },
  section: { marginBottom: 30 },
  sectionTitle: {
    fontSize: 11,
    fontWeight: "800",
    color: "#AAA",
    letterSpacing: 1.5,
    marginBottom: 12,
  },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    ...SHADOW.sm,
  },
  row: { flexDirection: "row", alignItems: "center", gap: 10 },
  label: { fontSize: 16, fontWeight: "600", color: COLORS.text },
  slider: { width: "100%", height: 40 },
  subtext: { fontSize: 11, color: "#888", marginTop: 8 },
  subtextCount: { fontSize: 12, color: "#999", marginTop: 2 },
  expertBtn: {
    backgroundColor: COLORS.primary,
    padding: 16,
    borderRadius: RADIUS.md,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 10,
    ...SHADOW.sm,
  },
  resetBtn: {
    backgroundColor: "#c8ff00",
    borderWidth: 0,
    borderColor: COLORS.primary,
  },
  expertBtnText: { color: "#FFF", fontWeight: "bold" },
});
