import Ionicons from "@expo/vector-icons/Ionicons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";
import { doc, getDoc } from "firebase/firestore";
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
import { auth, db } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Header from "../components/Header";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

export default function SettingsScreen({ navigation }) {
  const [intensity, setIntensity] = useState(1.0);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [expertRole, setExpertRole] = useState(null);

  useEffect(() => {
    loadSettings();
    checkExpertStatus();
  }, []);

  const loadSettings = async () => {
    try {
      const savedIntensity = await AsyncStorage.getItem(
        "enhancement_intensity",
      );
      const savedAudio = await AsyncStorage.getItem("audio_feedback_enabled");
      if (savedIntensity !== null) setIntensity(parseFloat(savedIntensity));
      if (savedAudio !== null) setAudioEnabled(savedAudio === "true");
    } catch (e) {
      /* Silent fail */
    }
  };

  const checkExpertStatus = async () => {
    if (auth.currentUser) {
      const userDoc = await getDoc(doc(db, "users", auth.currentUser.uid));
      if (
        userDoc.exists() &&
        (userDoc.data().role === "admin" ||
          userDoc.data().role === "researcher")
      ) {
        setExpertRole(userDoc.data().role);
      }
    }
  };

  const handleResetOnboarding = async () => {
    Alert.alert("Reset Flow", "Return to Splash screen? (Expert Only)", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Reset",
        style: "destructive",
        onPress: async () => {
          await AsyncStorage.removeItem("has_completed_onboarding");
          navigation.replace("Splash");
        },
      },
    ]);
  };

  return (
    <View style={styles.root}>
      <Header title="Settings" subtitle="App Configuration" back />
      <BackgroundBubbles />
      <ScrollView contentContainerStyle={styles.scroll}>
        {/* USER PREFERENCES */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PREFERENCES</Text>
          <View style={styles.card}>
            <View style={styles.row}>
              <Ionicons
                name="contrast-outline"
                size={20}
                color={COLORS.primary}
              />
              <Text style={styles.label}>
                Intensity: {Math.round(intensity * 100)}%
              </Text>
            </View>
            <Slider
              style={styles.slider}
              minimumValue={0}
              maximumValue={1}
              value={intensity}
              onSlidingComplete={(v) => {
                setIntensity(v);
                AsyncStorage.setItem("enhancement_intensity", v.toString());
              }}
              minimumTrackTintColor={COLORS.primary}
              thumbTintColor={COLORS.primary}
            />

            <View style={[styles.row, { marginTop: 20 }]}>
              <Ionicons
                name="volume-high-outline"
                size={20}
                color={COLORS.primary}
              />
              <Text style={[styles.label, { flex: 1 }]}>Audio Feedback</Text>
              <Switch
                value={audioEnabled}
                onValueChange={(v) => {
                  setAudioEnabled(v);
                  AsyncStorage.setItem("audio_feedback_enabled", v.toString());
                }}
                trackColor={{ true: COLORS.primary }}
              />
            </View>
            <Text style={styles.subtext}>
              Announce identified colors automatically.
            </Text>
          </View>
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
    backgroundColor: "transparent",
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  expertBtnText: { color: "#FFF", fontWeight: "bold" },
});
