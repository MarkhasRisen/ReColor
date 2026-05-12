import Ionicons from "@expo/vector-icons/Ionicons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";
import * as Haptics from "expo-haptics";
import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
} from "firebase/firestore";
import { useEffect, useState } from "react";
import {
  Alert,
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, { FadeInDown } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import BackgroundBubbles from "../components/BackgroundBubbles";

import { auth, db } from "../../firebaseConfig";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";
import { loadCalibration } from "../utils/cameraCalibration";

export default function SettingsScreen({ navigation }) {
  const insets = useSafeAreaInsets();
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

  const loadSettings = async () => {
    const savedIntensity = await AsyncStorage.getItem("filterIntensity");
    const savedAudio = await AsyncStorage.getItem("audio_feedback_enabled");
    if (savedIntensity) setIntensity(parseFloat(savedIntensity));
    if (savedAudio) setAudioEnabled(savedAudio === "true");
  };

  const saveIntensity = async (val) => {
    setIntensity(val);
    await AsyncStorage.setItem("filterIntensity", val.toString());
  };

  const checkExpertStatus = async () => {
    if (!user) return;
    const docRef = doc(db, "users", user.uid);
    const docSnap = await getDoc(docRef);
    if (docSnap.exists()) setExpertRole(docSnap.data().role);
  };

  const fetchTestCount = async () => {
    if (!user) return;
    const q = collection(db, "users", user.uid, "history");
    const snap = await getDocs(q);
    setTestCount(snap.size);
  };

  const refreshCalibration = async () => {
    const cal = await loadCalibration();
    setCalibration(cal);
  };
  const handleResetOnboarding = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      "Reset Tutorial",
      "This will clear your onboarding status. You will be redirected to the initial vision screening and app walkthrough.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset & Restart",
          style: "default",
          onPress: async () => {
            try {
              // Clear the local flag that bypasses onboarding
              await AsyncStorage.removeItem("hasSeenOnboarding");
              // Force navigation back to the start of the clinical flow
              navigation.replace("AppOnboarding");
            } catch (error) {
              console.error("Failed to reset onboarding:", error);
            }
          },
        },
      ],
    );
  };
  const handleDeleteAccount = () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    Alert.alert(
      "Delete Account",
      "This action is irreversible. All clinical history and calibration data will be permanently erased.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete Everything",
          style: "destructive",
          onPress: async () => {
            try {
              await deleteDoc(doc(db, "users", user.uid));
              await auth.currentUser.delete();
              navigation.replace("Login");
            } catch (error) {
              Alert.alert(
                "Error",
                "Please re-authenticate to perform this action.",
              );
            }
          },
        },
      ],
    );
  };

  return (
    <View style={styles.root}>
      <BackgroundBubbles />
      <View style={[styles.header, { paddingTop: insets.top + 10 }]}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        >
          <Ionicons name="chevron-back" size={24} color={COLORS.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>System Settings</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView
        contentContainerStyle={[
          styles.scroll,
          { paddingBottom: insets.bottom + 20 },
        ]}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View entering={FadeInDown.duration(400)}>
          {/* VISION ENGINE SETTINGS */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>VISION ENGINE</Text>
            <View style={styles.card}>
              <View style={styles.rowBetween}>
                <View style={styles.rowLabelGroup}>
                  <Ionicons
                    name="color-filter-outline"
                    size={20}
                    color={COLORS.primary}
                  />
                  <Text style={styles.label}>Enhancement Intensity</Text>
                </View>
                <Text style={styles.valueText}>
                  {Math.round(intensity * 100)}%
                </Text>
              </View>
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={1}
                value={intensity}
                onSlidingComplete={saveIntensity}
                minimumTrackTintColor={COLORS.primary}
                maximumTrackTintColor="#CBD5E1"
                thumbTintColor={Platform.OS === "ios" ? "#FFF" : COLORS.primary}
              />

              <View style={[styles.rowBetween, { marginTop: 24 }]}>
                <View style={styles.rowLabelGroup}>
                  <Ionicons
                    name="volume-medium-outline"
                    size={20}
                    color={COLORS.primary}
                  />
                  <Text style={styles.label}>Audio Feedback</Text>
                </View>
                <Switch
                  value={audioEnabled}
                  onValueChange={(val) => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    setAudioEnabled(val);
                    AsyncStorage.setItem(
                      "audio_feedback_enabled",
                      val.toString(),
                    );
                  }}
                  trackColor={{ false: "#CBD5E1", true: COLORS.primary }}
                />
              </View>

              {/* RESTORED: CAMERA CALIBRATION TRIGGER */}
              <TouchableOpacity
                style={styles.calibrationRow}
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                  navigation.navigate("CameraCalibration");
                }}
              >
                <View style={styles.rowLabelGroup}>
                  <Ionicons
                    name="aperture-outline"
                    size={20}
                    color={COLORS.primary}
                  />
                  <View>
                    <Text style={styles.label}>Camera Calibration</Text>
                    <Text
                      style={[
                        styles.statusValue,
                        {
                          color: calibration ? COLORS.success : COLORS.warning,
                        },
                      ]}
                    >
                      {calibration
                        ? "Manual Balance Active"
                        : "Using Auto-Balance (Uncalibrated)"}
                    </Text>
                  </View>
                </View>
                <Ionicons
                  name="chevron-forward"
                  size={18}
                  color={COLORS.textLight}
                />
              </TouchableOpacity>
            </View>
          </View>

          {/* CLINICAL DATA */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>DIAGNOSTIC STATUS</Text>
            <View style={styles.card}>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Total Assessments</Text>
                <Text style={styles.infoValue}>{testCount}</Text>
              </View>

              {expertRole && (
                <TouchableOpacity
                  style={styles.expertBtn}
                  onPress={() =>
                    navigation.navigate("AdminHub", { role: expertRole })
                  }
                >
                  <Ionicons name="ribbon-outline" size={18} color="#FFF" />
                  <Text style={styles.expertBtnText}>
                    View {expertRole.toUpperCase()} Dashboard
                  </Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
          {/* SUPPORT & UTILITIES */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>SUPPORT & UTILITIES</Text>
            <View style={styles.card}>
              <TouchableOpacity
                style={[styles.infoRow, { borderBottomWidth: 0 }]}
                onPress={handleResetOnboarding}
              >
                <View style={styles.rowLabelGroup}>
                  <Ionicons
                    name="refresh-circle-outline"
                    size={20}
                    color={COLORS.primary}
                  />
                  <Text style={styles.label}>Reset Onboarding</Text>
                </View>
                <Ionicons
                  name="chevron-forward"
                  size={18}
                  color={COLORS.textLight}
                />
              </TouchableOpacity>
            </View>
          </View>
          {/* ACCOUNT DESTRUCTION */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>ACCOUNT MANAGEMENT</Text>
            <View
              style={[styles.card, { borderColor: "#FEE2E2", borderWidth: 1 }]}
            >
              <Text style={styles.dangerTitle}>Danger Zone</Text>
              <Text style={styles.subtext}>
                Deleting your account will purge all Ishihara results and custom
                color calibration parameters.
              </Text>
              <TouchableOpacity
                style={styles.deleteBtn}
                onPress={handleDeleteAccount}
              >
                <Ionicons name="trash-outline" size={18} color="#EF4444" />
                <Text style={styles.deleteBtnText}>
                  Delete Account Permanently
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#F8FAFC" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: SPACING.md,
    paddingBottom: 15,
    backgroundColor: "#FFF",
    ...SHADOW.sm,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F1F5F9",
  },
  headerTitle: { fontSize: 18, fontWeight: "800", color: COLORS.text },
  scroll: { padding: SPACING.lg },
  section: { marginBottom: 30 },
  sectionTitle: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: 12,
    marginLeft: 4,
  },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.xl,
    padding: SPACING.lg,
    ...SHADOW.sm,
  },
  rowBetween: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  rowLabelGroup: { flexDirection: "row", alignItems: "center", gap: 12 },
  label: { fontSize: 16, fontWeight: "700", color: COLORS.text },
  valueText: { fontSize: 14, fontWeight: "800", color: COLORS.primary },
  slider: { width: "100%", height: 40, marginTop: 12 },
  statusValue: { fontSize: 11, fontWeight: "700", marginTop: 2 },
  calibrationRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 24,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: "#F1F5F9",
  },
  subtext: {
    fontSize: 12,
    color: COLORS.textLight,
    marginTop: 8,
    lineHeight: 18,
  },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#F1F5F9",
  },
  infoLabel: { fontSize: 14, color: COLORS.text, fontWeight: "600" },
  infoValue: { fontSize: 14, color: COLORS.textLight, fontWeight: "800" },
  expertBtn: {
    backgroundColor: COLORS.primary,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    padding: 14,
    borderRadius: RADIUS.lg,
    marginTop: 16,
    gap: 8,
  },
  expertBtnText: { color: "#FFF", fontWeight: "800", fontSize: 14 },
  dangerTitle: {
    fontSize: 14,
    fontWeight: "900",
    color: "#EF4444",
    marginBottom: 4,
  },
  deleteBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    padding: 14,
    borderRadius: RADIUS.lg,
    marginTop: 16,
    gap: 8,
    backgroundColor: "#FFF",
    borderWidth: 1,
    borderColor: "#FCA5A5",
  },
  deleteBtnText: { color: "#EF4444", fontWeight: "800", fontSize: 14 },
  footerVersion: {
    textAlign: "center",
    fontSize: 10,
    color: COLORS.textLight,
    marginTop: 10,
    opacity: 0.5,
  },
});
