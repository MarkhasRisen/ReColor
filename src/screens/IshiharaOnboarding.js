import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { MotiView } from "moti";
import { useEffect, useRef, useState } from "react";
import {
  Alert,
  Dimensions,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import * as Brightness from "expo-brightness";
import * as Haptics from "expo-haptics";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

const { width } = Dimensions.get("window");

const INSTRUCTIONS = [
  {
    id: "1",
    step: 1,
    icon: "sunny-outline",
    iconBg: "#FFF3E0",
    iconColor: "#FF9F43",
    label: "SCREEN LUMINOSITY",
    infoTitle: "Screen Luminosity",
    infoText:
      "Low brightness reduces contrast, making it harder to distinguish subtle color differences in the Ishihara plates.",
    title: "Min. 80% Brightness",
    points: [
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Set screen brightness to at least 80%",
      },
      {
        icon: "ellipse",
        color: "#CCC",
        text: "App will automatically verify your brightness level",
      },
      {
        icon: "close-circle",
        color: COLORS.danger,
        text: "Dim screens distort colour perception",
      },
    ],
    tip: "Like pre-flight checks: Just as pilots verify all instruments before takeoff, ensure your screen brightness is optimal.",
    cta: "Next: Lighting Conditions",
  },
  {
    id: "2",
    step: 2,
    icon: "bulb-outline",
    iconBg: "#E8F5E9",
    iconColor: "#2ECC71",
    label: "LIGHTING CONDITIONS",
    infoTitle: "Lighting Conditions",
    infoText:
      "Colored ambient light shifts your perception of hues and can produce false results. Natural or pure white light is required.",
    title: "Natural or White Light",
    points: [
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Sit in a well-lit room with white light",
      },
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Avoid direct sunlight on screen (glare)",
      },
      {
        icon: "close-circle",
        color: COLORS.danger,
        text: "Avoid yellow incandescent lighting",
      },
    ],
    tip: "Coloured ambient light shifts your perception of hues and can produce false results.",
    cta: "Next: Viewing Distance",
  },
  {
    id: "3",
    step: 3,
    icon: "expand-outline",
    iconBg: "#E3F2FD",
    iconColor: "#2196F3",
    label: "VIEWING DISTANCE",
    infoTitle: "Viewing Distance",
    infoText:
      "The Ishihara test is calibrated for a specific viewing angle. Distance consistency ensures the color patches fall on the correct part of your retina.",
    title: "35–75 cm Distance",
    points: [
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Hold device 35–75 cm (arm's length) away",
      },
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Plates fill your central vision field",
      },
      {
        icon: "close-circle",
        color: COLORS.danger,
        text: "Too close or too far skews the test",
      },
    ],
    tip: "The Ishihara test is calibrated for a specific viewing angle. Distance consistency matters.",
    cta: "Next: Time Limit",
  },
  {
    id: "4",
    step: 4,
    icon: "timer-outline",
    iconBg: "#FCE4EC",
    iconColor: "#FF4081",
    label: "TIME LIMIT",
    infoTitle: "Time Limit",
    infoText:
      "Prolonged viewing allows non-color cues (like dot size or density) to bias the result. Your first impression is the most clinically valid.",
    title: "3-Second Rule",
    points: [
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Each plate is visible for 3 seconds",
      },
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Enter your answer using the numpad",
      },
      {
        icon: "close-circle",
        color: COLORS.danger,
        text: "Do not guess — trust your first impression",
      },
    ],
    tip: "Prolonged viewing allows non-colour cues (size, density) to bias the result. Your first impression is the most clinically valid.",
    cta: "Next: Corrective Lenses",
  },
  {
    id: "5",
    step: 5,
    icon: "glasses-outline",
    iconBg: "#EDE7F6",
    iconColor: "#6C63FF",
    label: "CORRECTIVE LENSES",
    infoTitle: "Corrective Lenses",
    infoText:
      "Tinted lenses alter color perception. Clear prescription lenses do not affect Ishihara results.",
    title: "Glasses & Contact Lenses",
    points: [
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Wear your usual prescription lenses",
      },
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Standard clear lenses are fine",
      },
      {
        icon: "close-circle",
        color: COLORS.danger,
        text: "Remove tinted or coloured contact lenses",
      },
    ],
    tip: "Tinted lenses alter colour perception. Clear prescription lenses do not affect Ishihara results.",
    cta: "Next: Environment",
  },
  {
    id: "6",
    step: 6,
    icon: "home-outline",
    iconBg: "#E8F5E9",
    iconColor: "#2ECC71",
    label: "ENVIRONMENT",
    infoTitle: "Environment",
    infoText:
      "Visual fatigue affects color discrimination. Best results are obtained when well-rested in a distraction-free environment.",
    title: "Quiet, Distraction-Free",
    points: [
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Find a calm, quiet environment",
      },
      {
        icon: "checkmark-circle",
        color: COLORS.success,
        text: "Do not rush — take each plate seriously",
      },
      {
        icon: "close-circle",
        color: COLORS.danger,
        text: "Avoid testing when fatigued or stressed",
      },
    ],
    tip: "Visual fatigue affects colour discrimination. Best results are obtained when well-rested.",
    cta: "I'm Ready",
    isFinal: true,
  },
];

function InstructionSlide({ item, currentIndex, onNext, onPrev, isFinal, onBegin, brightness, boostBrightness, testType }) {
  return (
    <View style={[styles.slide, { width }]}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Header card */}
        <MotiView
          from={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "timing", duration: 300 }}
          style={styles.headerCard}
        >
          <View style={[styles.iconCircle, { backgroundColor: item.iconBg }]}>
            <Ionicons name={item.icon} size={28} color={item.iconColor} />
          </View>
          <View style={{ flex: 1 }}>
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <Text style={styles.label}>{item.label}</Text>
              {item.infoText && (
                <TouchableOpacity
                  onPress={() =>
                    Alert.alert(item.infoTitle || "Info", item.infoText)
                  }
                  style={{ marginLeft: 6, padding: 2 }}
                >
                  <Ionicons
                    name="information-circle-outline"
                    size={14}
                    color="rgba(255,255,255,0.8)"
                  />
                </TouchableOpacity>
              )}
            </View>
            <Text style={styles.cardTitle}>{item.title}</Text>
          </View>
        </MotiView>

        {/* Live Brightness Checking (Only shows on Step 1) */}
        {item.id === "1" && (
          <MotiView
            from={{ opacity: 0, translateY: 10 }}
            animate={{ opacity: 1, translateY: 0 }}
            transition={{ type: "timing", duration: 300, delay: 100 }}
            style={styles.brightnessWidget}
          >
            <View style={styles.widgetHeader}>
              <Ionicons
                name={brightness >= 0.8 ? "checkmark-circle" : "warning"}
                size={20}
                color={brightness >= 0.8 ? COLORS.success : COLORS.warning}
              />
              <Text style={styles.widgetTitle}>
                Current Brightness: {Math.round(brightness * 100)}%
              </Text>
            </View>
            <Text style={styles.widgetText}>
              {brightness >= 0.8
                ? "Luminosity is optimal. Your display contrast is high enough for accurate testing."
                : "Your screen is too dim. Dim screens can distort screening results."}
            </Text>
            {brightness < 0.8 && (
              <TouchableOpacity style={styles.boostBtn} onPress={boostBrightness}>
                <Ionicons name="flash" size={16} color="#FFF" />
                <Text style={styles.boostBtnText}>Auto-Boost to 80%</Text>
              </TouchableOpacity>
            )}
          </MotiView>
        )}

        {/* Visual Distance Diagram (Only shows on Step 3) */}
        {item.id === "3" && (
          <MotiView
            from={{ opacity: 0, translateY: 10 }}
            animate={{ opacity: 1, translateY: 0 }}
            transition={{ type: "timing", duration: 300, delay: 100 }}
            style={styles.distanceDiagram}
          >
            <View style={styles.diagramContent}>
              {/* Person Icon */}
              <Ionicons
                name="person"
                size={54}
                color={COLORS.primary}
                style={{ zIndex: 2 }}
              />

              {/* Extended Arm Graphic */}
              <View style={styles.armContainer}>
                <Text style={styles.armLabel}>{"Arm's Length"}</Text>
                <Text style={styles.armMeasurement}>35 - 75 cm</Text>
                <View style={styles.armBar} />
              </View>

              {/* Phone Icon */}
              <View style={styles.deviceContainer}>
                <Ionicons name="phone-portrait" size={44} color={COLORS.text} />
              </View>
            </View>
          </MotiView>
        )}

        {/* Checklist */}
        <MotiView
          from={{ opacity: 0, translateY: 16 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 300, delay: 150 }}
          style={styles.checklistCard}
        >
          <View style={styles.checklistHeader}>
            <View style={styles.dividerLine} />
            <Text style={styles.checklistLabel}>
              PREPARATION CHECKLIST · {INSTRUCTIONS.length} REQUIREMENTS
            </Text>
            <View style={styles.dividerLine} />
          </View>
          {item.points.map((pt, i) => (
            <View key={i} style={styles.pointRow}>
              <Ionicons name={pt.icon} size={20} color={pt.color} />
              <Text style={styles.pointText}>{pt.text}</Text>
            </View>
          ))}
        </MotiView>

        {/* Tip */}
        {item.tip && (
          <MotiView
            from={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 300 }}
            style={styles.tipCard}
          >
            <Text style={styles.tipText}>
              <Text style={{ fontWeight: "700", color: COLORS.primary }}>
                Tip:{" "}
              </Text>
              {item.tip}
            </Text>
          </MotiView>
        )}

        <View style={{ height: 100 }} />
      </ScrollView>

      {/* CTA */}
      <View style={styles.ctaContainer}>
        <View style={{ flexDirection: "row", gap: 12, alignItems: "center", width: "100%" }}>
          {currentIndex > 0 && (
            <TouchableOpacity
              style={styles.backCtaBtn}
              onPress={onPrev}
              activeOpacity={0.8}
            >
              <Ionicons name="arrow-back" size={18} color={COLORS.primary} />
              <Text style={styles.backCtaText}>Back</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity
            style={[styles.ctaBtn, { flex: 1 }]}
            onPress={isFinal ? onBegin : onNext}
            activeOpacity={0.85}
          >
            <Text style={styles.ctaText}>{item.cta}</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.ctaHint}>
          The test takes approximately {testType === "quick" ? "2–3 minutes" : "5–8 minutes"}
        </Text>
      </View>
    </View>
  );
}

export default function IshiharaOnboarding({ navigation, route }) {
  const { testType = "comprehensive" } = route?.params || {};
  const insets = useSafeAreaInsets();
  const scrollRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [hasSeen, setHasSeen] = useState(false);
  const [brightness, setBrightness] = useState(1.0);
  const [hasBrightnessPerm, setHasBrightnessPerm] = useState(false);

  const checkBrightness = async () => {
    try {
      const { status } = await Brightness.requestPermissionsAsync();
      if (status === "granted") {
        setHasBrightnessPerm(true);
        const val = await Brightness.getBrightnessAsync();
        setBrightness(val);
      }
    } catch (e) {
      console.log("Error checking brightness", e);
    }
  };

  const boostBrightness = async () => {
    try {
      if (hasBrightnessPerm) {
        await Brightness.setBrightnessAsync(0.8);
        setBrightness(0.8);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } else {
        const { status } = await Brightness.requestPermissionsAsync();
        if (status === "granted") {
          setHasBrightnessPerm(true);
          await Brightness.setBrightnessAsync(0.8);
          setBrightness(0.8);
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        }
      }
    } catch (e) {
      Alert.alert("Permission Needed", "Please enable screen brightness permissions in settings to auto-boost.");
    }
  };

  useEffect(() => {
    checkBrightness();
    const timer = setInterval(checkBrightness, 1500);
    return () => clearInterval(timer);
  }, [hasBrightnessPerm]);

  useEffect(() => {
    AsyncStorage.getItem("@seen_ishihara_onboard").then((val) => {
      if (val === "1") setHasSeen(true);
    });
  }, []);

  const goNext = () => {
    if (currentIndex < INSTRUCTIONS.length - 1) {
      const next = currentIndex + 1;
      scrollRef.current?.scrollTo({ x: next * width, animated: true });
      setCurrentIndex(next);
    }
  };

  const goPrev = () => {
    if (currentIndex > 0) {
      const prev = currentIndex - 1;
      scrollRef.current?.scrollTo({ x: prev * width, animated: true });
      setCurrentIndex(prev);
    }
  };

  const begin = async () => {
    try {
      const { status } = await Brightness.requestPermissionsAsync();
      if (status === "granted") {
        await Brightness.setBrightnessAsync(0.8);
      }
    } catch (e) {
      console.log("Error checking brightness before test", e);
    }
    await AsyncStorage.setItem("@seen_ishihara_onboard", "1");
    navigation.replace("IshiharaTest", { testType });
  };

  return (
    <View style={styles.container}>
      {/* Top bar */}
      <View style={[styles.topBar, { paddingTop: insets.top + 8 }]}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backBtn}
        >
          <Ionicons name="arrow-back" size={22} color={COLORS.text} />
        </TouchableOpacity>
        <Text style={styles.topTitle}>Before You Begin</Text>
        {hasSeen ? (
          <TouchableOpacity onPress={begin}>
            <Text style={{ color: COLORS.primary, fontWeight: "700" }}>
              Skip
            </Text>
          </TouchableOpacity>
        ) : (
          <Image
            source={require("../../assets/icon.png")}
            style={{ width: 36, height: 36, resizeMode: "contain" }}
          />
        )}
      </View>

      {/* Progress dots */}
      <View style={styles.dotsRow}>
        {INSTRUCTIONS.map((_, i) => (
          <MotiView
            key={i}
            animate={{
              width: i === currentIndex ? 20 : 6,
              opacity: i === currentIndex ? 1 : 0.3,
            }}
            transition={{ type: "timing", duration: 250 }}
            style={[styles.dot, { backgroundColor: COLORS.primary }]}
          />
        ))}
      </View>

      {/* Sticky disclaimer */}
      <View style={styles.disclaimerBanner}>
        <Ionicons name="warning-outline" size={13} color="#F59E0B" />
        <Text style={styles.disclaimerText}>
          NOT A MEDICAL DIAGNOSIS — SCREENING PURPOSE ONLY
        </Text>
      </View>

      <ScrollView
        ref={scrollRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        scrollEnabled={false}
      >
        {INSTRUCTIONS.map((item, index) => (
          <InstructionSlide
            key={item.id}
            item={item}
            currentIndex={index}
            onNext={goNext}
            onPrev={goPrev}
            isFinal={index === INSTRUCTIONS.length - 1}
            onBegin={begin}
            brightness={brightness}
            boostBrightness={boostBrightness}
            testType={testType}
          />
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: SPACING.md,
    paddingBottom: SPACING.sm,
    backgroundColor: COLORS.card,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  backBtn: { padding: 4 },
  topTitle: { fontSize: 16, fontWeight: "700", color: COLORS.text },
  eyeCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1.5,
    borderColor: COLORS.primary,
    alignItems: "center",
    justifyContent: "center",
  },
  dotsRow: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 6,
    paddingVertical: SPACING.sm,
    backgroundColor: COLORS.card,
  },
  dot: { height: 6, borderRadius: 3 },
  disclaimerBanner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: "#FFF8E1",
    paddingVertical: 6,
  },
  disclaimerText: {
    fontSize: 10,
    fontWeight: "700",
    color: "#F59E0B",
    letterSpacing: 0.5,
  },
  slide: { flex: 1 },
  scrollContent: { padding: SPACING.md, gap: SPACING.md },
  headerCard: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.md,
    ...SHADOW.md,
  },
  iconCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: "center",
    justifyContent: "center",
  },
  label: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 1.5,
    color: "rgba(255,255,255,0.7)",
  },
  cardTitle: { fontSize: 18, fontWeight: "800", color: "#FFF", marginTop: 2 },

  // --- New Native Arm's Length Diagram Styles ---
  distanceDiagram: {
    backgroundColor: COLORS.surfaceAlt,
    padding: SPACING.lg,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: COLORS.primary + "20",
    ...SHADOW.sm,
  },
  diagramContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  armContainer: {
    flex: 1,
    alignItems: "center",
    marginHorizontal: -12, // Pulls the arm slightly under the person and phone
    zIndex: 1,
  },
  armLabel: {
    fontSize: 12,
    fontWeight: "800",
    color: COLORS.primary,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  armMeasurement: {
    fontSize: 11,
    color: COLORS.textLight,
    marginBottom: 8,
    fontWeight: "600",
  },
  armBar: {
    width: "100%",
    height: 14,
    backgroundColor: COLORS.primary + "40", // Soft primary color for the arm
    borderRadius: 7,
  },
  deviceContainer: {
    zIndex: 2,
    alignItems: "center",
    justifyContent: "center",
  },
  // ----------------------------------------------

  checklistCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    gap: SPACING.md,
    ...SHADOW.sm,
  },
  checklistHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
  },
  dividerLine: { flex: 1, height: 1, backgroundColor: COLORS.border },
  checklistLabel: {
    fontSize: 9,
    fontWeight: "700",
    letterSpacing: 1.2,
    color: COLORS.textLight,
  },
  pointRow: { flexDirection: "row", alignItems: "flex-start", gap: SPACING.sm },
  pointText: { flex: 1, fontSize: 14, color: COLORS.text, lineHeight: 20 },
  tipCard: {
    backgroundColor: COLORS.surfaceAlt,
    borderRadius: RADIUS.md,
    padding: SPACING.md,
    borderLeftWidth: 3,
    borderLeftColor: COLORS.primary,
  },
  tipText: { fontSize: 13, color: COLORS.text, lineHeight: 20 },
  ctaContainer: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.card,
    padding: SPACING.md,
    paddingBottom: SPACING.xl,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    gap: SPACING.sm,
  },
  ctaBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.lg,
    paddingVertical: 16,
    alignItems: "center",
    ...SHADOW.md,
  },
  ctaText: { color: "#FFF", fontSize: 16, fontWeight: "700" },
  ctaHint: { fontSize: 12, color: COLORS.textLight, textAlign: "center" },

  brightnessWidget: {
    backgroundColor: "#FFF",
    padding: 16,
    borderRadius: RADIUS.xl,
    marginVertical: 12,
    borderWidth: 1,
    borderColor: "#F1F5F9",
    ...SHADOW.sm,
  },
  widgetHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  widgetTitle: {
    fontSize: 15,
    fontWeight: "800",
    color: COLORS.text,
  },
  widgetText: {
    fontSize: 13,
    color: COLORS.textLight,
    lineHeight: 18,
  },
  boostBtn: {
    backgroundColor: COLORS.primary,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
    borderRadius: RADIUS.lg,
    marginTop: 12,
    gap: 6,
    ...SHADOW.sm,
  },
  boostBtnText: {
    color: "#FFF",
    fontWeight: "800",
    fontSize: 13,
  },

  backCtaBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: COLORS.card,
    borderWidth: 1.5,
    borderColor: COLORS.primary + "30",
    borderRadius: RADIUS.lg,
    paddingVertical: 16,
    paddingHorizontal: 20,
    gap: 6,
    ...SHADOW.sm,
  },
  backCtaText: {
    color: COLORS.primary,
    fontSize: 15,
    fontWeight: "700",
  },
});
