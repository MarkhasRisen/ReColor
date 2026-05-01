import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { MotiView } from "moti";
import { useEffect, useRef, useState } from "react";
import {
  Dimensions,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
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

function InstructionSlide({ item, onNext, isFinal, onBegin }) {
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
          transition={{ type: "spring", damping: 18 }}
          style={styles.headerCard}
        >
          <View style={[styles.iconCircle, { backgroundColor: item.iconBg }]}>
            <Ionicons name={item.icon} size={28} color={item.iconColor} />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.label}>{item.label}</Text>
            <Text style={styles.cardTitle}>{item.title}</Text>
          </View>
        </MotiView>

        {/* Checklist */}
        <MotiView
          from={{ opacity: 0, translateY: 16 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "spring", damping: 18, delay: 150 }}
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
        <TouchableOpacity
          style={styles.ctaBtn}
          onPress={isFinal ? onBegin : onNext}
          activeOpacity={0.85}
        >
          <Text style={styles.ctaText}>{item.cta}</Text>
        </TouchableOpacity>
        <Text style={styles.ctaHint}>
          The test takes approximately 3–5 minutes
        </Text>
      </View>
    </View>
  );
}

export default function IshiharaOnboarding({ navigation, route }) {
  const { testType = "comprehensive" } = route?.params || {};
  const scrollRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [hasSeen, setHasSeen] = useState(false);

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

  const begin = () => {
    AsyncStorage.setItem("@seen_ishihara_onboard", "1");
    navigation.replace("IshiharaTest", { testType });
  };

  return (
    <View style={styles.container}>
      {/* Top bar */}
      <View style={styles.topBar}>
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
            transition={{ type: "spring", damping: 18 }}
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
            onNext={goNext}
            isFinal={index === INSTRUCTIONS.length - 1}
            onBegin={begin}
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
    paddingTop: 52,
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
});
