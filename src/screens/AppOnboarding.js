import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { MotiView } from "moti";
import { useEffect, useRef, useState } from "react";
import {
  Dimensions,
  FlatList,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { Camera } from "react-native-vision-camera";
import * as ImagePicker from "expo-image-picker";
import * as Haptics from "expo-haptics";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";
import { useSafeAreaInsets } from "react-native-safe-area-context";

const { width } = Dimensions.get("window");

const SLIDES = [
  {
    id: "1",
    icon: "eye-outline",
    iconColor: COLORS.primary,
    badge: "WELCOME",
    title: "Let's personalise\nyour view of the world.",
    subtitle:
      "ReColor uses image-processing algorithms and clinical screening to help you understand and enhance your color perception.",
    cta: "Get Started",
  },
  {
    id: "2",
    icon: "color-palette-outline",
    iconColor: "#FF6B6B",
    badge: "COLOR VISION",
    title: "What is your\ncolor vision type?",
    subtitle:
      "Millions of people experience color differently. Our Ishihara test identifies Deuteranomaly, Protanomaly, and Tritanomaly.",
    cta: "Next",
    options: [
      "Red Sensitivity (Protanomaly)",
      "Green Sensitivity (Deuteranomaly)",
      "Blue Sensitivity (Tritanomaly)",
      "Not sure yet",
    ],
  },
  {
    id: "3",
    icon: "phone-portrait-outline",
    iconColor: COLORS.accent,
    badge: "LIVE PREVIEW",
    title: "See the difference,\nright now.",
    subtitle:
      "ReColor identifies colors and applies adaptive enhancement filters to assist your color perception.",
    cta: "Next",
  },
  {
    id: "4",
    icon: "camera-outline",
    iconColor: COLORS.warning,
    badge: "PERMISSIONS",
    title: "We'll need your camera\nto act as your eyes.",
    subtitle:
      "Camera access enables color identification and CVD simulation. Photo library access lets you enhance saved images.",
    cta: "Continue",
    perms: ["Camera Access", "Photo Storage"],
  },
  {
    id: "5",
    icon: "options-outline",
    iconColor: COLORS.success,
    badge: "READY",
    title: "Choose your\nexperience.",
    subtitle:
      "Start with the clinical screening test or jump straight into color enhancement mode.",
    cta: "Begin →",
    finalCta: true,
  },
];

// Helper to match colors to specific sensitivities based on manuscript taxonomy
const getOptionColor = (opt) => {
  if (opt.includes("Red")) return "#FF5252"; // Protan Red
  if (opt.includes("Green")) return "#4CAF50"; // Deutan Green
  if (opt.includes("Blue")) return "#2196F3"; // Tritan Blue
  return COLORS.primary; // Default Purple
};

const getColorBandColor = (band, cvd) => {
  if (cvd === "Normal") {
    if (band === "Red") return "#EF4444";
    if (band === "Green") return "#10B981";
    if (band === "Blue") return "#3B82F6";
    return "#FBBF24"; // Yellow
  }
  if (cvd === "Protan") {
    if (band === "Red") return "#8A7320"; // Olive-brown
    if (band === "Green") return "#C0A830"; // Yellowish
    if (band === "Blue") return "#3B82F6";
    return "#CA8A04"; // Soft gold/dark ochre
  }
  if (cvd === "Deutan") {
    if (band === "Red") return "#A56B24"; // Orange-brown
    if (band === "Green") return "#8C8C24"; // Olive-yellow
    if (band === "Blue") return "#3B82F6";
    return "#E2E8F0"; // Yellow-gray/light grey
  }
  if (cvd === "Tritan") {
    if (band === "Red") return "#EF4444"; // Preserved
    if (band === "Green") return "#10B981"; // Preserved
    if (band === "Blue") return "#06B6D4"; // Blue looks greenish-cyan
    return "#F472B6"; // Yellow looks pink under Tritan!
  }
  return "#777";
};

function SlideItem({ item, currentIndex, onNext, onPrev, isLast, onFinish }) {
  const [selected, setSelected] = useState(null);
  const [mockCvd, setMockCvd] = useState("Normal");
  const [camStatus, setCamStatus] = useState("undetermined");
  const [galleryStatus, setGalleryStatus] = useState("undetermined");

  useEffect(() => {
    if (item.perms) {
      const status = Camera.getCameraPermissionStatus();
      setCamStatus(status);
      ImagePicker.getMediaLibraryPermissionsAsync().then((res) => setGalleryStatus(res.status));
    }
  }, [item.perms]);

  const requestPerm = async (type) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (type === "camera") {
      const status = await Camera.requestCameraPermission();
      setCamStatus(status);
    } else {
      const res = await ImagePicker.requestMediaLibraryPermissionsAsync();
      setGalleryStatus(res.status);
    }
  };

  return (
    <View style={[styles.slide, { width }]}>
      {/* Smooth entry for Icon - No bounce */}
      <MotiView
        from={{ opacity: 0, translateY: 30 }}
        animate={{ opacity: 1, translateY: 0 }}
        transition={{ type: "timing", duration: 600, delay: 150 }}
        style={styles.iconWrap}
      >
        <Ionicons name={item.icon} size={48} color={item.iconColor} />
      </MotiView>

      {/* Smooth entry for Text */}
      <MotiView
        from={{ opacity: 0, translateY: 20 }}
        animate={{ opacity: 1, translateY: 0 }}
        transition={{ type: "timing", duration: 700, delay: 200 }}
      >
        <Text style={styles.badge}>{item.badge}</Text>
        <Text style={styles.title}>{item.title}</Text>
        <Text style={styles.subtitle}>{item.subtitle}</Text>
      </MotiView>

      {/* Dynamic Highlighted Options */}
      {item.options && (
        <MotiView
          from={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ type: "timing", duration: 500, delay: 350 }}
          style={styles.optionsWrap}
        >
          {item.options.map((opt) => {
            const isActive = selected === opt;
            const activeColor = getOptionColor(opt);

            return (
              <TouchableOpacity
                key={opt}
                style={[
                  styles.optionChip,
                  isActive && {
                    borderColor: activeColor,
                    backgroundColor: activeColor + "15", // 10% opacity
                  },
                ]}
                onPress={() => setSelected(opt)}
              >
                <Text
                  style={[
                    styles.optionText,
                    isActive && { color: activeColor, fontWeight: "700" },
                  ]}
                >
                  {opt}
                </Text>
              </TouchableOpacity>
            );
          })}
        </MotiView>
      )}

      {/* Slide 3: Interactive Color Band Simulator mockup */}
      {item.id === "3" && (
        <MotiView
          from={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "timing", duration: 500, delay: 300 }}
          style={styles.simulatorCard}
        >
          <Text style={styles.simCardTitle}>INTERACTIVE COLOR WHEEL BAND</Text>
          <View style={styles.colorBandRow}>
            <View style={styles.bandCell}>
              <View style={[styles.colorBand, { backgroundColor: getColorBandColor("Red", mockCvd) }]} />
              <Text style={styles.bandLabel}>Red</Text>
            </View>
            <View style={styles.bandCell}>
              <View style={[styles.colorBand, { backgroundColor: getColorBandColor("Green", mockCvd) }]} />
              <Text style={styles.bandLabel}>Green</Text>
            </View>
            <View style={styles.bandCell}>
              <View style={[styles.colorBand, { backgroundColor: getColorBandColor("Blue", mockCvd) }]} />
              <Text style={styles.bandLabel}>Blue</Text>
            </View>
            <View style={styles.bandCell}>
              <View style={[styles.colorBand, { backgroundColor: getColorBandColor("Yellow", mockCvd) }]} />
              <Text style={styles.bandLabel}>Yellow</Text>
            </View>
          </View>

          <View style={styles.tabsRow}>
            {["Normal", "Protan", "Deutan", "Tritan"].map((mode) => (
              <TouchableOpacity
                key={mode}
                style={[styles.tabBtn, mockCvd === mode && styles.tabBtnActive]}
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                  setMockCvd(mode);
                }}
              >
                <Text style={[styles.tabText, mockCvd === mode && styles.tabTextActive]}>
                  {mode}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </MotiView>
      )}

      {item.perms && (
        <MotiView
          from={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ type: "timing", duration: 500, delay: 350 }}
          style={styles.permsWrap}
        >
          {/* Camera row */}
          <TouchableOpacity
            activeOpacity={0.7}
            style={[
              styles.permRow,
              camStatus === "granted" && { borderColor: COLORS.success, borderWidth: 1.5 }
            ]}
            onPress={() => requestPerm("camera")}
          >
            <Ionicons
              name={camStatus === "granted" ? "checkmark-circle" : "ellipse-outline"}
              size={20}
              color={camStatus === "granted" ? COLORS.success : COLORS.textLight}
            />
            <View style={{ flex: 1, marginLeft: 8 }}>
              <Text style={styles.permText}>Camera Access</Text>
              <Text style={styles.permSub}>
                {camStatus === "granted" ? "Permission active" : "Tap to authorize camera access"}
              </Text>
            </View>
            {camStatus !== "granted" && (
              <Ionicons name="chevron-forward" size={16} color={COLORS.textLight} />
            )}
          </TouchableOpacity>

          {/* Photo Library row */}
          <TouchableOpacity
            activeOpacity={0.7}
            style={[
              styles.permRow,
              galleryStatus === "granted" && { borderColor: COLORS.success, borderWidth: 1.5 }
            ]}
            onPress={() => requestPerm("gallery")}
          >
            <Ionicons
              name={galleryStatus === "granted" ? "checkmark-circle" : "ellipse-outline"}
              size={20}
              color={galleryStatus === "granted" ? COLORS.success : COLORS.textLight}
            />
            <View style={{ flex: 1, marginLeft: 8 }}>
              <Text style={styles.permText}>Photo Storage</Text>
              <Text style={styles.permSub}>
                {galleryStatus === "granted" ? "Permission active" : "Tap to authorize library access"}
              </Text>
            </View>
            {galleryStatus !== "granted" && (
              <Ionicons name="chevron-forward" size={16} color={COLORS.textLight} />
            )}
          </TouchableOpacity>
        </MotiView>
      )}

      {/* Primary CTA */}
      <MotiView
        from={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: "timing", duration: 500, delay: 450 }}
        style={styles.ctaWrap}
      >
        <View style={{ flexDirection: "row", gap: 12, alignItems: "center" }}>
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
            onPress={isLast ? onFinish : onNext}
            activeOpacity={0.85}
          >
            <Text style={styles.ctaText}>{item.cta}</Text>
          </TouchableOpacity>
        </View>
      </MotiView>
    </View>
  );
}

export default function AppOnboarding({ navigation }) {
  const flatRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const insets = useSafeAreaInsets();

  const goNext = () => {
    if (currentIndex < SLIDES.length - 1) {
      flatRef.current?.scrollToIndex({
        index: currentIndex + 1,
        animated: true,
      });
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goPrev = () => {
    if (currentIndex > 0) {
      flatRef.current?.scrollToIndex({
        index: currentIndex - 1,
        animated: true,
      });
      setCurrentIndex(currentIndex - 1);
    }
  };

  const finish = async () => {
    await AsyncStorage.setItem("@recolor_onboarded", "1").catch(() => {});
    navigation.replace("Login");
  };

  return (
    <View style={styles.container}>
      <FlatList
        ref={flatRef}
        data={SLIDES}
        keyExtractor={(s) => s.id}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        scrollEnabled={false}
        renderItem={({ item, index }) => (
          <SlideItem
            item={item}
            currentIndex={index}
            onNext={goNext}
            onPrev={goPrev}
            isLast={index === SLIDES.length - 1}
            onFinish={finish}
          />
        )}
      />

      {/* Progress indicators - Smoothed */}
      <View style={styles.dotsRow}>
        {SLIDES.map((_, i) => (
          <MotiView
            key={i}
            animate={{
              width: i === currentIndex ? 24 : 8,
              opacity: i === currentIndex ? 1 : 0.35,
            }}
            transition={{ type: "timing", duration: 300 }}
            style={[styles.dot, { backgroundColor: COLORS.primary }]}
          />
        ))}
      </View>

      <TouchableOpacity style={[styles.skipBtn, { top: insets.top + 10 }]} onPress={finish}>
        <Text style={styles.skipText}>Skip</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  slide: {
    flex: 1,
    paddingHorizontal: SPACING.md,
    paddingTop: 80,
    paddingBottom: 120,
    alignItems: "center",
  },
  iconWrap: {
    width: 96,
    height: 96,
    borderRadius: RADIUS.xl,
    backgroundColor: COLORS.card,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: SPACING.lg,
    ...SHADOW.md,
  },
  badge: {
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 1.5,
    color: COLORS.primary,
    textAlign: "center",
    marginBottom: SPACING.sm,
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: COLORS.text,
    textAlign: "center",
    lineHeight: 36,
    marginBottom: SPACING.md,
  },
  subtitle: {
    fontSize: 15,
    color: COLORS.textLight,
    textAlign: "center",
    lineHeight: 22,
    paddingHorizontal: SPACING.sm,
  },
  optionsWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: SPACING.sm,
    marginTop: SPACING.lg,
  },
  optionChip: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: 20,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    backgroundColor: COLORS.card,
  },
  optionText: {
    fontSize: 14,
    color: COLORS.textLight,
    fontWeight: "500",
  },
  permsWrap: {
    marginTop: SPACING.lg,
    alignSelf: "stretch",
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
  },
  permRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    backgroundColor: COLORS.card,
    padding: SPACING.md,
    borderRadius: RADIUS.md,
    ...SHADOW.sm,
    borderWidth: 1.5,
    borderColor: "transparent",
  },
  permText: { fontSize: 14, color: COLORS.text, fontWeight: "500" },
  permSub: {
    fontSize: 11,
    color: COLORS.textLight,
    marginTop: 2,
  },
  ctaWrap: {
    position: "absolute",
    bottom: SPACING.xxl,
    left: SPACING.md,
    right: SPACING.md,
  },
  ctaBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.lg,
    paddingVertical: 16,
    alignItems: "center",
    ...SHADOW.md,
  },
  ctaText: {
    color: "#FFF",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  dotsRow: {
    position: "absolute",
    bottom: 110,
    alignSelf: "center",
    flexDirection: "row",
    gap: 6,
    alignItems: "center",
  },
  dot: { height: 8, borderRadius: 4 },
  skipBtn: {
    position: "absolute",
    top: 52,
    right: SPACING.md,
    padding: SPACING.sm,
  },
  skipText: { color: COLORS.textLight, fontSize: 14, fontWeight: "500" },

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

  simulatorCard: {
    backgroundColor: "#FFF",
    padding: 16,
    borderRadius: RADIUS.xl,
    marginTop: SPACING.md,
    width: "100%",
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  simCardTitle: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    textAlign: "center",
    marginBottom: 12,
    letterSpacing: 1.5,
  },
  colorBandRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginBottom: 15,
  },
  bandCell: {
    alignItems: "center",
    gap: 4,
  },
  colorBand: {
    width: 58,
    height: 18,
    borderRadius: 9,
    ...SHADOW.sm,
  },
  bandLabel: {
    fontSize: 11,
    color: COLORS.text,
    fontWeight: "700",
  },
  tabsRow: {
    flexDirection: "row",
    justifyContent: "center",
    backgroundColor: "#F1F5F9",
    borderRadius: RADIUS.lg,
    padding: 3,
    gap: 2,
  },
  tabBtn: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    alignItems: "center",
  },
  tabBtnActive: {
    backgroundColor: "#FFF",
    ...SHADOW.xs,
  },
  tabText: {
    fontSize: 12,
    color: COLORS.textLight,
    fontWeight: "700",
  },
  tabTextActive: {
    color: COLORS.primary,
  },
});
