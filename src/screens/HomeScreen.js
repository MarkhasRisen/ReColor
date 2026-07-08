import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  collection,
  limit,
  onSnapshot,
  orderBy,
  query,
} from "firebase/firestore";
import { useEffect, useState } from "react";
import {
  Dimensions,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, {
  FadeInDown,
  FadeInRight,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withSpring,
  withTiming,
} from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { auth, db } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

const { width } = Dimensions.get("window");

const ActionCard = ({
  title,
  desc,
  icon,
  color,
  onPress,
  isDominant = false,
  badge = null,
}) => {
  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: opacity.value,
  }));

  const handlePressIn = () => {
    scale.value = withSpring(0.97);
    opacity.value = withTiming(0.9);
  };

  const handlePressOut = () => {
    scale.value = withSpring(1);
    opacity.value = withTiming(1);
  };

  return (
    <Animated.View
      style={[animatedStyle, { width: isDominant ? "100%" : "48%" }]}
    >
      <TouchableOpacity
        activeOpacity={1}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        onPress={onPress}
        style={[
          styles.baseCard,
          isDominant ? styles.dominantCard : styles.secondaryCard,
          { borderColor: color + "30" },
        ]}
      >
        <View
          style={isDominant ? styles.dominantContent : styles.secondaryContent}
        >
          <View style={[styles.iconBox, { backgroundColor: color + "15" }]}>
            <Ionicons name={icon} size={isDominant ? 32 : 24} color={color} />
          </View>
          <View
            style={isDominant ? styles.textGroup : styles.secondaryTextGroup}
          >
            {badge && (
              <Text style={[styles.cardBadge, { color }]}>{badge}</Text>
            )}
            <Text style={styles.cardTitle}>{title}</Text>
            {isDominant && <Text style={styles.cardDesc}>{desc}</Text>}
          </View>
          {isDominant && (
            <Ionicons
              name="chevron-forward"
              size={20}
              color={COLORS.textLight}
            />
          )}
        </View>
      </TouchableOpacity>
    </Animated.View>
  );
};

const getStatusColors = (profile) => {
  const p = profile ? profile.toLowerCase() : "";
  if (p.includes("normal")) {
    return { text: "#166534", bg: "#F0FDF4", dot: "#22C55E" }; // Green (emerald/success)
  }
  if (p.includes("prot")) {
    return { text: "#991B1B", bg: "#FEF2F2", dot: "#EF4444" }; // Red (protan)
  }
  if (p.includes("deut")) {
    return { text: "#15803D", bg: "#F0FDF4", dot: "#22C55E" }; // Green (deutan)
  }
  if (p.includes("trit")) {
    return { text: "#0D47A1", bg: "#E3F2FD", dot: "#3B82F6" }; // Blue (tritan)
  }
  return { text: "#64748B", bg: "#F1F5F9", dot: "#94A3B8" }; // Slate/Grey (unscreened/unknown)
};

export default function HomeScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const [visionProfile, setVisionProfile] = useState("Unscreened");
  const userName = (auth.currentUser?.email || "Guest").split("@")[0];

  const logoScale = useSharedValue(1);
  const glowOpacity = useSharedValue(0.2);

  useEffect(() => {
    // Continuous Breathing Pulse
    logoScale.value = withRepeat(
      withSequence(
        withTiming(1.06, { duration: 2500 }),
        withTiming(1, { duration: 2500 }),
      ),
      -1,
      true,
    );

    glowOpacity.value = withRepeat(
      withSequence(
        withTiming(0.5, { duration: 2500 }),
        withTiming(0.2, { duration: 2500 }),
      ),
      -1,
      true,
    );
  }, []);

  const animatedLogoStyle = useAnimatedStyle(() => ({
    transform: [{ scale: logoScale.value }],
  }));

  const animatedGlowStyle = useAnimatedStyle(() => ({
    transform: [{ scale: logoScale.value * 1.15 }],
    opacity: glowOpacity.value * 0.4,
  }));

  useEffect(() => {
    const user = auth.currentUser;
    if (!user) {
      AsyncStorage.getItem("@recolor_latest_diagnosis")
        .then((latestDiag) => {
          if (latestDiag) {
            setVisionProfile(latestDiag);
          } else {
            setVisionProfile("Unscreened");
          }
        })
        .catch((err) => {
          console.error("Failed to read latest diagnosis:", err);
          setVisionProfile("Unscreened");
        });
      return;
    }
    const q = query(
      collection(db, "users", user.uid, "history"),
      orderBy("date", "desc"),
      limit(1),
    );
    return onSnapshot(
      q,
      (snap) => {
        if (!snap.empty)
          setVisionProfile(snap.docs[0].data().diagnosis || "Normal Vision");
      },
      (err) => {
        console.error("HomeScreen latest diagnosis fetch error:", err);
      }
    );
  }, []);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <BackgroundBubbles />

      <View
        style={[styles.disclaimer, { paddingTop: insets.top + SPACING.xs }]}
      >
        <Ionicons name="shield-checkmark" size={12} color={COLORS.warning} />
        <Text style={styles.disclaimerText}>
          SCREENING PURPOSE ONLY • NOT A MEDICAL DIAGNOSIS
        </Text>
      </View>

      <ScrollView
        contentContainerStyle={[
          styles.scrollBody,
          { paddingBottom: insets.bottom + 80 },
        ]}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View
          entering={FadeInDown.duration(800)}
          style={styles.header}
        >
          <View style={styles.headerText}>
            <Text style={styles.greeting}>Good day, {userName}</Text>
            <Text style={styles.mainTitle}>Welcome to{"\n"}ReColor</Text>
            {(() => {
              const statusColors = getStatusColors(visionProfile);
              return (
                <View
                  style={[
                    styles.profileBadge,
                    {
                      backgroundColor: statusColors.bg,
                      borderColor: statusColors.text + "20",
                      borderWidth: 1,
                    },
                  ]}
                >
                  <View
                    style={[styles.pulseDot, { backgroundColor: statusColors.dot }]}
                  />
                  <Text style={[styles.profileText, { color: statusColors.text }]}>
                    Status: {visionProfile}
                  </Text>
                </View>
              );
            })()}
          </View>

          {(() => {
            const statusColors = getStatusColors(visionProfile);
            return (
              <View style={styles.logoContainer}>
                <Animated.View
                  style={[
                    styles.logoGlow,
                    { backgroundColor: statusColors.dot },
                    animatedGlowStyle,
                  ]}
                />
                <Animated.Image
                  entering={FadeInRight.delay(300).springify()}
                  source={require("../../assets/icon.png")}
                  style={[styles.heroLogo, animatedLogoStyle]}
                />
              </View>
            );
          })()}
        </Animated.View>

        <View style={styles.section}>
          <Text style={styles.sectionLabel}>CLINICAL TOOLS</Text>
          <ActionCard
            isDominant
            title="Vision Assessment"
            desc="Standardized Ishihara screening to identify specific color sensitivities."
            icon="eye-outline"
            color={COLORS.primary}
            badge="RECOMMENDED"
            onPress={() => navigation.navigate("IshiharaIntro")}
          />
          <ActionCard
            isDominant
            title="Cameras and Simulation"
            desc="Apply adaptive color filters to enhance perception and simulate vision types."
            icon="camera-outline"
            color="#9C27B0"
            onPress={() => navigation.navigate("CameraEnhance")}
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionLabel}>KNOWLEDGE BASE</Text>
          <View style={styles.gridRow}>
            <ActionCard
              title="Education"
              icon="book-outline"
              color="#E91E63"
              onPress={() => navigation.navigate("EducationList")}
            />
            <ActionCard
              title="Survey"
              icon="stats-chart-outline"
              color="#4CAF50"
              onPress={() => navigation.navigate("Survey")}
            />
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  scrollBody: { paddingHorizontal: SPACING.lg, paddingBottom: 100 },
  disclaimer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: "#FFF8E1",
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: "#FFE082",
    zIndex: 10,
  },
  disclaimerText: {
    fontSize: 9,
    fontWeight: "800",
    color: "#B45309",
    letterSpacing: 0.5,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: SPACING.lg,
    marginBottom: SPACING.xl,
  },
  headerText: { flex: 1 },
  logoContainer: {
    width: 110,
    height: 110,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  logoGlow: {
    position: "absolute",
    width: 100,
    height: 100,
    borderRadius: 50,
  },
  heroLogo: {
    width: 78,
    height: 78,
    resizeMode: "contain",
    zIndex: 2,
  },
  greeting: {
    fontSize: 14,
    fontWeight: "600",
    color: COLORS.textLight,
    marginBottom: 4,
  },
  mainTitle: {
    fontSize: 34,
    fontWeight: "900",
    color: COLORS.text,
    lineHeight: 40,
    letterSpacing: -0.5,
  },
  profileBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: COLORS.card,
    alignSelf: "flex-start",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: RADIUS.md,
    marginTop: SPACING.md,
    ...SHADOW.sm,
  },
  pulseDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.success,
    marginRight: 8,
  },
  profileText: { fontSize: 12, fontWeight: "700", color: COLORS.text },
  section: { marginBottom: SPACING.xl },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: SPACING.md,
    opacity: 0.6,
  },
  baseCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.xl,
    borderWidth: 1,
    ...SHADOW.md,
  },
  dominantCard: { padding: SPACING.lg, marginBottom: SPACING.md },
  secondaryCard: { padding: SPACING.md, height: 120, justifyContent: "center" },
  dominantContent: { flexDirection: "row", alignItems: "center" },
  secondaryContent: { alignItems: "center" },
  iconBox: {
    width: 56,
    height: 56,
    borderRadius: RADIUS.lg,
    alignItems: "center",
    justifyContent: "center",
  },
  textGroup: { flex: 1, marginLeft: SPACING.md, marginRight: SPACING.sm },
  secondaryTextGroup: { marginTop: SPACING.sm, alignItems: "center" },
  cardBadge: {
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 1,
    marginBottom: 2,
  },
  cardTitle: { fontSize: 18, fontWeight: "800", color: COLORS.text },
  cardDesc: {
    fontSize: 13,
    color: COLORS.textLight,
    lineHeight: 18,
    marginTop: 4,
  },
  gridRow: { flexDirection: "row", justifyContent: "space-between" },
});
