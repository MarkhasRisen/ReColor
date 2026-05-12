import { Ionicons } from "@expo/vector-icons";
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

export default function HomeScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const [visionProfile, setVisionProfile] = useState("Unscreened");
  const userName = (auth.currentUser?.email || "Guest").split("@")[0];

  // LOGO PULSE ANIMATION
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
    shadowOpacity: glowOpacity.value,
  }));

  useEffect(() => {
    const user = auth.currentUser;
    if (!user) return;
    const q = query(
      collection(db, "users", user.uid, "history"),
      orderBy("date", "desc"),
      limit(1),
    );
    return onSnapshot(q, (snap) => {
      if (!snap.empty)
        setVisionProfile(snap.docs[0].data().diagnosis || "Normal Vision");
    });
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
          { paddingBottom: insets.bottom + SPACING.xl },
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
            <View style={styles.profileBadge}>
              <View style={styles.pulseDot} />
              <Text style={styles.profileText}>Status: {visionProfile}</Text>
            </View>
          </View>

          <Animated.Image
            entering={FadeInRight.delay(300).springify()}
            source={require("../../assets/icon.png")}
            style={[styles.heroLogo, animatedLogoStyle]}
          />
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
            desc="Apply adaptive enhancement filters via live camera processing."
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
  scrollBody: { paddingHorizontal: SPACING.lg },
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
  heroLogo: {
    width: 135,
    height: 135,
    resizeMode: "contain",
    ...SHADOW.lg,
    shadowColor: COLORS.primary,
    shadowRadius: 15,
  },
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
