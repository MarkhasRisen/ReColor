import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Haptics from "expo-haptics";
import { signOut } from "firebase/auth";
import {
  collection,
  limit,
  onSnapshot,
  orderBy,
  query,
} from "firebase/firestore";
import { useEffect, useState } from "react";
import {
  Alert,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { auth, db } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

const ActionRow = ({
  icon,
  title,
  value,
  onPress,
  color = COLORS.primary,
  isLast = false,
}) => (
  <TouchableOpacity
    onPress={onPress}
    style={[styles.actionRow, isLast && { borderBottomWidth: 0 }]}
    activeOpacity={0.6}
  >
    <View style={[styles.rowIcon, { backgroundColor: color + "10" }]}>
      <Ionicons name={icon} size={20} color={color} />
    </View>
    <View style={styles.rowContent}>
      <Text style={styles.rowTitle}>{title}</Text>
      {value && <Text style={styles.rowValue}>{value}</Text>}
    </View>
    <Ionicons name="chevron-forward" size={16} color={COLORS.textLight} />
  </TouchableOpacity>
);

export default function ProfileScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const [latestAssessment, setLatestAssessment] = useState(null);
  const user = auth.currentUser;

  useEffect(() => {
    if (!user) {
      AsyncStorage.getItem("@recolor_local_history")
        .then((localData) => {
          if (localData) {
            try {
              const parsed = JSON.parse(localData);
              if (Array.isArray(parsed) && parsed.length > 0) {
                setLatestAssessment(parsed[0]);
              } else {
                setLatestAssessment(null);
              }
            } catch (_e) {
              setLatestAssessment(null);
            }
          } else {
            setLatestAssessment(null);
          }
        })
        .catch((err) => {
          console.error("Failed to read local history:", err);
          setLatestAssessment(null);
        });
      return;
    }
    const q = query(
      collection(db, "users", user.uid, "history"),
      orderBy("date", "desc"),
      limit(1),
    );
    return onSnapshot(q, (snap) => {
      if (!snap.empty) setLatestAssessment(snap.docs[0].data());
    });
  }, [user]);

  const handleSignOut = () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    Alert.alert("Sign Out", "Are you sure you want to exit your session?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Sign Out",
        style: "destructive",
        onPress: () => signOut(auth).then(() => navigation.replace("Login")),
      },
    ]);
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <BackgroundBubbles />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* IDENTITY CARD */}
        <View style={styles.idSection}>
          <View style={styles.avatarWrapper}>
            <Image
              source={{
                uri: `https://ui-avatars.com/api/?name=${user?.email || "Guest"}&background=4F46E5&color=fff&size=256`,
              }}
              style={styles.mainAvatar}
            />
            <View style={styles.onlineIndicator} />
          </View>
          <Text style={styles.idName}>
            {(user?.email || "Guest").split("@")[0]}
          </Text>
          <Text style={styles.idEmail}>{user?.email || "Guest User"}</Text>
        </View>

        {/* CLINICAL STATUS MODULE */}
        <View style={styles.moduleCard}>
          <Text style={styles.moduleHeader}>DIAGNOSTIC SUMMARY</Text>
          <ActionRow
            icon="shield-checkmark-outline"
            title="Vision Profile"
            value={latestAssessment?.diagnosis || "Not Screened"}
            color="#10B981"
            onPress={() => navigation.navigate("History")}
          />
          <ActionRow
            icon="refresh-outline"
            title="Re-calibrate Vision"
            value="Start Ishihara Test"
            color="#6366F1"
            onPress={() => navigation.navigate("IshiharaOnboarding")}
            isLast
          />
        </View>

        {/* SYSTEM CONFIGURATION MODULE */}
        <View style={styles.moduleCard}>
          <Text style={styles.moduleHeader}>SYSTEM & PREFERENCES</Text>
          <ActionRow
            icon="settings-outline"
            title="App Settings"
            value="Accessibility & Haptics"
            color={COLORS.text}
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              navigation.navigate("Settings"); // THIS CONNECTS TO THE NEW SCREEN
            }}
          />
          <ActionRow
            icon="help-buoy-outline"
            title="Support Center"
            color="#F59E0B"
            onPress={() => Alert.alert("Support", "Documentation coming soon.")}
            isLast
          />
        </View>

        {/* LOGOUT ACTION */}
        <TouchableOpacity style={styles.logoutButton} onPress={handleSignOut}>
          <Ionicons name="log-out-outline" size={20} color="#EF4444" />
          <Text style={styles.logoutText}>Sign Out of ReColor</Text>
        </TouchableOpacity>

        <Text style={styles.versionText}>Version 1.0 Beta </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  scrollContent: { padding: SPACING.lg, paddingBottom: 120 },
  idSection: { alignItems: "center", marginBottom: SPACING.xl },
  avatarWrapper: { position: "relative" },
  mainAvatar: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: "#E2E8F0",
  },
  onlineIndicator: {
    position: "absolute",
    bottom: 5,
    right: 5,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: "#10B981",
    borderWidth: 3,
    borderColor: "#F8FAFC",
  },
  idName: {
    fontSize: 22,
    fontWeight: "900",
    color: COLORS.text,
    marginTop: 12,
    textTransform: "capitalize",
  },
  idEmail: { fontSize: 14, color: COLORS.textLight, marginTop: 2 },
  moduleCard: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    padding: SPACING.md,
    marginBottom: SPACING.lg,
    ...SHADOW.sm,
  },
  moduleHeader: {
    fontSize: 10,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: SPACING.md,
    marginLeft: 4,
  },
  actionRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: "#F1F5F9",
  },
  rowIcon: {
    width: 38,
    height: 38,
    borderRadius: RADIUS.md,
    alignItems: "center",
    justifyContent: "center",
    marginRight: SPACING.md,
  },
  rowContent: { flex: 1 },
  rowTitle: { fontSize: 15, fontWeight: "700", color: COLORS.text },
  rowValue: { fontSize: 12, color: COLORS.textLight, marginTop: 2 },
  logoutButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFF",
    padding: 16,
    borderRadius: RADIUS.xl,
    borderWidth: 1,
    borderColor: "#FEE2E2",
  },
  logoutText: {
    color: "#EF4444",
    fontWeight: "800",
    fontSize: 15,
    marginLeft: 8,
  },
  versionText: {
    textAlign: "center",
    color: COLORS.textLight,
    fontSize: 10,
    marginTop: SPACING.xl,
    opacity: 0.4,
  },
});
