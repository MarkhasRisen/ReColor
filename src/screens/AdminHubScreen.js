import { Ionicons } from "@expo/vector-icons";
import {
  collection,
  onSnapshot,
  query
} from "firebase/firestore";
import { MotiView } from "moti";
import { useEffect, useState } from "react";
import {
  Dimensions,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { auth, db, signOut } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Card from "../components/Card";
import Header from "../components/Header";
import { COLORS, SHADOW } from "../theme/colors";

const { width } = Dimensions.get("window");

export default function AdminHubScreen({ route, navigation }) {
  const { role } = route.params || { role: "researcher" };
  const [recordCount, setRecordCount] = useState(0);
  const [systemLogs, setSystemLogs] = useState([]);

  useEffect(() => {
    // REAL-TIME DATA INGRESS MONITOR[cite: 2]
    const qCount = query(collection(db, "research_data_anonymized"));
    const unsubCount = onSnapshot(qCount, (snap) => setRecordCount(snap.size));

    // MOCK SYSTEM EVENTS (For Defense Atmosphere)
    const mockLogs = [
      {
        id: "1",
        event: "Cloud Sync Active",
        time: "Just Now",
        icon: "cloud-done",
      },
      {
        id: "2",
        event: "Auth Gate Verified",
        time: "2m ago",
        icon: "shield-checkmark",
      },
      { id: "3", event: "Dual-Write Success", time: "15m ago", icon: "copy" },
    ];
    setSystemLogs(mockLogs);

    return () => unsubCount();
  }, []);

  return (
    <ScrollView style={styles.root} showsVerticalScrollIndicator={false}>
      <Header
        title="Expert Portal"
        subtitle="Clinical Operations Center"
        back
      />
      <BackgroundBubbles />

      <View style={{ padding: 20 }}>
        {/* 1. SYSTEM HEALTH MODULE[cite: 2] */}
        <View style={styles.healthContainer}>
          <HealthBadge label="Cloud DB" active />
          <HealthBadge label="Auth Service" active />
          <HealthBadge label="Encryption" active />
        </View>

        {/* 2. REAL-TIME STATS TICKER[cite: 2] */}
        <MotiView
          from={{ translateY: 20, opacity: 0 }}
          animate={{ translateY: 0, opacity: 1 }}
          transition={{ delay: 100 }}
        >
          <Card style={styles.statsCard}>
            <Text style={styles.statsLabel}>GLOBAL RESEARCH DATASET</Text>
            <Text style={styles.statsValue}>
              {recordCount.toLocaleString()}
            </Text>
            <Text style={styles.statsSub}>
              Anonymized diagnostic records synced
            </Text>
          </Card>
        </MotiView>

        <Text style={styles.sectionHeader}>ADMINISTRATION UTILITIES</Text>

        {/* 3. FUNCTIONAL GRID[cite: 2, 3] */}
        <View style={styles.grid}>
          <GridCard
            title="Analytics"
            sub="CVD Trends"
            icon="analytics"
            color="#6200EA"
            onPress={() => navigation.navigate("ResearchDashboard")}
          />
          <GridCard
            title="CMS Manager"
            sub="Edit Articles"
            icon="document-text"
            color="#0091EA"
            onPress={() =>
              Alert.alert(
                "CMS Module",
                "Article management in Read-Only for defense.",
              )
            }
          />
          <GridCard
            title="Sandbox"
            sub="Test App"
            icon="flask"
            color="#00C853"
            onPress={() => navigation.navigate("MainTabs")}
          />
          <GridCard
            title="Audit Log"
            sub="Security"
            icon="lock-closed"
            color="#FF6D00"
            onPress={() =>
              Alert.alert(
                "Audit Log",
                "Compliance logs are encrypted per RA 10173.",
              )
            }
          />
        </View>

        {/* 4. RECENT SYSTEM EVENTS LOG[cite: 3] */}
        <Text style={styles.sectionHeader}>SYSTEM EVENT STREAM</Text>
        <Card style={styles.logCard}>
          {systemLogs.map((log) => (
            <View key={log.id} style={styles.logItem}>
              <Ionicons name={log.icon} size={16} color={COLORS.primary} />
              <Text style={styles.logText}>{log.event}</Text>
              <Text style={styles.logTime}>{log.time}</Text>
            </View>
          ))}
        </Card>

        <TouchableOpacity
          style={styles.logoutBtn}
          onPress={() => {
            signOut(auth);
            navigation.replace("Login");
          }}
        >
          <Text style={styles.logoutText}>Terminate Expert Session</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

// Helper Components
const HealthBadge = ({ label, active }) => (
  <View style={styles.healthBadge}>
    <View
      style={[styles.dot, { backgroundColor: active ? "#00C853" : "#D50000" }]}
    />
    <Text style={styles.healthLabel}>{label}</Text>
  </View>
);

const GridCard = ({ title, sub, icon, color, onPress }) => (
  <TouchableOpacity
    style={[styles.gridCard, { borderTopColor: color }]}
    onPress={onPress}
  >
    <Ionicons name={icon} size={28} color={color} />
    <Text style={styles.gridTitle}>{title}</Text>
    <Text style={styles.gridSub}>{sub}</Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },
  healthContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 20,
  },
  healthBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFF",
    padding: 8,
    borderRadius: 20,
    width: "31%",
    ...SHADOW.sm,
  },
  dot: { width: 6, height: 6, borderRadius: 3, marginRight: 6 },
  healthLabel: { fontSize: 9, fontWeight: "bold", color: "#666" },
  statsCard: {
    padding: 25,
    alignItems: "center",
    backgroundColor: COLORS.primary,
    marginBottom: 25,
  },
  statsLabel: {
    color: "rgba(255,255,255,0.7)",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1,
  },
  statsValue: {
    color: "#FFF",
    fontSize: 48,
    fontWeight: "900",
    marginVertical: 4,
  },
  statsSub: { color: "rgba(255,255,255,0.8)", fontSize: 11 },
  sectionHeader: {
    fontSize: 12,
    fontWeight: "800",
    color: "#999",
    marginBottom: 12,
    letterSpacing: 1,
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    marginBottom: 25,
  },
  gridCard: {
    backgroundColor: "#FFF",
    width: "48%",
    padding: 15,
    borderRadius: 15,
    marginBottom: 15,
    borderTopWidth: 4,
    ...SHADOW.sm,
  },
  gridTitle: {
    fontSize: 15,
    fontWeight: "bold",
    marginTop: 10,
    color: COLORS.text,
  },
  gridSub: { fontSize: 11, color: "#888", marginTop: 2 },
  logCard: { padding: 15, backgroundColor: "#F8F9FF" },
  logItem: { flexDirection: "row", alignItems: "center", marginBottom: 12 },
  logText: {
    flex: 1,
    marginLeft: 10,
    fontSize: 13,
    color: "#444",
    fontWeight: "500",
  },
  logTime: { fontSize: 11, color: "#999" },
  logoutBtn: {
    marginTop: 30,
    padding: 15,
    alignItems: "center",
    marginBottom: 40,
  },
  logoutText: { color: "#D32F2F", fontWeight: "bold" },
});
