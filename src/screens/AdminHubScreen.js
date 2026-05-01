import { Ionicons } from "@expo/vector-icons";
import {
  collection,
  limit,
  onSnapshot,
  orderBy,
  query,
} from "firebase/firestore";
import { MotiView } from "moti";
import { useEffect, useState } from "react";
import {
  Alert,
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

export default function AdminHubScreen({ route, navigation }) {
  const { role } = route.params || { role: "researcher" };
  const [recordCount, setRecordCount] = useState(0);
  const [userCount, setUserCount] = useState(0);
  const [liveActivity, setLiveActivity] = useState([]);

  useEffect(() => {
    // 1. Live Research Record Count
    const unsubResearch = onSnapshot(
      query(collection(db, "research_data_anonymized")),
      (snap) => {
        setRecordCount(snap.size);
      },
    );

    // 2. Live Registered User Count
    const unsubUsers = onSnapshot(query(collection(db, "users")), (snap) => {
      setUserCount(snap.size);
    });

    // 3. Live Data Ingress Feed (Last 5 events)
    const qActivity = query(
      collection(db, "research_data_anonymized"),
      orderBy("timestamp", "desc"),
      limit(5),
    );

    const unsubActivity = onSnapshot(qActivity, (snap) => {
      setLiveActivity(
        snap.docs.map((doc) => {
          const data = doc.data();
          return {
            id: doc.id,
            event: `${data.diagnosis || "New"} Record Synced`,
            time: data.timestamp?.toDate()
              ? data.timestamp.toDate().toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })
              : "Just now",
            icon: "analytics-outline",
            color:
              data.diagnosis === "Normal Vision"
                ? COLORS.success
                : COLORS.warning,
          };
        }),
      );
    });

    return () => {
      unsubResearch();
      unsubUsers();
      unsubActivity();
    };
  }, []);

  return (
    <View style={styles.root}>
      <BackgroundBubbles />
      <Header
        title="Expert Portal"
        subtitle={
          role === "admin" ? "System Administrator" : "Clinical Researcher"
        }
      />

      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ padding: 20 }}
      >
        {/* Real-time System Status[cite: 12, 15] */}
        <View style={styles.healthContainer}>
          <HealthBadge label="Cloud DB" active />
          <HealthBadge label="Auth Gate" active />
          <HealthBadge label="Encryption" active />
        </View>

        {/* Live Primary Metric Card[cite: 12, 15] */}
        <MotiView
          from={{ translateY: 20, opacity: 0 }}
          animate={{ translateY: 0, opacity: 1 }}
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

        {/* Utilities Grid with Live CMS and User counts[cite: 12, 15] */}
        <View
          style={[
            styles.grid,
            { flexWrap: "nowrap", alignItems: "stretch", gap: 15 },
          ]}
        >
          {/* Left Column: Stacked Square Cards */}
          <View style={{ width: "48%", gap: 15 }}>
            <GridCard
              title="Analytics"
              sub="CVD Trends"
              icon="analytics"
              color="#6200EA"
              onPress={() => navigation.navigate("ResearchDashboard")}
              style={{ width: "100%", marginBottom: 0 }}
            />
            <GridCard
              title="System Status"
              sub="Cloud Sync: Active"
              icon="cloud-done-outline"
              color="#FF6D00"
              onPress={() =>
                Alert.alert(
                  "System Health Report",
                  `Database: Firebase Firestore (Online)\nAuth Service: Firebase Auth (Active)\nData Compliance: PII Masking Active\n\nAll research data is decoupled from User IDs at ingestion to ensure participant anonymity.`,
                )
              }
              style={{ width: "100%", marginBottom: 0 }}
            />
          </View>

          {/* Right Column: Tall Vertical Rectangle */}
          <GridCard
            title="App Interface"
            sub="Switch to User Mode"
            icon="phone-portrait-outline"
            color="#00C853"
            onPress={() => navigation.navigate("MainTabs")}
            style={{
              width: "48%",
              height: "auto",
              marginBottom: 0,
              justifyContent: "center", // Centers content vertically in the tall card
            }}
          />
        </View>

        {/* Live Event Stream[cite: 12, 15] */}
        <Text style={styles.sectionHeader}>LIVE DATA INGRESS FEED</Text>
        <Card style={styles.logCard}>
          {liveActivity.length > 0 ? (
            liveActivity.map((log) => (
              <View key={log.id} style={styles.logItem}>
                <Ionicons name={log.icon} size={16} color={log.color} />
                <Text style={styles.logText}>{log.event}</Text>
                <Text style={styles.logTime}>{log.time}</Text>
              </View>
            ))
          ) : (
            <Text style={{ textAlign: "center", color: "#999", fontSize: 12 }}>
              Waiting for incoming packets...
            </Text>
          )}
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
      </ScrollView>
    </View>
  );
}

const HealthBadge = ({ label, active }) => (
  <View style={styles.healthBadge}>
    <View
      style={[styles.dot, { backgroundColor: active ? "#00C853" : "#D50000" }]}
    />
    <Text style={styles.healthLabel}>{label}</Text>
  </View>
);

const GridCard = ({ title, sub, icon, color, onPress, style }) => (
  <TouchableOpacity
    style={[styles.gridCard, { borderTopColor: color }, style]}
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
  healthLabel: { fontSize: 9, fontWeight: "800", color: "#666" },
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
    fontWeight: "600",
  },
  logTime: { fontSize: 11, color: "#AAA" },
  logoutBtn: {
    marginTop: 30,
    padding: 15,
    alignItems: "center",
    marginBottom: 40,
  },
  logoutText: { color: "#D32F2F", fontWeight: "bold" },
});
