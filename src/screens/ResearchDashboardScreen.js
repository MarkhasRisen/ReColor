import { Ionicons } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system/legacy";
import * as Sharing from "expo-sharing";
import { collection, onSnapshot, orderBy, query } from "firebase/firestore";
import { useEffect, useState } from "react";
import {
    ActivityIndicator,
    Alert,
    ScrollView,
    Text,
    TouchableOpacity,
    View,
} from "react-native";
import { db } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Card from "../components/Card";
import Header from "../components/Header";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";

function StatBar({ label, count, total, color, badge }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <View style={{ marginBottom: 15 }}>
      <View
        style={{
          flexDirection: "row",
          justifyContent: "space-between",
          marginBottom: 5,
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center" }}>
          <Text style={{ fontSize: 13, color: "#333" }}>{label}</Text>
          {badge && (
            <View
              style={{
                marginLeft: 8,
                borderWidth: 1,
                borderColor: "#EEE",
                paddingHorizontal: 6,
                borderRadius: 4,
              }}
            >
              <Text style={{ fontSize: 10, color: "#555" }}>{badge}</Text>
            </View>
          )}
        </View>
        <Text style={{ fontWeight: "bold", fontSize: 13 }}>
          {count}{" "}
          <Text style={{ color: "#999", fontWeight: "normal" }}>({pct}%)</Text>
        </Text>
      </View>
      <View style={{ height: 8, backgroundColor: "#F0F0F0", borderRadius: 4 }}>
        <View
          style={{
            width: `${pct}%`,
            height: "100%",
            backgroundColor: color,
            borderRadius: 4,
          }}
        />
      </View>
    </View>
  );
}

function StatTile({ bg, icon, iconColor, val, label, valColor }) {
  return (
    <View
      style={{
        width: "48%",
        backgroundColor: bg,
        padding: 15,
        borderRadius: 12,
        marginBottom: 10,
      }}
    >
      <Ionicons name={icon} size={20} color={iconColor} />
      <Text
        style={{
          fontSize: 24,
          fontWeight: "bold",
          color: valColor,
          marginTop: 10,
        }}
      >
        {val}
      </Text>
      <Text style={{ fontSize: 11, color: iconColor }}>{label}</Text>
    </View>
  );
}

function countBy(docs, field) {
  const counts = {};
  docs.forEach((d) => {
    const key = d[field] || "Unknown";
    counts[key] = (counts[key] || 0) + 1;
  });
  return counts;
}

function toCSV(rows) {
  if (!rows.length) return "";
  const keys = Object.keys(rows[0]);
  const header = keys.join(",");
  const body = rows.map((r) =>
    keys
      .map((k) => {
        const v = r[k] == null ? "" : String(r[k]);
        return v.includes(",") || v.includes('"')
          ? `"${v.replace(/"/g, '""')}"`
          : v;
      })
      .join(","),
  );
  return [header, ...body].join("\n");
}

export default function ResearchDashboardScreen({ navigation }) {
  const [activeTab, setActiveTab] = useState("View Data");
  const [testDocs, setTestDocs] = useState([]);
  const [surveyDocs, setSurveyDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    setLoading(true);
    let done = 0;
    const finish = () => {
      done++;
      if (done === 2) setLoading(false);
    };

    const unsubTests = onSnapshot(
      query(
        collection(db, "research_data_anonymized"),
        orderBy("timestamp", "desc"),
      ),
      (snap) => {
        setTestDocs(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
        finish();
      },
      () => finish(),
    );
    const unsubSurveys = onSnapshot(
      query(collection(db, "surveys"), orderBy("timestamp", "desc")),
      (snap) => {
        setSurveyDocs(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
        finish();
      },
      () => finish(),
    );

    return () => {
      unsubTests();
      unsubSurveys();
    };
  }, []);

  const diagCounts = countBy(testDocs, "diagnosis");
  const total = testDocs.length;

  const handleExport = async () => {
    if (!total) {
      Alert.alert("No Data", "No test results to export yet.");
      return;
    }
    setExporting(true);
    try {
      const rows = testDocs.map(({ id, timestamp, ...rest }) => ({
        ...rest,
        timestamp: timestamp?.toDate ? timestamp.toDate().toISOString() : "",
      }));
      const csv = toCSV(rows);
      const path = `${FileSystem.documentDirectory}recolor_research_${Date.now()}.csv`;

      // Fix for Screenshot 2: Use EncodingType explicitly
      await FileSystem.writeAsStringAsync(path, csv, {
        encoding: "utf8",
      });

      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(path);
      } else {
        Alert.alert(
          "Not Available",
          "Sharing is not supported on this device.",
        );
      }
    } catch (e) {
      Alert.alert("Export Failed", e?.message || "Unknown error");
    } finally {
      setExporting(false);
    }
  };

  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      <Header
        title="Research Dashboard"
        subtitle="PERI Researcher Portal"
        back
      />

      <View style={{ flexDirection: "row", padding: 20, paddingBottom: 0 }}>
        {["View Data", "Guidelines"].map((tab, i) => (
          <TouchableOpacity
            key={tab}
            style={{
              flex: 1,
              paddingVertical: 10,
              backgroundColor: activeTab === tab ? "#FFF" : "#F5F5F5",
              alignItems: "center",
              borderTopLeftRadius: i === 0 ? 20 : 0,
              borderBottomLeftRadius: i === 0 ? 20 : 0,
              borderTopRightRadius: i === 1 ? 20 : 0,
              borderBottomRightRadius: i === 1 ? 20 : 0,
              borderWidth: 1,
              borderColor: "#E0E0E0",
            }}
            onPress={() => setActiveTab(tab)}
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <Ionicons
                name={i === 0 ? "stats-chart" : "document-text"}
                size={18}
                color={activeTab === tab ? "#333" : "#999"}
              />
              <Text
                style={{
                  marginLeft: 8,
                  fontWeight: "bold",
                  color: activeTab === tab ? "#333" : "#999",
                }}
              >
                {tab}
              </Text>
            </View>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView
        contentContainerStyle={{ padding: 20 }}
        showsVerticalScrollIndicator={false}
      >
        {activeTab === "View Data" ? (
          <>
            <View
              style={{
                backgroundColor: "#E1F5FE",
                padding: 15,
                borderRadius: 8,
                marginBottom: 20,
                borderWidth: 1,
                borderColor: "#B3E5FC",
              }}
            >
              <Text style={{ color: "#0277BD", fontSize: 12, lineHeight: 18 }}>
                <Text style={{ fontWeight: "bold" }}>Privacy Protected: </Text>
                All data is anonymized. No personally identifiable information
                (PII) is visible.
              </Text>
            </View>

            {loading ? (
              <ActivityIndicator
                size="large"
                color={COLORS.primary}
                style={{ marginTop: 40 }}
              />
            ) : (
              <>
                <View
                  style={{
                    flexDirection: "row",
                    flexWrap: "wrap",
                    justifyContent: "space-between",
                    marginBottom: 20,
                  }}
                >
                  <StatTile
                    bg="#E3F2FD"
                    icon="trending-up"
                    iconColor="#1E88E5"
                    val={total}
                    label="Total Tests"
                    valColor="#1565C0"
                  />
                  <StatTile
                    bg="#F3E5F5"
                    icon="people"
                    iconColor="#8E24AA"
                    val={surveyDocs.length}
                    label="Surveys"
                    valColor="#6A1B9A"
                  />
                </View>

                <TouchableOpacity
                  style={{
                    backgroundColor: "#FFF",
                    borderWidth: 1,
                    borderColor: "#DDD",
                    padding: 15,
                    borderRadius: 12,
                    alignItems: "center",
                    marginBottom: 25,
                    flexDirection: "row",
                    justifyContent: "center",
                  }}
                  onPress={handleExport}
                >
                  <Ionicons
                    name="download-outline"
                    size={20}
                    color="#333"
                    style={{ marginRight: 10 }}
                  />
                  <Text style={{ fontWeight: "bold" }}>
                    {exporting ? "Preparing..." : "Export Full Report (CSV)"}
                  </Text>
                </TouchableOpacity>

                <Card style={{ marginBottom: 20 }}>
                  <Text style={{ fontWeight: "bold", marginBottom: 15 }}>
                    Screening Statistics
                  </Text>
                  <StatBar
                    label="Normal Vision"
                    count={diagCounts["Normal Vision"] || 0}
                    total={total}
                    color={COLORS.success}
                    badge="N"
                  />
                  <StatBar
                    label="CVD Indicated"
                    count={total - (diagCounts["Normal Vision"] || 0)}
                    total={total}
                    color={COLORS.danger}
                    badge="CVD"
                  />
                </Card>
              </>
            )}
          </>
        ) : (
          <Card>
            <Text style={{ fontWeight: "bold", marginBottom: 10 }}>
              Active Protocols
            </Text>
            <Text style={{ fontSize: 12, color: "#555", lineHeight: 20 }}>
              • Ishihara Test — 25 plates (comprehensive){"\n"}• Stage 1:
              Screening via plates 1–21{"\n"}• Stage 2: Differentiation via
              plates 22–25{"\n"}• Thresholds: ≥17 Normal · 14–16 Indeterminate
              {"\n"}• Offline-first: Firestore persistentLocalCache
            </Text>
          </Card>
        )}
      </ScrollView>
    </View>
  );
}
