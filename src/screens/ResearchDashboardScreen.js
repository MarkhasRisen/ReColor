import { Ionicons } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system";
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
  const severityCounts = countBy(testDocs, "severity");
  const total = testDocs.length;

  const DIAG_BARS = [
    {
      key: "Normal Vision",
      label: "Normal Vision",
      badge: "N",
      color: COLORS.success,
    },
    {
      key: "Indeterminate Result",
      label: "Indeterminate",
      badge: "I",
      color: "#9C27B0",
    },
    { key: "Mild", label: "Mild CVD", badge: "Mild", color: "#2979FF" },
    { key: "Moderate", label: "Moderate CVD", badge: "Mod", color: "#FF9800" },
    { key: "Severe", label: "Severe CVD", badge: "Sev", color: COLORS.danger },
  ];

  const diagBarData = DIAG_BARS.map((b) => ({
    ...b,
    count:
      b.key === "Mild" || b.key === "Moderate" || b.key === "Severe"
        ? testDocs.filter((d) => (d.severity || "") === b.key).length
        : diagCounts[b.key] || 0,
  }));

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
      await FileSystem.writeAsStringAsync(path, csv, {
        encoding: "utf8",
      });
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(path, {
          mimeType: "text/csv",
          dialogTitle: "Export Research Data",
          UTI: "public.comma-separated-values-text",
        });
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
      <Header
        title="Research Dashboard"
        subtitle="PERI Researcher Portal"
        back
      />
      <BackgroundBubbles />

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
              borderRightWidth: i === 0 ? 0 : 1,
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
        {activeTab === "View Data" && (
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
              <View style={{ alignItems: "center", paddingVertical: 40 }}>
                <ActivityIndicator size="large" color={COLORS.primary} />
                <Text style={{ color: "#999", marginTop: 10 }}>
                  Loading live data…
                </Text>
              </View>
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
                    label="Survey Responses"
                    valColor="#6A1B9A"
                  />
                  <StatTile
                    bg="#E8F5E9"
                    icon="bar-chart"
                    iconColor="#43A047"
                    val={diagCounts["Normal Vision"] || 0}
                    label="Normal Results"
                    valColor="#2E7D32"
                  />
                  <StatTile
                    bg="#FCE4EC"
                    icon="alert-circle"
                    iconColor="#E91E63"
                    val={total - (diagCounts["Normal Vision"] || 0)}
                    label="CVD Indicated"
                    valColor="#C2185B"
                  />
                </View>

                <TouchableOpacity
                  style={{
                    backgroundColor: "#FFF",
                    borderWidth: 1,
                    borderColor: "#DDD",
                    padding: 12,
                    borderRadius: 8,
                    alignItems: "center",
                    marginBottom: 25,
                    flexDirection: "row",
                    justifyContent: "center",
                    gap: 8,
                  }}
                  onPress={handleExport}
                  disabled={exporting}
                >
                  {exporting ? (
                    <ActivityIndicator size="small" color="#333" />
                  ) : (
                    <Ionicons name="download-outline" size={18} color="#333" />
                  )}
                  <Text style={{ fontWeight: "bold", color: "#333" }}>
                    {exporting
                      ? "Preparing CSV…"
                      : `Export Full Report (CSV) — ${total} records`}
                  </Text>
                </TouchableOpacity>

                <Card style={{ marginBottom: 20 }}>
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      marginBottom: 20,
                    }}
                  >
                    <Ionicons name="bar-chart-outline" size={20} color="#333" />
                    <Text
                      style={{
                        fontWeight: "bold",
                        marginLeft: 10,
                        fontSize: 16,
                      }}
                    >
                      Screening Statistics
                    </Text>
                  </View>
                  {diagBarData.map((b) => (
                    <StatBar
                      key={b.key}
                      label={b.label}
                      badge={b.badge}
                      count={b.count}
                      total={total}
                      color={b.color}
                    />
                  ))}
                </Card>

                <Card style={{ marginBottom: 20 }}>
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      marginBottom: 20,
                    }}
                  >
                    <Ionicons name="people-outline" size={20} color="#333" />
                    <Text
                      style={{
                        fontWeight: "bold",
                        marginLeft: 10,
                        fontSize: 16,
                      }}
                    >
                      Survey Demographics
                    </Text>
                  </View>
                  {surveyDocs.length === 0 ? (
                    <Text style={{ color: "#999", fontSize: 13 }}>
                      No survey responses yet.
                    </Text>
                  ) : (
                    <>
                      <Text
                        style={{
                          fontSize: 12,
                          color: "#666",
                          marginBottom: 10,
                          fontWeight: "600",
                        }}
                      >
                        Sex Distribution
                      </Text>
                      {["Male", "Female"].map((sex) => {
                        const count = surveyDocs.filter(
                          (d) => d.sex === sex,
                        ).length;
                        return (
                          <StatBar
                            key={sex}
                            label={sex}
                            count={count}
                            total={surveyDocs.length}
                            color={sex === "Male" ? "#2979FF" : "#E91E63"}
                          />
                        );
                      })}
                      <Text
                        style={{
                          fontSize: 12,
                          color: "#666",
                          marginBottom: 10,
                          marginTop: 10,
                          fontWeight: "600",
                        }}
                      >
                        Perceived Cause
                      </Text>
                      {["Medical Intake", "Genetics", "Ageing", "Others"].map(
                        (cause) => {
                          const count = surveyDocs.filter(
                            (d) => d.cause === cause,
                          ).length;
                          return (
                            <StatBar
                              key={cause}
                              label={cause}
                              count={count}
                              total={surveyDocs.length}
                              color={COLORS.primary}
                            />
                          );
                        },
                      )}
                    </>
                  )}
                </Card>
              </>
            )}
          </>
        )}

        {activeTab === "Guidelines" && (
          <>
            <Card style={{ marginBottom: 20 }}>
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  marginBottom: 10,
                }}
              >
                <Ionicons name="document-text-outline" size={20} color="#333" />
                <Text
                  style={{ fontWeight: "bold", marginLeft: 10, fontSize: 16 }}
                >
                  Clinical Scoring Thresholds
                </Text>
              </View>
              <View
                style={{
                  backgroundColor: "#F5F5F5",
                  borderRadius: 8,
                  padding: 15,
                }}
              >
                {[
                  {
                    range: "≥ 80%",
                    label: "Normal Vision",
                    color: COLORS.success,
                  },
                  {
                    range: "65–79%",
                    label: "Indeterminate (borderline)",
                    color: "#9C27B0",
                  },
                  { range: "45–64%", label: "Mild CVD", color: "#2979FF" },
                  { range: "25–44%", label: "Moderate CVD", color: "#FF9800" },
                  { range: "< 25%", label: "Severe CVD", color: COLORS.danger },
                ].map((row) => (
                  <View
                    key={row.range}
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      marginBottom: 8,
                    }}
                  >
                    <View
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 5,
                        backgroundColor: row.color,
                        marginRight: 10,
                      }}
                    />
                    <Text style={{ fontSize: 13, color: "#333", flex: 1 }}>
                      {row.label}
                    </Text>
                    <Text
                      style={{ fontSize: 12, color: "#999", fontWeight: "600" }}
                    >
                      {row.range}
                    </Text>
                  </View>
                ))}
              </View>
              <View
                style={{
                  marginTop: 15,
                  backgroundColor: "#E3F2FD",
                  padding: 12,
                  borderRadius: 8,
                }}
              >
                <Text
                  style={{ fontSize: 11, color: "#1565C0", lineHeight: 17 }}
                >
                  Indeterminate zone mirrors the clinical 14–16 correct plates
                  threshold derived from the standard Ishihara scoring protocol
                  (Ishihara, 1917). Stage 1 uses 21 plates; Stage 2 adds 4
                  diagnostic plates.
                </Text>
              </View>
            </Card>

            <Card>
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  marginBottom: 10,
                }}
              >
                <Ionicons
                  name="information-circle-outline"
                  size={20}
                  color="#333"
                />
                <Text
                  style={{ fontWeight: "bold", marginLeft: 10, fontSize: 16 }}
                >
                  Active Protocols
                </Text>
              </View>
              <Text style={{ fontSize: 11, color: "#555", lineHeight: 20 }}>
                {
                  "• Ishihara Test — 25 plates (comprehensive)\n• Stage 1: Screening via plates 1–21 (demo + 20 screening/vanishing/hidden)\n• Stage 2: Differentiation via plates 22–25 (weight ×2)\n• Thresholds: ≥17 Normal · 14–16 Indeterminate · ≤13 proceed to Stage 2\n• Dual-write: private history + anonymized research\n• Offline-first: Firestore persistentLocalCache"
                }
              </Text>
            </Card>
          </>
        )}
      </ScrollView>
    </View>
  );
}
