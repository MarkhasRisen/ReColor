import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { collection, onSnapshot, orderBy, query } from "firebase/firestore";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, { FadeInDown, FadeInUp } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { auth, db, onAuthStateChanged } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Header from "../components/Header";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

const getDiagnosisColors = (type) => {
  const t = type ? type.toLowerCase() : "";
  if (t.includes("normal")) {
    return {
      text: "#1E293B",
      bg: "#F1F5F9",
      border: "#E2E8F0",
    };
  } else if (t.includes("prot")) {
    return {
      text: "#991B1B",
      bg: "#FEF2F2",
      border: "#FEE2E2",
    };
  } else if (t.includes("deut")) {
    return {
      text: "#166534",
      bg: "#F0FDF4",
      border: "#DCFCE7",
    };
  } else if (t.includes("trit")) {
    return {
      text: "#0D47A1",
      bg: "#E3F2FD",
      border: "#BBDEFB",
    };
  } else {
    return {
      text: "#B45309",
      bg: "#FEF3C7",
      border: "#FDE68A",
    };
  }
};

const getSeverityColors = (severity) => {
  const s = severity ? severity.toLowerCase() : "";
  if (s.includes("severe")) {
    return {
      text: "#EF4444", // Red
      bg: "#FEF2F2",
      border: "#FEE2E2",
    };
  } else if (s.includes("moderate")) {
    return {
      text: "#F97316", // Orange
      bg: "#FFF7ED",
      border: "#FFEDD5",
    };
  } else if (s.includes("mild")) {
    return {
      text: "#22C55E", // Green
      bg: "#F0FDF4",
      border: "#DCFCE7",
    };
  } else if (s.includes("borderline")) {
    return {
      text: "#D97706", // Amber
      bg: "#FEF3C7",
      border: "#FDE68A",
    };
  } else {
    // None / N/A
    return {
      text: "#64748B", // Gray
      bg: "#F1F5F9",
      border: "#E2E8F0",
    };
  }
};

export default function HistoryScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let unsubFirestore = null;

    const unsubAuth = onAuthStateChanged(auth, (user) => {
      if (unsubFirestore) {
        unsubFirestore();
        unsubFirestore = null;
      }
      if (!user) {
        AsyncStorage.getItem("@recolor_local_history")
          .then((localData) => {
            if (localData) {
              try {
                const parsed = JSON.parse(localData);
                const mapped = (Array.isArray(parsed) ? parsed : []).map((item) => ({
                  id: item.id || Date.now().toString(),
                  type: item.diagnosis || item.type || "Unknown",
                  score: item.score !== undefined ? item.score : "?",
                  total: item.total || 14,
                  severity: item.severity || "N/A",
                  date: item.date || "Just now",
                }));
                setHistoryData(mapped);
              } catch (_e) {
                setHistoryData([]);
              }
            } else {
              setHistoryData([]);
            }
          })
          .catch((err) => {
            console.error("Local history fetch error:", err);
            setHistoryData([]);
          })
          .finally(() => {
            setLoading(false);
          });
        return;
      }

      setLoading(true);
      const q = query(
        collection(db, "users", user.uid, "history"),
        orderBy("date", "desc"),
      );
      unsubFirestore = onSnapshot(
        q,
        (snap) => {
          setHistoryData(
            snap.docs.map((doc) => {
              const d = doc.data();
              return {
                id: doc.id,
                type: d.diagnosis || "Unknown",
                score: d.score !== undefined ? d.score : "?",
                total: d.total || 14,
                severity: d.severity || "N/A",
                date: d.date?.toDate
                  ? d.date.toDate().toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })
                  : "Just now",
              };
            }),
          );
          setLoading(false);
        },
        (err) => {
          console.error("History fetch error:", err);
          setLoading(false);
        },
      );
    });

    return () => {
      unsubAuth();
      if (unsubFirestore) unsubFirestore();
    };
  }, []);

  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      <Header title="Clinical History" back />

      {/* Utility Disclaimer Banner */}
      <View style={styles.disclaimer}>
        <Ionicons name="shield-checkmark" size={12} color="#B45309" />
        <Text style={styles.disclaimerText}>
          OFFICIAL SCREENING LOG • NOT A MEDICAL DIAGNOSIS
        </Text>
      </View>

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator size="large" color={COLORS.primary} />
        </View>
      ) : historyData.length === 0 ? (
        <Animated.View entering={FadeInUp} style={styles.emptyContainer}>
          <View style={styles.emptyIconBox}>
            <Ionicons name="stats-chart" size={60} color={COLORS.primary} />
          </View>
          <Text style={styles.emptyTitle}>No Assessments Yet</Text>
          <Text style={styles.emptyDesc}>
            Your color perception journey starts here. Take your first Ishihara
            test to generate your clinical profile.
          </Text>
          <TouchableOpacity
            style={styles.ctaButton}
            onPress={() => navigation.navigate("IshiharaIntro")}
          >
            <Text style={styles.ctaText}>Begin Initial Screening</Text>
            <Ionicons name="arrow-forward" size={18} color="#FFF" />
          </TouchableOpacity>
        </Animated.View>
      ) : (
        <ScrollView
          contentContainerStyle={[
            styles.listContent,
            { paddingBottom: insets.bottom + 20 },
          ]}
          showsVerticalScrollIndicator={false}
        >
          <Text style={styles.sectionLabel}>RECENT SCREENINGS</Text>
          {historyData.map((item, index) => {
            const colors = getDiagnosisColors(item.type);
            const sevColors = getSeverityColors(item.severity);
            return (
              <Animated.View
                key={item.id}
                entering={FadeInDown.delay(index * 100).duration(500)}
                style={[
                  styles.historyCard,
                  {
                    borderLeftWidth: 6,
                    borderLeftColor: colors.text,
                    borderColor: colors.border,
                  },
                ]}
              >
                <View style={styles.cardHeader}>
                  <View
                    style={[
                      styles.typeBadge,
                      {
                        backgroundColor: colors.bg,
                        borderColor: colors.border,
                        borderWidth: 1,
                      },
                    ]}
                  >
                    <Text style={[styles.typeText, { color: colors.text }]}>
                      {item.type}
                    </Text>
                  </View>
                  <Text style={styles.dateText}>{item.date}</Text>
                </View>

              <View style={styles.cardBody}>
                <View>
                  <Text style={styles.scoreLabel}>Accuracy Score</Text>
                  <Text style={styles.scoreValue}>
                    {item.score}
                    <Text style={styles.scoreTotal}>/{item.total}</Text>
                  </Text>
                </View>

                <View style={styles.severityContainer}>
                  <Text style={styles.severityLabel}>SEVERITY</Text>
                  <View
                    style={[
                      styles.severityBadge,
                      {
                        backgroundColor: sevColors.bg,
                        borderColor: sevColors.border,
                        borderWidth: 1,
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.severityText,
                        {
                          color: sevColors.text,
                        },
                      ]}
                    >
                      {item.severity.toUpperCase()}
                    </Text>
                  </View>
                </View>
              </View>
            </Animated.View>
          );
        })}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F8FAFC" },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  disclaimer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: "#FFF8E1",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#FFE082",
  },
  disclaimerText: {
    fontSize: 9,
    fontWeight: "900",
    color: "#B45309",
    letterSpacing: 1,
  },
  listContent: { padding: SPACING.lg },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: SPACING.md,
    opacity: 0.7,
  },
  historyCard: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    padding: SPACING.lg,
    marginBottom: SPACING.md,
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  typeBadge: {
    backgroundColor: COLORS.primary + "10",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: RADIUS.md,
  },
  typeText: {
    fontSize: 13,
    fontWeight: "800",
    color: COLORS.primary,
  },
  dateText: {
    fontSize: 12,
    color: COLORS.textLight,
    fontWeight: "600",
  },
  cardBody: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
  },
  scoreLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.textLight,
    marginBottom: 2,
  },
  scoreValue: {
    fontSize: 28,
    fontWeight: "900",
    color: COLORS.text,
  },
  scoreTotal: {
    fontSize: 16,
    color: COLORS.textLight,
  },
  severityContainer: { alignItems: "flex-end" },
  severityLabel: {
    fontSize: 9,
    fontWeight: "900",
    color: COLORS.textLight,
    marginBottom: 4,
  },
  severityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: RADIUS.md,
  },
  severityText: {
    fontSize: 11,
    fontWeight: "900",
  },
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 40,
  },
  emptyIconBox: {
    backgroundColor: "#EEF2FF",
    padding: 30,
    borderRadius: 50,
    marginBottom: 24,
  },
  emptyTitle: {
    fontSize: 22,
    fontWeight: "900",
    color: COLORS.text,
    textAlign: "center",
  },
  emptyDesc: {
    fontSize: 14,
    color: COLORS.textLight,
    textAlign: "center",
    marginTop: 12,
    lineHeight: 22,
  },
  ctaButton: {
    backgroundColor: COLORS.primary,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: RADIUS.lg,
    marginTop: 32,
    gap: 10,
    ...SHADOW.md,
  },
  ctaText: {
    color: "#FFF",
    fontWeight: "800",
    fontSize: 16,
  },
});
