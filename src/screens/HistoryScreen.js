import { Ionicons } from "@expo/vector-icons";
import { collection, onSnapshot, orderBy, query } from "firebase/firestore";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { auth, db, onAuthStateChanged } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Card from "../components/Card";
import Header from "../components/Header";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";

export default function HistoryScreen({ navigation }) {
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
        setHistoryData([]);
        setLoading(false);
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
                  ? d.date.toDate().toLocaleDateString()
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
      <Header title="Your History" back />
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: 3,
          backgroundColor: "#FFF8E1",
          paddingVertical: 10,
          paddingHorizontal: 8,
          borderBottomWidth: 1,
          borderBottomColor: "#FFE082",
        }}
      >
        <Ionicons name="warning-outline" size={12} color="#F59E0B" />
        <Text style={{ fontSize: 10, fontWeight: "700", color: "#F59E0B" }}>
          NOT A MEDICAL DIAGNOSIS — SCREENING PURPOSE ONLY
        </Text>
      </View>
      {loading ? (
        <View
          style={{ flex: 1, justifyContent: "center", alignItems: "center" }}
        >
          <ActivityIndicator size="large" color={COLORS.primary} />
        </View>
      ) : historyData.length === 0 ? (
        <View
          style={{
            flex: 1,
            justifyContent: "center",
            alignItems: "center",
            padding: 30,
          }}
        >
          {/* Approachable Illustration  */}
          <View
            style={{
              backgroundColor: "#F0F2FF",
              padding: 25,
              borderRadius: 100,
              marginBottom: 20,
            }}
          >
            <Ionicons name="stats-chart" size={60} color={COLORS.primary} />
          </View>

          {/* Human Labels & Conversational Copy  */}
          <Text
            style={{
              fontSize: 20,
              fontWeight: "800",
              color: "#1A1A2E",
              textAlign: "center",
            }}
          >
            Ready for your first screening?
          </Text>

          <Text
            style={{
              fontSize: 14,
              color: "#4A4A4A",
              textAlign: "center",
              marginTop: 10,
              lineHeight: 20,
            }}
          >
            Take a 5-minute test to start tracking your color perception
            journey. Your history will appear here after your first screening.
          </Text>

          {/* Call to Action Button  */}
          <TouchableOpacity
            style={[
              styles.btnPrimary,
              { marginTop: 30, width: "100%", paddingVertical: 16 },
            ]}
            onPress={() => navigation.navigate("IshiharaIntro")}
          >
            <Text style={[styles.btnText, { fontSize: 16 }]}>
              Take the test{" "}
            </Text>
          </TouchableOpacity>
        </View>
      ) : (
        <ScrollView contentContainerStyle={{ padding: 20 }}>
          {historyData.map((item, index) => (
            <Card key={index} style={{ marginBottom: 15, paddingVertical: 18 }}>
              <View
                style={{
                  flexDirection: "row",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <View style={{ flex: 1 }}>
                  {/* Improved Color Contrast for Accessibility  */}
                  <Text
                    style={{
                      fontSize: 17,
                      fontWeight: "bold",
                      color: "#1A1A2E",
                    }}
                  >
                    {item.type}
                  </Text>
                  <Text
                    style={{ fontSize: 13, color: "#4A4A4A", marginTop: 2 }}
                  >
                    Completed on {item.date}
                  </Text>
                </View>

                <View style={{ alignItems: "flex-end", marginLeft: 10 }}>
                  <Text
                    style={{
                      fontSize: 22,
                      fontWeight: "800",
                      color: COLORS.primary,
                    }}
                  >
                    {item.score}/{item.total}
                  </Text>
                  <View
                    style={{
                      backgroundColor:
                        item.severity === "Severe" ? "#FFEBEE" : "#FFF3E0",
                      paddingHorizontal: 8,
                      paddingVertical: 2,
                      borderRadius: 6,
                      marginTop: 4,
                    }}
                  >
                    <Text
                      style={{
                        fontSize: 11,
                        fontWeight: "800",
                        color:
                          item.severity === "Severe"
                            ? COLORS.danger
                            : "#E65100",
                      }}
                    >
                      {item.severity.toUpperCase()}
                    </Text>
                  </View>
                </View>
              </View>
            </Card>
          ))}
        </ScrollView>
      )}
    </View>
  );
}
