import { Ionicons } from "@expo/vector-icons";
import { ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Card from "../components/Card";
import Header from "../components/Header";
import { styles } from "../theme/styles";
export default function IshiharaIntroScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      <Header title="Choose Test Type" back />
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
      <ScrollView contentContainerStyle={{ padding: 20 }}>
        <Text style={{ color: "#777", marginBottom: 20 }}>
          Select the test duration that works best for you.
        </Text>

        <Card
          onPress={() =>
            navigation.navigate("IshiharaOnboarding", {
              testType: "comprehensive",
            })
          }
          style={{ marginBottom: 20 }}
        >
          <View style={{ flexDirection: "row" }}>
            <View style={{ flex: 1 }}>
              <Text style={styles.cardTitle}>Comprehensive Test</Text>
              <Text style={styles.cardDesc}>Complete 25-plate assessment</Text>
              <View
                style={{
                  marginTop: 10,
                  flexDirection: "row",
                  alignItems: "center",
                }}
              >
                <Ionicons name="time-outline" size={16} color="#666" />
                <Text style={{ fontSize: 12, marginLeft: 5, color: "#666" }}>
                  5-8 minutes
                </Text>
              </View>
            </View>
            <View style={{ justifyContent: "center", alignItems: "center" }}>
              <View style={[styles.iconCircle, { backgroundColor: "#E3F2FD" }]}>
                <Ionicons name="shield-checkmark" size={24} color="#2196F3" />
              </View>
              <Text
                style={{
                  fontSize: 10,
                  color: "#2196F3",
                  marginTop: 5,
                  textAlign: "center",
                }}
              >
                Accurate
              </Text>
            </View>
          </View>
        </Card>

        <Card
          onPress={() =>
            navigation.navigate("IshiharaOnboarding", { testType: "quick" })
          }
          style={{ marginBottom: 20 }}
        >
          <View style={{ flexDirection: "row" }}>
            <View style={{ flex: 1 }}>
              <Text style={styles.cardTitle}>Quick Test</Text>
              <Text style={styles.cardDesc}>14-plate screening</Text>
              <View
                style={{
                  marginTop: 10,
                  flexDirection: "row",
                  alignItems: "center",
                }}
              >
                <Ionicons name="time-outline" size={16} color="#666" />
                <Text style={{ fontSize: 12, marginLeft: 5, color: "#666" }}>
                  2-3 minutes
                </Text>
              </View>
            </View>
            <View style={{ justifyContent: "center", alignItems: "center" }}>
              <View style={[styles.iconCircle, { backgroundColor: "#F3E5F5" }]}>
                <Ionicons name="flash" size={24} color="#9C27B0" />
              </View>
              <Text
                style={{
                  fontSize: 10,
                  color: "#9C27B0",
                  marginTop: 5,
                  textAlign: "center",
                }}
              >
                Fast
              </Text>
            </View>
          </View>
        </Card>
      </ScrollView>
    </View>
  );
}
