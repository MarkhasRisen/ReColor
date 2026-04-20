import { Ionicons } from "@expo/vector-icons";
import {
  Image,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { auth } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import { styles } from "../theme/styles";

const bentoCard = (bg, onPress, icon, iconColor, title, desc, cardStyle) => (
  <TouchableOpacity
    activeOpacity={0.85}
    onPress={onPress}
    style={[
      {
        backgroundColor: bg,
        borderRadius: 20,
        padding: 16,
        borderWidth: 1,
        borderColor: "rgba(0,0,0,0.06)",
      },
      cardStyle,
    ]}
  >
    <Ionicons
      name={icon}
      size={26}
      color={iconColor}
      style={{ marginBottom: 8 }}
    />
    <Text
      style={{
        fontWeight: "700",
        fontSize: 15,
        color: "#1A1A2E",
        marginBottom: 4,
      }}
    >
      {title}
    </Text>
    <Text style={{ fontSize: 12, color: "#666", lineHeight: 17 }}>{desc}</Text>
  </TouchableOpacity>
);

export default function HomeScreen({ navigation }) {
  const userName = (auth.currentUser?.email || "Guest").split("@")[0];
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      {/* Sticky disclaimer — padded for notch/camera bump */}
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: 6,
          backgroundColor: "#FFF8E1",
          paddingTop: insets.top + 4,
          paddingBottom: 5,
          paddingHorizontal: 12,
          borderBottomWidth: 1,
          borderBottomColor: "#FFE082",
        }}
      >
        <Ionicons name="warning-outline" size={12} color="#F59E0B" />
        <Text
          style={{
            fontSize: 10,
            fontWeight: "700",
            color: "#F59E0B",
            letterSpacing: 0.4,
          }}
        >
          NOT A MEDICAL DIAGNOSIS — SCREENING PURPOSE ONLY
        </Text>
      </View>
      <ScrollView
        contentContainerStyle={{ padding: 20, paddingBottom: 32 }}
        showsVerticalScrollIndicator={false}
        style={{ backgroundColor: "transparent" }}
      >
        <View style={{ marginTop: 10, marginBottom: 24 }}>
          <Text style={{ color: "#666", marginBottom: 5 }}>
            Hello, {userName}
          </Text>
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text
              style={{
                fontSize: 28,
                fontWeight: "bold",
                flex: 1,
                marginRight: 10,
              }}
            >
              Welcome to ReColor
            </Text>
            <Image
              source={require("../../assets/icon.png")}
              style={{ width: 180, height: 180, resizeMode: "contain" }}
            />
          </View>
        </View>

        {/* Hero — Ishihara Test */}
        {bentoCard(
          "#E3F2FD",
          () => navigation.navigate("IshiharaIntro"),
          "eye",
          "#2196F3",
          "Take Ishihara Test",
          "Screen for color vision deficiency with our digital 38-plate test",
          { marginBottom: 12 },
        )}

        {/* Row — Enhancement + Education */}
        <View style={{ flexDirection: "row", gap: 12, marginBottom: 12 }}>
          {bentoCard(
            "#F3E5F5",
            () => navigation.navigate("CameraEnhance"),
            "camera",
            "#9C27B0",
            "Color Enhancement",
            "Adaptive color correction",
            { flex: 1 },
          )}
          {bentoCard(
            "#FCE4EC",
            () => navigation.navigate("EducationList"),
            "book",
            "#E91E63",
            "Awareness & Education",
            "Articles and CVD simulation",
            { flex: 1 },
          )}
        </View>

        {/* Secondary — Survey */}
        {bentoCard(
          "#E8F5E9",
          () => navigation.navigate("Survey"),
          "clipboard",
          "#4CAF50",
          "Quick Survey",
          "Help us understand your color vision experience",
          { marginBottom: 0 },
        )}
      </ScrollView>
    </View>
  );
}
