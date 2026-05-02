import { Ionicons } from "@expo/vector-icons";
import { Image, ScrollView, Text, TouchableOpacity, View } from "react-native";
import { auth } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Card from "../components/Card";
import { COLORS } from "../theme/colors";

export default function ProfileScreen({ navigation }) {
  const userEmail = auth.currentUser?.email || "Guest";
  const userName = userEmail.split("@")[0];

  return (
    <View style={{ flex: 1, backgroundColor: COLORS.background }}>
      <BackgroundBubbles />
      <ScrollView
        contentContainerStyle={{ padding: 20 }}
        showsVerticalScrollIndicator={false}
      >
        <View
          style={{
            marginTop: 20,
            marginBottom: 20,
            flexDirection: "row",
            alignItems: "center",
          }}
        >
          <View style={{ flex: 1 }}>
            <Text style={{ color: "#666", fontSize: 16 }}>
              Hello, {userName}
            </Text>
            <Text style={{ fontSize: 24, fontWeight: "bold", color: "#333" }}>
              Welcome to ReColor
            </Text>
          </View>
          <Image
            source={require("../../assets/icon.png")}
            style={{
              width: 120,
              height: 120,
              resizeMode: "contain",
              marginRight: 40,
            }}
          />
        </View>

        <Card style={{ alignItems: "center", paddingVertical: 30 }}>
          <View
            style={{
              width: 80,
              height: 80,
              borderRadius: 40,
              backgroundColor: COLORS.primary,
              alignItems: "center",
              justifyContent: "center",
              marginBottom: 15,
              elevation: 5,
            }}
          >
            <Ionicons name="person" size={40} color="#FFF" />
          </View>
          <Text style={{ fontSize: 20, fontWeight: "bold", color: "#333" }}>
            {userName}
          </Text>
          <Text style={{ color: "#888", marginBottom: 25 }}>{userEmail}</Text>
          <TouchableOpacity
            style={{
              backgroundColor: "#E8EAF6",
              paddingVertical: 12,
              paddingHorizontal: 40,
              borderRadius: 8,
              width: "100%",
              alignItems: "center",
            }}
            onPress={() => navigation.navigate("Settings")}
          >
            <Text style={{ color: "#333", fontWeight: "600" }}>
              Manage Settings
            </Text>
          </TouchableOpacity>
        </Card>
      </ScrollView>
    </View>
  );
}
