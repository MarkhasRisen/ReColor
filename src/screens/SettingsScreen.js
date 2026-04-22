import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";
import { useEffect, useState } from "react";
import {
  Alert,
  ScrollView,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { auth, signOut } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Card from "../components/Card";
import Header from "../components/Header";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";
import { AppLog } from "../utils/logger";

export default function SettingsScreen({ navigation }) {
  const userEmail = auth.currentUser?.email || "Guest";
  const userName = userEmail.split("@")[0];
  const [enhancement, setEnhancement] = useState(100);

  useEffect(() => {
    AsyncStorage.getItem("@recolor_intensity").then((val) => {
      if (val !== null) setEnhancement(Number(val));
    });
  }, []);

  const handleIntensityChange = (val) => {
    setEnhancement(val);
    AsyncStorage.setItem("@recolor_intensity", String(Math.round(val)));
  };
  const [audio, setAudio] = useState(true);
  const [showLogs, setShowLogs] = useState(false);
  const [logText, setLogText] = useState("");

  return (
    <View style={styles.container}>
      <Header title="Settings" back />
      <BackgroundBubbles />

      <ScrollView
        contentContainerStyle={{ padding: 20 }}
        showsVerticalScrollIndicator={false}
      >
        <Card style={{ marginBottom: 20 }}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              marginBottom: 15,
            }}
          >
            <Ionicons name="person-outline" size={20} color="#555" />
            <Text style={{ marginLeft: 10, fontWeight: "bold", fontSize: 16 }}>
              Profile
            </Text>
          </View>
          <View style={{ marginBottom: 10 }}>
            <Text style={{ color: "#999", fontSize: 12 }}>Name</Text>
            <Text style={{ fontSize: 16, color: "#333", fontWeight: "500" }}>
              {userName}
            </Text>
          </View>
          <View>
            <Text style={{ color: "#999", fontSize: 12 }}>Email</Text>
            <Text style={{ fontSize: 16, color: "#333", fontWeight: "500" }}>
              {userEmail}
            </Text>
          </View>
        </Card>

        <Card style={{ marginBottom: 20 }}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              marginBottom: 15,
            }}
          >
            <Ionicons name="settings-outline" size={20} color="#555" />
            <Text style={{ marginLeft: 10, fontWeight: "bold", fontSize: 16 }}>
              Preferences
            </Text>
          </View>

          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-between",
              marginBottom: 5,
            }}
          >
            <Text style={{ fontWeight: "600", color: "#333" }}>
              Color Enhancement Intensity
            </Text>
            <Text style={{ color: "#999" }}>{Math.round(enhancement)}%</Text>
          </View>
          <Slider
            style={{ width: "100%", height: 40 }}
            minimumValue={0}
            maximumValue={100}
            step={1}
            value={enhancement}
            onValueChange={handleIntensityChange}
            minimumTrackTintColor={COLORS.primary}
            maximumTrackTintColor="#E0E0E0"
            thumbTintColor={COLORS.primary}
          />
          <Text style={{ fontSize: 11, color: "#999", marginBottom: 25 }}>
            Adjust the strength of color enhancement in camera mode
          </Text>

          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <View style={{ flex: 1, paddingRight: 20 }}>
              <Text style={{ fontWeight: "600", color: "#333" }}>
                Audio Feedback
              </Text>
              <Text style={{ fontSize: 11, color: "#999", marginTop: 2 }}>
                Enable voice announcements for color identification
              </Text>
            </View>
            <Switch
              value={audio}
              onValueChange={setAudio}
              trackColor={{ false: "#767577", true: "#333" }}
              thumbColor={audio ? "#FFF" : "#f4f3f4"}
            />
          </View>
        </Card>

        <Card style={{ marginBottom: 20 }}>
          <Text
            style={{
              marginBottom: 15,
              fontWeight: "bold",
              fontSize: 16,
              color: "#333",
            }}
          >
            Data
          </Text>
          <TouchableOpacity
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
            }}
            onPress={() =>
              navigation.navigate("MainTabs", { screen: "History" })
            }
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <Ionicons name="time-outline" size={24} color="#333" />
              <View style={{ marginLeft: 15 }}>
                <Text style={{ fontWeight: "600", fontSize: 15 }}>
                  View Test History
                </Text>
                <Text style={{ color: "#999", fontSize: 12 }}>
                  3 tests completed
                </Text>
              </View>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#CCC" />
          </TouchableOpacity>
        </Card>

        <Card style={{ marginBottom: 20 }}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: 10,
            }}
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <Ionicons name="bug-outline" size={20} color="#555" />
              <Text
                style={{
                  marginLeft: 10,
                  fontWeight: "bold",
                  fontSize: 16,
                  color: "#333",
                }}
              >
                Debug Logs
              </Text>
            </View>
            <View style={{ flexDirection: "row" }}>
              <TouchableOpacity
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: 6,
                  backgroundColor: COLORS.primary,
                  marginRight: 8,
                }}
                onPress={async () => {
                  const logs = await AppLog.readAll();
                  setLogText(logs);
                  setShowLogs(!showLogs);
                }}
              >
                <Text
                  style={{ color: "#FFF", fontSize: 12, fontWeight: "600" }}
                >
                  {showLogs ? "Hide" : "View"}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: 6,
                  backgroundColor: "#D32F2F",
                }}
                onPress={() => {
                  AppLog.clear();
                  setLogText("");
                  setShowLogs(false);
                  Alert.alert("Cleared", "Debug logs cleared.");
                }}
              >
                <Text
                  style={{ color: "#FFF", fontSize: 12, fontWeight: "600" }}
                >
                  Clear
                </Text>
              </TouchableOpacity>
            </View>
          </View>
          <Text style={{ fontSize: 11, color: "#999", marginBottom: 5 }}>
            Camera and pipeline logs are saved here for debugging crashes.
          </Text>
          {showLogs && (
            <ScrollView
              style={{
                maxHeight: 250,
                backgroundColor: "#1a1a2e",
                borderRadius: 8,
                padding: 10,
                marginTop: 8,
              }}
              nestedScrollEnabled
            >
              <Text
                style={{ color: "#0f0", fontSize: 10, fontFamily: "monospace" }}
              >
                {logText || "(no logs yet)"}
              </Text>
            </ScrollView>
          )}
        </Card>

        <Card>
          <Text
            style={{
              marginBottom: 15,
              fontWeight: "bold",
              fontSize: 16,
              color: "#333",
            }}
          >
            Account
          </Text>
          <TouchableOpacity
            style={[
              styles.btnPrimary,
              {
                backgroundColor: "#D32F2F",
                paddingVertical: 12,
                marginBottom: 12,
              },
            ]}
            onPress={() => {
              signOut(auth).catch(() => {});
              navigation.replace("Login");
            }}
          >
            <Ionicons
              name="log-out-outline"
              size={20}
              color="#FFF"
              style={{ marginRight: 8 }}
            />
            <Text style={{ color: "#FFF", fontWeight: "bold" }}>Log Out</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btnPrimary, { backgroundColor: "#607D8B" }]}
            onPress={() =>
              Alert.alert(
                "Reset Onboarding",
                "This will restart the onboarding flow on next launch. Continue?",
                [
                  { text: "Cancel", style: "cancel" },
                  {
                    text: "Reset",
                    style: "destructive",
                    onPress: () => {
                      AsyncStorage.removeItem("@recolor_onboarded");
                      navigation.replace("AppOnboarding");
                    },
                  },
                ],
              )
            }
          >
            <Ionicons
              name="refresh-outline"
              size={20}
              color="#FFF"
              style={{ marginRight: 8 }}
            />
            <Text style={{ color: "#FFF", fontWeight: "bold" }}>
              Reset Onboarding (Dev)
            </Text>
          </TouchableOpacity>
        </Card>
      </ScrollView>
    </View>
  );
}
