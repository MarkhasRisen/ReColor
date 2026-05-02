import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import { Image, Text, TouchableOpacity, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context"; // Import this!
import { auth } from "../../firebaseConfig";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";

export default function Header({ title, subtitle, back }) {
  const navigation = useNavigation();
  const insets = useSafeAreaInsets();
  const isActive = !!auth.currentUser;

  return (
    <View
      style={[
        styles.header,
        {
          paddingTop: insets.top + 10, // Dynamic padding for the notch
          paddingBottom: 15,
          backgroundColor: "rgba(255, 255, 255, 0.8)", // Glass look
        },
      ]}
    >
      <View style={{ flexDirection: "row", alignItems: "center", flex: 1 }}>
        {back ? (
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={{ marginRight: 12 }}
          >
            <Ionicons name="arrow-back" size={24} color={COLORS.text} />
          </TouchableOpacity>
        ) : (
          /* NEW: User Icon to make 'Hello' feel integrated */
          <View
            style={{
              width: 40,
              height: 40,
              borderRadius: 20,
              backgroundColor: COLORS.primary + "15",
              justifyContent: "center",
              alignItems: "center",
              marginRight: 12,
            }}
          >
            <Ionicons
              name="person-circle-outline"
              size={28}
              color={COLORS.primary}
            />
          </View>
        )}

        <View>
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <Text
              style={[styles.headerTitle, { fontSize: 18, fontWeight: "800" }]}
            >
              {title}
            </Text>
          </View>
          {subtitle && (
            <Text
              style={[
                styles.headerSubtitle,
                { fontSize: 12, color: COLORS.textLight },
              ]}
            >
              {subtitle}
            </Text>
          )}
        </View>
      </View>

      {/* Logo: Made slightly smaller (45x45) so it doesn't overpower the text */}
      <Image
        source={require("../../assets/icon.png")}
        style={{ width: 45, height: 45, resizeMode: "contain" }}
      />
    </View>
  );
}
