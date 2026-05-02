import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import { Image, Text, TouchableOpacity, View } from "react-native";
import { styles } from "../theme/styles";

export default function Header({ title, subtitle, back }) {
  const navigation = useNavigation();
  return (
    <View style={styles.header}>
      <View style={{ flexDirection: "row", alignItems: "center", flex: 1 }}>
        {back && (
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={{ marginRight: 15 }}
          >
            <Ionicons name="arrow-back" size={24} color="#000" />
          </TouchableOpacity>
        )}
        <View>
          <Text style={styles.headerTitle}>{title}</Text>
          {subtitle && <Text style={styles.headerSubtitle}>{subtitle}</Text>}
        </View>
      </View>
      <Image
        source={require("../../assets/icon.png")}
        style={{ width: 60, height: 60, resizeMode: "contain" }}
      />
    </View>
  );
}
