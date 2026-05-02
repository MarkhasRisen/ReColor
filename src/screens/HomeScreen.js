import { Ionicons } from "@expo/vector-icons";
import { MotiView } from "moti";
import {
  Dimensions,
  Image,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { auth } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import { COLORS, SHADOW } from "../theme/colors";
import { styles } from "../theme/styles";

const { width } = Dimensions.get("window");

// --- DOMINANT CARD COMPONENT ---
// Uses pulsing shadow and scale to feel "alive"
const DominantCard = ({
  bg,
  onPress,
  icon,
  iconColor,
  title,
  desc,
  delay = 0,
}) => (
  <MotiView
    from={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ type: "timing", duration: 800, delay }}
  >
    <TouchableOpacity activeOpacity={0.9} onPress={onPress}>
      <MotiView
        animate={{
          scale: [1, 1.02, 1],
          shadowOpacity: [0.1, 0.3, 0.1],
        }}
        transition={{
          loop: true,
          duration: 3000,
          type: "timing",
        }}
        style={{
          backgroundColor: bg,
          borderRadius: 28,
          padding: 24,
          marginBottom: 16,
          flexDirection: "row",
          alignItems: "center",
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.6)",
          ...SHADOW.lg,
        }}
      >
        <View style={{ flex: 1 }}>
          <Text
            style={{
              fontSize: 11,
              fontWeight: "800",
              color: iconColor,
              letterSpacing: 1.5,
              marginBottom: 4,
            }}
          >
            RECOMMENDED
          </Text>
          <Text
            style={{
              fontSize: 24,
              fontWeight: "900",
              color: "#1A1A2E",
              marginBottom: 6,
            }}
          >
            {title}
          </Text>
          <Text
            style={{
              fontSize: 13,
              color: "#4A4A4A",
              lineHeight: 18,
              paddingRight: 20,
            }}
          >
            {desc}
          </Text>
        </View>

        <MotiView
          animate={{ translateY: [-5, 5, -5] }}
          transition={{ loop: true, duration: 2500, type: "timing" }}
          style={{
            width: 70,
            height: 70,
            borderRadius: 35,
            backgroundColor: "rgba(255,255,255,0.5)",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Ionicons name={icon} size={36} color={iconColor} />
        </MotiView>
      </MotiView>
    </TouchableOpacity>
  </MotiView>
);

// --- SECONDARY SMALL CARD ---
const SecondaryCard = ({ bg, onPress, icon, iconColor, title, delay = 0 }) => (
  <MotiView
    from={{ opacity: 0, scale: 0.5 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ type: "spring", delay }}
    style={{ flex: 1 }}
  >
    <TouchableOpacity
      onPress={onPress}
      style={{
        backgroundColor: bg,
        borderRadius: 20,
        padding: 16,
        alignItems: "center",
        justifyContent: "center",
        height: 100,
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.3)",
      }}
    >
      <Ionicons
        name={icon}
        size={24}
        color={iconColor}
        style={{ marginBottom: 8 }}
      />
      <Text style={{ fontSize: 12, fontWeight: "700", color: "#1A1A2E" }}>
        {title}
      </Text>
    </TouchableOpacity>
  </MotiView>
);

export default function HomeScreen({ navigation }) {
  const userName = (auth.currentUser?.email || "Guest").split("@")[0];
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.container}>
      <BackgroundBubbles />

      {/* Sticky disclaimer — notch-aware */}
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: 6,
          backgroundColor: "#FFF8E1",
          paddingTop: insets.top + 4,
          paddingBottom: 6,
          borderBottomWidth: 1,
          borderBottomColor: "#FFE082",
        }}
      >
        <Ionicons name="warning-outline" size={12} color="#F59E0B" />
        <Text style={{ fontSize: 10, fontWeight: "700", color: "#F59E0B" }}>
          NOT A MEDICAL DIAGNOSIS — SCREENING ONLY
        </Text>
      </View>

      <ScrollView
        contentContainerStyle={{ padding: 20, paddingBottom: 40 }}
        showsVerticalScrollIndicator={false}
      >
        {/* Welcome Section */}
        <View
          style={{
            flexDirection: "row",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 30,
            marginTop: 10,
          }}
        >
          <View>
            <Text
              style={{
                color: COLORS.textLight,
                fontSize: 14,
                fontWeight: "600",
              }}
            >
              Hello, {userName}
            </Text>
            <Text
              style={{ fontSize: 32, fontWeight: "900", color: COLORS.text }}
            >
              ReColor
            </Text>
          </View>
          <MotiView
            from={{ rotate: "0deg" }}
            animate={{ rotate: "10deg" }}
            transition={{ loop: true, repeatReverse: true, duration: 2000 }}
          >
            <Image
              source={require("../../assets/icon.png")}
              style={{ width: 60, height: 60, resizeMode: "contain" }}
            />
          </MotiView>
        </View>

        {/* DOMINANT CARD 1: VISION ASSESSMENT */}
        <DominantCard
          bg="#E3F2FD" // Light Blue
          icon="eye"
          iconColor="#2196F3"
          title="Vision Assessment"
          desc="Standardized Ishihara screening to identify your specific color sensitivity."
          onPress={() => navigation.navigate("IshiharaIntro")}
          delay={200}
        />

        {/* DOMINANT CARD 2: VISION ASSISTANT */}
        <DominantCard
          bg="#F3E5F5" // Light Purple
          icon="camera"
          iconColor="#9C27B0"
          title="Camera and ColorBlind Simulation"
          desc="Point your camera to apply adaptive correction and identify colors in real-time."
          onPress={() => navigation.navigate("CameraEnhance")}
          delay={400}
        />

        {/* SECONDARY ROW */}
        <Text
          style={{
            fontSize: 12,
            fontWeight: "800",
            color: "#999",
            marginBottom: 12,
            letterSpacing: 1,
          }}
        >
          TOOLS & RESEARCH
        </Text>
        <View style={{ flexDirection: "row", gap: 12 }}>
          <SecondaryCard
            bg="#FCE4EC"
            title="Education"
            icon="book"
            iconColor="#E91E63"
            onPress={() => navigation.navigate("EducationList")}
            delay={600}
          />
          <SecondaryCard
            bg="#E8F5E9"
            title="Survey"
            icon="clipboard"
            iconColor="#4CAF50"
            onPress={() => navigation.navigate("Survey")}
            delay={700}
          />
        </View>
      </ScrollView>
    </View>
  );
}
