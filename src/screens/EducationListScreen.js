import { Ionicons } from "@expo/vector-icons";
import {
  Image,
  Linking,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Card from "../components/Card";
import Header from "../components/Header";
import { ARTICLES } from "../data/articles";
import { CAREER_ARTICLES } from "../data/careerData";
import { styles } from "../theme/styles";

export default function EducationListScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Header title="Learn & Understand" back />
      <BackgroundBubbles />

      <ScrollView
        contentContainerStyle={{ padding: 20 }}
        showsVerticalScrollIndicator={false}
      >
        <View
          style={{
            backgroundColor: "#F3E5F5",
            borderRadius: 16,
            overflow: "hidden",
            marginBottom: 25,
            borderWidth: 1,
            borderColor: "#E1BEE7",
          }}
        >
          <Image
            source={require("../../assets/peri.png")}
            style={{ width: "100%", height: 180 }}
            resizeMode="cover"
          />
          <View style={{ padding: 20 }}>
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                marginBottom: 10,
              }}
            >
              <Ionicons name="business" size={20} color="#9C27B0" />
              <Text
                style={{
                  marginLeft: 10,
                  color: "#4A148C",
                  fontWeight: "bold",
                  fontSize: 16,
                  flex: 1,
                }}
              >
                About Philippine Eye Research Institute
              </Text>
            </View>
            <Text
              style={{
                color: "#4A148C",
                fontSize: 13,
                lineHeight: 20,
                marginBottom: 15,
              }}
            >
              The Philippine Eye Research Institute (PERI) is the premier eye
              research institution in the Philippines, dedicated to preventing
              blindness.
            </Text>
            <Text style={{ fontSize: 13, color: "#333", lineHeight: 20 }}>
              <Text style={{ fontWeight: "bold" }}>Our Mission:</Text> To
              improve eye health outcomes through innovative research and
              evidence-based clinical practices.
            </Text>
            <TouchableOpacity
              style={{
                marginTop: 20,
                backgroundColor: "#FFF",
                paddingVertical: 12,
                borderRadius: 8,
                alignItems: "center",
                flexDirection: "row",
                justifyContent: "center",
                borderWidth: 1,
                borderColor: "#D1C4E9",
              }}
              onPress={() => Linking.openURL("https://peri.ph/")}
            >
              <Text
                style={{ color: "#4A148C", fontWeight: "bold", marginRight: 8 }}
              >
                Visit PERI Website
              </Text>
              <Ionicons name="open-outline" size={16} color="#4A148C" />
            </TouchableOpacity>
          </View>
        </View>

        {/* ── CVD Simulations — Bento Grid ── */}
        <View style={{ marginBottom: 25 }}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              marginBottom: 12,
            }}
          >
            <Ionicons name="eye-outline" size={22} color="#333" />
            <Text
              style={{
                marginLeft: 8,
                fontWeight: "bold",
                fontSize: 16,
                color: "#333",
              }}
            >
              CVD Simulations
            </Text>
          </View>
          <Text style={{ color: "#666", fontSize: 12, marginBottom: 14 }}>
            Tap a type to experience how colour vision deficiency affects
            perception.
          </Text>

          <View style={{ flexDirection: "row", gap: 10, marginBottom: 10 }}>
            {[
              {
                label: "Protanomaly",
                sub: "Red-weak",
                cvdType: "Protan",
                bg: "#FFEBEE",
                border: "#FFCDD2",
                textColor: "#B71C1C",
                icon: "radio-button-on",
              },
              {
                label: "Deuteranomaly",
                sub: "Green-weak",
                cvdType: "Deutan",
                bg: "#E8F5E9",
                border: "#C8E6C9",
                textColor: "#1B5E20",
                icon: "radio-button-on",
              },
            ].map((item) => (
              <TouchableOpacity
                key={item.label}
                style={{
                  flex: 1,
                  backgroundColor: item.bg,
                  padding: 16,
                  borderRadius: 16,
                  borderWidth: 1,
                  borderColor: item.border,
                  minHeight: 110,
                }}
                onPress={() =>
                  navigation.navigate("CVDSimulation", {
                    initialCvdType: item.cvdType,
                  })
                }
                activeOpacity={0.82}
              >
                <View
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 18,
                    backgroundColor: item.textColor + "22",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: 10,
                  }}
                >
                  <Ionicons
                    name="eye-outline"
                    size={18}
                    color={item.textColor}
                  />
                </View>
                <Text
                  style={{
                    fontWeight: "800",
                    color: item.textColor,
                    fontSize: 14,
                  }}
                >
                  {item.label}
                </Text>
                <Text
                  style={{
                    color: item.textColor + "AA",
                    fontSize: 11,
                    marginTop: 3,
                  }}
                >
                  {item.sub}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <TouchableOpacity
            style={{
              backgroundColor: "#E3F2FD",
              padding: 16,
              borderRadius: 16,
              borderWidth: 1,
              borderColor: "#BBDEFB",
              flexDirection: "row",
              alignItems: "center",
              gap: 14,
            }}
            onPress={() =>
              navigation.navigate("CVDSimulation", { initialCvdType: "Tritan" })
            }
            activeOpacity={0.82}
          >
            <View
              style={{
                width: 44,
                height: 44,
                borderRadius: 22,
                backgroundColor: "#0D47A122",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Ionicons name="eye-outline" size={22} color="#0D47A1" />
            </View>
            <View style={{ flex: 1 }}>
              <Text
                style={{ fontWeight: "800", color: "#0D47A1", fontSize: 15 }}
              >
                Tritanomaly
              </Text>
              <Text style={{ color: "#0D47A1AA", fontSize: 12, marginTop: 2 }}>
                Blue-weak vision · Simulate
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#0D47A1" />
          </TouchableOpacity>
        </View>

        {/* ── Career Awareness Section ── */}
        <View style={{ marginBottom: 25 }}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              marginBottom: 12,
            }}
          >
            <Ionicons name="briefcase-outline" size={22} color="#333" />
            <Text
              style={{
                marginLeft: 8,
                fontWeight: "bold",
                fontSize: 16,
                color: "#333",
              }}
            >
              Career Awareness
            </Text>
          </View>
          <TouchableOpacity
            style={{
              backgroundColor: "#E8EAF6",
              padding: 20,
              borderRadius: 16,
              borderWidth: 1,
              borderColor: "#C5CAE9",
              flexDirection: "row",
              alignItems: "center",
            }}
            activeOpacity={0.85}
            onPress={() =>
              navigation.navigate("CareerDetail", { item: CAREER_ARTICLES[0] })
            }
          >
            <View
              style={{
                width: 48,
                height: 48,
                borderRadius: 12,
                backgroundColor: "#3F51B522",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Ionicons name="school-outline" size={26} color="#3F51B5" />
            </View>
            <View style={{ flex: 1, marginLeft: 15 }}>
              <Text
                style={{ fontWeight: "bold", color: "#1A237E", fontSize: 16 }}
              >
                Occupations & CVD
              </Text>
              <Text style={{ color: "#1A237E99", fontSize: 12, marginTop: 2 }}>
                Explore 9 critical career paths affected by color vision.
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#3F51B5" />
          </TouchableOpacity>
        </View>

        <Text
          style={{
            fontWeight: "bold",
            fontSize: 18,
            marginBottom: 15,
            color: "#333",
          }}
        >
          Latest Articles
        </Text>
        {ARTICLES.map((article) => (
          <Card
            key={article.id}
            style={{ padding: 0, overflow: "hidden", marginBottom: 20 }}
            onPress={() => navigation.navigate("Article", { article })}
          >
            <Image
              source={article.coverImage}
              style={{ width: "100%", height: 140 }}
              resizeMode="cover"
            />
            <View style={{ padding: 20 }}>
              <Text
                style={{
                  fontSize: 16,
                  fontWeight: "bold",
                  marginBottom: 5,
                  color: "#333",
                }}
              >
                {article.title}
              </Text>
              <Text style={{ fontSize: 12, color: "#666", lineHeight: 18 }}>
                {article.summary}
              </Text>
            </View>
          </Card>
        ))}
      </ScrollView>
    </View>
  );
}
