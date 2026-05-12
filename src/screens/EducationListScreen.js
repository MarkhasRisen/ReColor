import { Ionicons } from "@expo/vector-icons";
import {
  Image,
  Linking,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, { FadeInDown, FadeInUp } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Header from "../components/Header";
import { ARTICLES } from "../data/articles";
import { CAREER_ARTICLES } from "../data/careerData";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

export default function EducationListScreen({ navigation }) {
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      <Header title="Learn & Understand" back />

      <ScrollView
        contentContainerStyle={[
          styles.scrollBody,
          { paddingBottom: insets.bottom + SPACING.xl },
        ]}
        showsVerticalScrollIndicator={false}
      >
        {/* --- PERI MODULE (RESTORED RIBBON & PARTNER STATUS) --- */}
        <Animated.View
          entering={FadeInUp.duration(600)}
          style={styles.periCard}
        >
          <Image
            source={require("../../assets/peri.png")}
            style={styles.periImage}
            resizeMode="cover"
          />
          <View style={styles.periContent}>
            <View style={styles.badgeRow}>
              <View style={styles.partnerBadge}>
                <Ionicons name="ribbon" size={14} color="#9C27B0" />
                <Text style={styles.partnerText}>RESEARCH PARTNER</Text>
              </View>
            </View>

            <Text style={styles.periTitle}>
              About Philippine Eye Research Institute
            </Text>

            <Text style={styles.periDesc}>
              The Philippine Eye Research Institute (PERI) is the premier eye
              research institution in the Philippines, dedicated to preventing
              blindness.
            </Text>

            <Text style={styles.periMission}>
              <Text style={{ fontWeight: "900" }}>Our Mission:</Text> To improve
              eye health outcomes through innovative research and evidence-based
              clinical practices.
            </Text>

            <TouchableOpacity
              style={styles.periButton}
              onPress={() => Linking.openURL("https://peri.ph/")}
              activeOpacity={0.8}
            >
              <Text style={styles.periButtonText}>Visit PERI Website</Text>
              <Ionicons name="open-outline" size={16} color="#9C27B0" />
            </TouchableOpacity>
          </View>
        </Animated.View>

        {/* --- INTERACTIVE: CVD SIMULATIONS --- */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>CVD SIMULATIONS</Text>
          <Text style={styles.sectionSub}>
            Tap a type to experience how colour vision deficiency affects
            perception.
          </Text>

          <View style={styles.gridRow}>
            {[
              {
                label: "Protanomaly",
                sub: "Red-weak",
                type: "Protan",
                color: "#EF4444",
              },
              {
                label: "Deuteranomaly",
                sub: "Green-weak",
                type: "Deutan",
                color: "#22C55E",
              },
            ].map((item, idx) => (
              <TouchableOpacity
                key={idx}
                style={styles.simCard}
                onPress={() =>
                  navigation.navigate("CVDSimulation", {
                    initialCvdType: item.type,
                  })
                }
              >
                <View
                  style={[
                    styles.simIcon,
                    { backgroundColor: item.color + "15" },
                  ]}
                >
                  <Ionicons name="eye-outline" size={20} color={item.color} />
                </View>
                <Text style={styles.simTitle}>{item.label}</Text>
                <Text style={styles.simDesc}>{item.sub}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <TouchableOpacity
            style={styles.tritanCard}
            onPress={() =>
              navigation.navigate("CVDSimulation", { initialCvdType: "Tritan" })
            }
          >
            <View style={styles.tritanIcon}>
              <Ionicons name="eye-outline" size={24} color="#0D47A1" />
            </View>
            <View style={styles.tritanTextGroup}>
              <Text style={styles.tritanTitle}>Tritanomaly</Text>
              <Text style={styles.tritanDesc}>Blue-weak vision · Simulate</Text>
            </View>
            <Ionicons name="chevron-forward" size={18} color="#0D47A1" />
          </TouchableOpacity>
        </View>

        {/* --- CAREER AWARENESS --- */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>CAREER AWARENESS</Text>
          <TouchableOpacity
            style={styles.careerModule}
            activeOpacity={0.9}
            onPress={() =>
              navigation.navigate("CareerDetail", { item: CAREER_ARTICLES[0] })
            }
          >
            <View style={styles.careerIconBox}>
              <Ionicons name="school-outline" size={24} color="#3F51B5" />
            </View>
            <View style={styles.careerText}>
              <Text style={styles.careerTitle}>Occupations & CVD</Text>
              <Text style={styles.careerDesc}>
                Explore 9 critical career paths affected by color vision.
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={18} color="#3F51B5" />
          </TouchableOpacity>
        </View>

        {/* --- LATEST ARTICLES --- */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>LATEST ARTICLES</Text>
          {ARTICLES.map((article, index) => (
            <Animated.View
              key={article.id}
              entering={FadeInDown.delay(index * 100)}
            >
              <TouchableOpacity
                style={styles.articleItem}
                onPress={() => navigation.navigate("Article", { article })}
              >
                <Image
                  source={article.coverImage}
                  style={styles.articleThumb}
                />
                <View style={styles.articleBody}>
                  <Text style={styles.articleTitle} numberOfLines={1}>
                    {article.title}
                  </Text>
                  <Text style={styles.articleSummary} numberOfLines={2}>
                    {article.summary}
                  </Text>
                </View>
                <Ionicons
                  name="chevron-forward"
                  size={16}
                  color={COLORS.textLight}
                />
              </TouchableOpacity>
            </Animated.View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8FAFC",
  },
  scrollBody: {
    padding: SPACING.lg,
  },
  section: {
    marginBottom: 32,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: 12,
    marginLeft: 4,
  },
  sectionSub: {
    fontSize: 12,
    color: COLORS.textLight,
    marginBottom: 16,
    marginLeft: 4,
    lineHeight: 18,
  },
  periCard: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    overflow: "hidden",
    ...SHADOW.md,
    marginBottom: 32,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  periImage: {
    width: "100%",
    height: 180,
  },
  periContent: {
    padding: SPACING.lg,
  },
  badgeRow: {
    flexDirection: "row",
    marginBottom: 12,
  },
  partnerBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#F3E5F5",
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
    gap: 6,
  },
  partnerText: {
    fontSize: 10,
    fontWeight: "900",
    color: "#9C27B0",
    letterSpacing: 0.5,
  },
  periTitle: {
    fontSize: 18,
    fontWeight: "900",
    color: COLORS.text,
    marginBottom: 10,
  },
  periDesc: {
    fontSize: 13,
    color: "#4A148C",
    lineHeight: 20,
    marginBottom: 15,
  },
  periMission: {
    fontSize: 13,
    color: COLORS.text,
    lineHeight: 20,
  },
  periButton: {
    backgroundColor: "#FFF",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    borderRadius: RADIUS.lg,
    marginTop: 20,
    gap: 8,
    borderWidth: 1,
    borderColor: "#D1C4E9",
  },
  periButtonText: {
    color: "#4A148C",
    fontWeight: "800",
    fontSize: 14,
  },
  gridRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  simCard: {
    width: "48%",
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    padding: 16,
    borderWidth: 1,
    borderColor: "#F1F5F9",
    ...SHADOW.sm,
  },
  simIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 12,
  },
  simTitle: {
    fontSize: 14,
    fontWeight: "800",
    color: COLORS.text,
  },
  simDesc: {
    fontSize: 11,
    color: COLORS.textLight,
    marginTop: 2,
  },
  tritanCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#E3F2FD",
    borderRadius: RADIUS.xl,
    padding: 16,
    borderWidth: 1,
    borderColor: "#BBDEFB",
  },
  tritanIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(13, 71, 161, 0.1)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 14,
  },
  tritanTextGroup: {
    flex: 1,
  },
  tritanTitle: {
    fontSize: 15,
    fontWeight: "800",
    color: "#0D47A1",
  },
  tritanDesc: {
    fontSize: 12,
    color: "rgba(13, 71, 161, 0.7)",
    marginTop: 2,
  },
  careerModule: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#E8EAF6",
    padding: 20,
    borderRadius: RADIUS.xl,
    borderWidth: 1,
    borderColor: "#C5CAE9",
  },
  careerIconBox: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: "rgba(63, 81, 181, 0.1)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 15,
  },
  careerText: {
    flex: 1,
  },
  careerTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#1A237E",
  },
  careerDesc: {
    fontSize: 12,
    color: "rgba(26, 35, 126, 0.6)",
    marginTop: 2,
  },
  articleItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFF",
    padding: 14,
    borderRadius: RADIUS.xl,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: "#F1F5F9",
    ...SHADOW.sm,
  },
  articleThumb: {
    width: 60,
    height: 60,
    borderRadius: RADIUS.lg,
    backgroundColor: "#F1F5F9",
  },
  articleBody: {
    flex: 1,
    paddingHorizontal: 15,
  },
  articleTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: COLORS.text,
  },
  articleSummary: {
    fontSize: 12,
    color: COLORS.textLight,
    lineHeight: 18,
    marginTop: 4,
  },
});
