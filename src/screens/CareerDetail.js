import { Ionicons } from "@expo/vector-icons";
import {
  ScrollView,
  StyleSheet,
  Text,
  View
} from "react-native";
import Animated, { FadeInDown, FadeInUp } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import Header from "../components/Header";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

export default function CareerDetail({ route, navigation }) {
  const { item } = route.params;
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.safeArea}>
      <Header title="Career Awareness" back />

      {/* Integrated Utility Banner */}
      <View style={styles.disclaimer}>
        <Ionicons name="shield-checkmark" size={12} color="#B45309" />
        <Text style={styles.disclaimerText}>
          SCREENING PURPOSE ONLY • NOT A MEDICAL DIAGNOSIS
        </Text>
      </View>

      <ScrollView
        style={styles.container}
        contentContainerStyle={{ paddingBottom: insets.bottom + 40 }}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View entering={FadeInUp.duration(600)} style={styles.content}>
          {/* Header Section */}
          <View style={styles.badgeRow}>
            <View
              style={[
                styles.tagBadge,
                { backgroundColor: COLORS.primary + "10" },
              ]}
            >
              <Text style={[styles.tagText, { color: COLORS.primary }]}>
                {item.tag}
              </Text>
            </View>
            <Text style={styles.readTime}>{item.readTime}</Text>
          </View>

          <Text style={styles.title}>{item.title}</Text>
          <Text style={styles.introText}>{item.intro}</Text>

          {/* OVERVIEW MODULES */}
          {item.type === "overview" &&
            item.cards.map((c, i) => (
              <Animated.View
                key={i}
                entering={FadeInDown.delay(i * 100)}
                style={[
                  styles.overviewCard,
                  { borderLeftColor: c.color || COLORS.primary },
                ]}
              >
                <View style={styles.cardIconBox}>
                  <Ionicons name={c.icon} size={22} color={COLORS.text} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.cardTitle}>{c.label}</Text>
                  <Text style={styles.cardDesc}>{c.desc}</Text>
                </View>
              </Animated.View>
            ))}

          {/* STATS MODULE */}
          {item.stats && (
            <View style={styles.statsContainer}>
              <Text style={styles.sectionLabel}>INDUSTRY DATA</Text>
              <View style={styles.statsRow}>
                {item.stats.map((s, i) => (
                  <View key={i} style={styles.statCard}>
                    <Text
                      style={[
                        styles.statValue,
                        { color: s.textColor || COLORS.primary },
                      ]}
                    >
                      {s.value}
                    </Text>
                    <Text style={styles.statLabel}>{s.label}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          {/* DYNAMIC CONTENT SECTIONS (Color boxes / Lists) */}
          {item.sections &&
            item.sections.map((section, idx) =>
              section.type === "list" ? (
                <View key={idx} style={styles.sectionBlock}>
                  <Text style={styles.sectionLabel}>
                    {section.title.toUpperCase()}
                  </Text>
                  <View style={styles.listModule}>
                    {section.items.map((li, liIdx) => (
                      <View
                        key={liIdx}
                        style={[
                          styles.listItem,
                          liIdx === section.items.length - 1 && {
                            borderBottomWidth: 0,
                          },
                        ]}
                      >
                        <View
                          style={[styles.colorDot, { backgroundColor: li.hex }]}
                        />
                        <View style={{ flex: 1 }}>
                          <Text style={styles.listLabel}>{li.label}</Text>
                          <Text style={styles.listDesc}>{li.desc}</Text>
                        </View>
                      </View>
                    ))}
                  </View>
                </View>
              ) : (
                <View key={idx} style={styles.sectionBlock}>
                  <Text style={styles.sectionLabel}>VISUAL REQUIREMENTS</Text>
                  <View
                    style={[
                      styles.colorCard,
                      { backgroundColor: section.bgColor || "#FFF" },
                    ]}
                  >
                    <Text style={styles.cardTitle}>{section.title}</Text>
                    <Text style={styles.cardSubtitle}>{section.subtitle}</Text>
                    <View style={styles.boxRow}>
                      {section.colors.map((c, ci) => (
                        <View
                          key={ci}
                          style={[styles.colorBox, { backgroundColor: c }]}
                        />
                      ))}
                    </View>
                    <View style={styles.cardFooterRow}>
                      <Ionicons
                        name="information-circle-outline"
                        size={12}
                        color={COLORS.textLight}
                      />
                      <Text style={styles.cardFooter}>{section.footer}</Text>
                    </View>
                  </View>
                </View>
              ),
            )}

          {/* KEY FACTS MODULE */}
          {item.facts && (
            <View style={styles.sectionBlock}>
              <Text style={styles.sectionLabel}>CRITICAL OBSERVATIONS</Text>
              <View style={styles.factsModule}>
                {item.facts.map((f, i) => (
                  <View key={i} style={styles.factItem}>
                    <Ionicons
                      name="checkmark-circle"
                      size={18}
                      color={COLORS.success}
                    />
                    <Text style={styles.factText}>{f}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          {/* ROLES CHIPS */}
          {item.roles && item.roles.length > 0 && (
            <View style={styles.sectionBlock}>
              <Text style={styles.sectionLabel}>CVD-FRIENDLY ROLES</Text>
              <View style={styles.rolesGrid}>
                {item.roles.map((role, rIdx) => (
                  <View key={rIdx} style={styles.roleChip}>
                    <Text style={styles.roleText}>{role}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#F8FAFC" },
  container: { flex: 1 },
  disclaimer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: "#FFF8E1",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#FFE082",
  },
  disclaimerText: {
    fontSize: 9,
    fontWeight: "900",
    color: "#B45309",
    letterSpacing: 1,
  },
  content: { padding: SPACING.lg },
  badgeRow: { flexDirection: "row", alignItems: "center", marginBottom: 12 },
  tagBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
    marginRight: 10,
  },
  tagText: { fontSize: 10, fontWeight: "900", letterSpacing: 0.5 },
  readTime: { color: COLORS.textLight, fontSize: 12, fontWeight: "600" },
  title: {
    fontSize: 28,
    fontWeight: "900",
    color: COLORS.text,
    marginBottom: 12,
    letterSpacing: -0.5,
  },
  introText: {
    fontSize: 15,
    color: COLORS.textLight,
    lineHeight: 24,
    marginBottom: 25,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: 12,
    marginLeft: 4,
  },
  overviewCard: {
    flexDirection: "row",
    padding: 16,
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    marginBottom: 12,
    alignItems: "center",
    ...SHADOW.sm,
    borderLeftWidth: 4,
  },
  cardIconBox: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: "#F1F5F9",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 15,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: COLORS.text,
    marginBottom: 2,
  },
  cardDesc: { fontSize: 13, color: COLORS.textLight, lineHeight: 18 },
  statsContainer: { marginBottom: 25 },
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
  },
  statCard: {
    flex: 1,
    padding: 16,
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    alignItems: "center",
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  statValue: { fontSize: 20, fontWeight: "900" },
  statLabel: {
    fontSize: 10,
    color: COLORS.textLight,
    fontWeight: "700",
    marginTop: 4,
    textAlign: "center",
  },
  sectionBlock: { marginBottom: 30 },
  listModule: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    padding: 16,
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  listItem: {
    flexDirection: "row",
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#F1F5F9",
  },
  colorDot: {
    width: 36,
    height: 36,
    borderRadius: 18,
    marginRight: 15,
    ...SHADOW.sm,
  },
  listLabel: { fontSize: 15, fontWeight: "800", color: COLORS.text },
  colorCard: {
    padding: 20,
    borderRadius: RADIUS.xl,
    ...SHADOW.md,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  cardSubtitle: {
    fontSize: 13,
    color: COLORS.textLight,
    marginBottom: 15,
    lineHeight: 18,
  },
  boxRow: { flexDirection: "row", marginBottom: 15, gap: 8 },
  colorBox: { width: 45, height: 45, borderRadius: RADIUS.md, ...SHADOW.sm },
  cardFooterRow: { flexDirection: "row", alignItems: "center", gap: 6 },
  cardFooter: { fontSize: 11, color: COLORS.textLight, fontStyle: "italic" },
  factsModule: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    padding: 16,
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  factItem: {
    flexDirection: "row",
    marginBottom: 12,
    alignItems: "flex-start",
    gap: 12,
  },
  factText: {
    fontSize: 14,
    color: COLORS.text,
    lineHeight: 20,
    flex: 1,
    fontWeight: "600",
  },
  rolesGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  roleChip: {
    backgroundColor: "#FFF",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: "#E2E8F0",
    ...SHADOW.xs,
  },
  roleText: { fontSize: 13, color: COLORS.text, fontWeight: "700" },
});
