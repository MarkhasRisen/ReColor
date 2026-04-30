import { Ionicons } from "@expo/vector-icons";
import {
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

export default function CareerDetail({ route, navigation }) {
  const { item } = route.params;
  const insets = useSafeAreaInsets();

  return (
    <View style={[styles.safeArea, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Learn & Understand</Text>
        <Image source={require("../../assets/icon.png")} style={styles.logo} />
      </View>

      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        <View style={styles.content}>
          <View style={styles.badgeRow}>
            <View style={styles.tagBadge}>
              <Text style={styles.tagText}>{item.tag}</Text>
            </View>
            <Text style={styles.readTime}>{item.readTime}</Text>
          </View>
          <Text style={styles.title}>{item.title}</Text>
          <Text style={styles.introText}>{item.intro}</Text>

          {/* OVERVIEW SPECIFIC CARDS (Screen 1 Logic) */}
          {item.type === "overview" &&
            item.cards.map((c, i) => (
              <View
                key={i}
                style={[styles.overviewCard, { backgroundColor: c.color }]}
              >
                <Ionicons name={c.icon} size={24} color="#555" />
                <View style={{ marginLeft: 15, flex: 1 }}>
                  <Text style={styles.cardTitle}>{c.label}</Text>
                  <Text style={styles.listDesc}>{c.desc}</Text>
                </View>
              </View>
            ))}

          {/* STATS ROW (Screen 2, 6, 7, 9 Logic) */}
          {item.stats && (
            <View style={styles.statsRow}>
              {item.stats.map((s, i) => (
                <View
                  key={i}
                  style={[styles.statCard, { backgroundColor: s.color }]}
                >
                  <Text style={[styles.statValue, { color: s.textColor }]}>
                    {s.value}
                  </Text>
                  <Text style={styles.statLabel}>{s.label}</Text>
                </View>
              ))}
            </View>
          )}

          {/* COLOR BOXES (Screen 3, 4, 8 Logic) */}
          {item.sections &&
            item.sections.map((section, idx) =>
              section.type === "list" ? (
                <View key={idx} style={styles.sectionBlock}>
                  <Text style={styles.sectionTitle}>{section.title}</Text>
                  {section.items.map((li, liIdx) => (
                    <View key={liIdx} style={styles.listItem}>
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
              ) : (
                <View
                  key={idx}
                  style={[
                    styles.colorCard,
                    { backgroundColor: section.bgColor },
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
                  <Text style={styles.cardFooter}>{section.footer}</Text>
                </View>
              ),
            )}

          {/* KEY FACTS (Screen 1, 5 Logic) */}
          {item.facts && (
            <View style={styles.sectionBlock}>
              <Text style={styles.sectionTitle}>Key Facts</Text>
              {item.facts.map((f, i) => (
                <View key={i} style={styles.factItem}>
                  <Ionicons
                    name="checkmark-circle-outline"
                    size={18}
                    color="#059669"
                  />
                  <Text style={styles.factText}>{f}</Text>
                </View>
              ))}
            </View>
          )}

          {/* ROLES CHIPS */}
          {item.roles.length > 0 && (
            <Text style={styles.sectionTitle}>
              CVD-Friendly {item.tag} Roles
            </Text>
          )}
          <View style={styles.rolesGrid}>
            {item.roles.map((role, rIdx) => (
              <View key={rIdx} style={styles.roleChip}>
                <Text style={styles.roleText}>{role}</Text>
              </View>
            ))}
          </View>
        </View>
        <View style={{ height: 100 }} />
      </ScrollView>

      <View style={styles.warningBar}>
        <Ionicons name="warning-outline" size={14} color="#856404" />
        <Text style={styles.warningText}>
          SCREENING PURPOSE ONLY. NOT A CLINICAL DIAGNOSIS.
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#FFF" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#F3F4F6",
  },
  headerTitle: { fontSize: 16, fontWeight: "600" },
  logo: { width: 60, height: 20, resizeMode: "contain" },
  container: { flex: 1 },
  content: { padding: 20 },
  badgeRow: { flexDirection: "row", alignItems: "center", marginBottom: 10 },
  tagBadge: {
    backgroundColor: "#D1FAE5",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
    marginRight: 8,
  },
  tagText: { color: "#059669", fontSize: 11, fontWeight: "bold" },
  readTime: { color: "#6B7280", fontSize: 11 },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#111827",
    marginBottom: 10,
  },
  introText: {
    fontSize: 15,
    color: "#4B5563",
    lineHeight: 22,
    marginBottom: 20,
  },
  overviewCard: {
    flexDirection: "row",
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    alignItems: "center",
  },
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 25,
  },
  statCard: {
    flex: 1,
    margin: 4,
    padding: 12,
    borderRadius: 12,
    alignItems: "center",
  },
  statValue: { fontSize: 18, fontWeight: "bold" },
  statLabel: {
    fontSize: 10,
    color: "#6B7280",
    marginTop: 2,
    textAlign: "center",
  },
  sectionBlock: { marginTop: 15, marginBottom: 20 },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#111827",
    marginBottom: 15,
  },
  listItem: { flexDirection: "row", marginBottom: 15 },
  colorDot: { width: 40, height: 40, borderRadius: 20, marginRight: 15 },
  listLabel: { fontSize: 15, fontWeight: "600", color: "#111827" },
  listDesc: { fontSize: 13, color: "#6B7280" },
  colorCard: { padding: 20, borderRadius: 16, marginBottom: 15 },
  cardTitle: { fontSize: 16, fontWeight: "bold" },
  cardSubtitle: { fontSize: 12, color: "#6B7280", marginBottom: 12 },
  boxRow: { flexDirection: "row", marginBottom: 12 },
  colorBox: { width: 45, height: 45, borderRadius: 8, marginRight: 8 },
  cardFooter: { fontSize: 11, color: "#4B5563", fontStyle: "italic" },
  factItem: { flexDirection: "row", marginBottom: 10, alignItems: "center" },
  factText: { fontSize: 14, color: "#4B5563", marginLeft: 10 },
  rolesGrid: { flexDirection: "row", flexWrap: "wrap" },
  roleChip: {
    backgroundColor: "#F3F4F6",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
  },
  roleText: { fontSize: 13, color: "#4B5563" },
  warningBar: {
    position: "absolute",
    bottom: 0,
    width: "100%",
    backgroundColor: "#FBBF24",
    padding: 12,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
  },
  warningText: {
    fontSize: 10,
    fontWeight: "bold",
    color: "#856404",
    marginLeft: 8,
  },
});
