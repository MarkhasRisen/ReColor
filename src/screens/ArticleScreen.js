import { Ionicons } from "@expo/vector-icons";
import { MotiView } from "moti";
import React from "react";
import {
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, { FadeInUp } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import Header from "../components/Header";
import BackgroundBubbles from "../components/BackgroundBubbles";
import { ARTICLES } from "../data/articles";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

// --- SECTION RENDERER: PRESERVING ALL YOUR CUSTOM LOGIC ---

function SectionRenderer({ section }) {
  switch (section.type) {
    case "heading":
      return <Text style={styles.sectionHeading}>{section.text}</Text>;

    case "paragraph":
      return <Text style={styles.paragraph}>{section.text}</Text>;

    case "dos_donts":
      return (
        <View style={styles.dosDontsRow}>
          <View
            style={[styles.dosDontsCard, { borderTopColor: COLORS.success }]}
          >
            <View style={styles.dosDontsHeader}>
              <Ionicons
                name="checkmark-circle"
                size={16}
                color={COLORS.success}
              />
              <Text style={[styles.dosDontsTitle, { color: COLORS.success }]}>
                Do
              </Text>
            </View>
            {section.dos.map((d, i) => (
              <Text key={i} style={styles.doItem}>
                • {d}
              </Text>
            ))}
          </View>
          <View
            style={[styles.dosDontsCard, { borderTopColor: COLORS.danger }]}
          >
            <View style={styles.dosDontsHeader}>
              <Ionicons name="close-circle" size={16} color={COLORS.danger} />
              <Text style={[styles.dosDontsTitle, { color: COLORS.danger }]}>
                {"Don't"}
              </Text>
            </View>
            {section.donts.map((d, i) => (
              <Text key={i} style={styles.dontItem}>
                • {d}
              </Text>
            ))}
          </View>
        </View>
      );

    case "contrast":
      return (
        <View style={styles.contrastList}>
          {section.items.map((item, i) => (
            <View key={i} style={styles.contrastRow}>
              <View
                style={[
                  styles.contrastBadge,
                  {
                    backgroundColor: item.pass ? COLORS.success : COLORS.danger,
                  },
                ]}
              >
                <Text style={styles.contrastRatio}>{item.ratio}</Text>
              </View>
              <Text style={styles.contrastLabel}>{item.label}</Text>
              <Ionicons
                name={item.pass ? "checkmark-circle" : "close-circle"}
                size={18}
                color={item.pass ? COLORS.success : COLORS.danger}
              />
            </View>
          ))}
        </View>
      );

    case "palettes":
      return (
        <View style={styles.palettesList}>
          {section.items.map((pal, i) => (
            <View key={i} style={styles.paletteRow}>
              <View style={styles.swatchRow}>
                {pal.colors.map((c, ci) => (
                  <View
                    key={ci}
                    style={[styles.swatch, { backgroundColor: c }]}
                  />
                ))}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.paletteName}>{pal.name}</Text>
                <Text style={styles.paletteSafe}>{pal.safe}</Text>
              </View>
            </View>
          ))}
        </View>
      );

    case "tools":
      return (
        <View style={styles.toolsRow}>
          {section.items.map((tool, i) => (
            <View key={i} style={styles.toolChip}>
              <Text style={styles.toolText}>{tool}</Text>
            </View>
          ))}
        </View>
      );

    case "severity_spectrum":
      return (
        <View style={styles.spectrumWrap}>
          <View style={styles.spectrumBar} />
          <View style={styles.spectrumLabels}>
            <Text style={styles.spectrumLabel}>Mild</Text>
            <Text style={styles.spectrumLabel}>Moderate</Text>
            <Text style={styles.spectrumLabel}>Severe</Text>
          </View>
        </View>
      );

    case "cvd_types":
      return (
        <View style={styles.cvdTypesList}>
          {section.items.map((cvd, i) => (
            <View key={i} style={styles.cvdTypeCard}>
              <View style={styles.cvdTypeHeader}>
                <View style={styles.cvdSwatches}>
                  {cvd.colors.map((c, ci) => (
                    <View
                      key={ci}
                      style={[styles.cvdSwatch, { backgroundColor: c }]}
                    />
                  ))}
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.cvdTypeName}>{cvd.name}</Text>
                  <Text style={styles.cvdTypeSub}>{cvd.sub}</Text>
                </View>
              </View>
              <Text style={styles.cvdTypeDesc}>{cvd.desc}</Text>
            </View>
          ))}
        </View>
      );

    case "stats":
      return (
        <View style={styles.statsRow}>
          {section.items.map((s, i) => (
            <View key={i} style={styles.statItem}>
              <Text style={styles.statValue}>{s.value}</Text>
              <Text style={styles.statLabel}>{s.label}</Text>
            </View>
          ))}
        </View>
      );

    case "cone_diagram":
      return (
        <View style={styles.coneRow}>
          {section.cones.map((cone, i) => (
            <View key={i} style={styles.coneItem}>
              <View
                style={[styles.coneCircle, { backgroundColor: cone.color }]}
              />
              <Text style={styles.coneLabel}>{cone.label}</Text>
              <Text style={styles.coneSub}>{cone.sub}</Text>
            </View>
          ))}
        </View>
      );

    case "causes":
      return (
        <View style={styles.causesRow}>
          {section.items.map((cause, i) => (
            <View key={i} style={styles.causeChip}>
              <Text style={styles.causeText}>{cause}</Text>
            </View>
          ))}
        </View>
      );

    case "bento_grid":
      return (
        <View style={styles.bentoGrid}>
          {section.items.map((item, i) => (
            <View key={i} style={styles.bentoCell}>
              <Ionicons
                name={item.icon}
                size={22}
                color={COLORS.primary}
                style={{ marginBottom: 6 }}
              />
              <Text style={styles.bentoCellTitle}>{item.title}</Text>
              <Text style={styles.bentoCellDesc}>{item.desc}</Text>
            </View>
          ))}
        </View>
      );

    case "quick_wins":
      return (
        <View style={styles.quickWinsList}>
          {section.items.map((item, i) => (
            <View key={i} style={styles.quickWinRow}>
              <Ionicons
                name="checkmark-circle"
                size={18}
                color={COLORS.success}
              />
              <Text style={styles.quickWinText}>{item}</Text>
            </View>
          ))}
        </View>
      );

    case "list":
      return (
        <View style={styles.articleList}>
          {section.items.map((item, i) => (
            <View key={i} style={styles.listItem}>
              <View style={styles.listDot} />
              <View style={{ flex: 1 }}>
                <Text style={styles.listItemTitle}>{item.title}</Text>
                <Text style={styles.listItemDesc}>{item.desc}</Text>
              </View>
            </View>
          ))}
        </View>
      );

    default:
      return null;
  }
}

// --- ARTICLE LIST VIEW ---

function ArticleCard({ article, onPress }) {
  return (
    <MotiView
      from={{ opacity: 0, translateY: 15 }}
      animate={{ opacity: 1, translateY: 0 }}
      transition={{ type: "spring", damping: 20 }}
    >
      <TouchableOpacity
        style={styles.articleCard}
        onPress={onPress}
        activeOpacity={0.9}
      >
        <Image source={article.coverImage} style={styles.articleCover} />
        <View style={styles.articleCardBody}>
          <View style={styles.metaRow}>
            <View style={styles.categoryBadge}>
              <Text style={styles.categoryText}>
                {article.category.toUpperCase()}
              </Text>
            </View>
            <Text style={styles.readTime}>{article.readTime}</Text>
          </View>
          <Text style={styles.articleTitle}>{article.title}</Text>
          <Text style={styles.articleSummary} numberOfLines={2}>
            {article.summary}
          </Text>
        </View>
      </TouchableOpacity>
    </MotiView>
  );
}

// --- ARTICLE DETAIL VIEW ---

function ArticleDetail({ article, onBack }) {
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.detailRoot}>
      <BackgroundBubbles />
      <ScrollView showsVerticalScrollIndicator={false}>
        <Image source={article.coverImage} style={styles.detailCover} />

        <Animated.View
          entering={FadeInUp.duration(600)}
          style={styles.detailCard}
        >
          <View style={styles.metaRow}>
            <Text style={styles.categoryTag}>{article.category}</Text>
            <Text style={styles.readTime}>{article.readTime}</Text>
          </View>
          <Text style={styles.detailTitle}>{article.title}</Text>
          <Text style={styles.detailSummary}>{article.summary}</Text>

          <View style={styles.divider} />

          {article.sections.map((sec, i) => (
            <SectionRenderer key={i} section={sec} />
          ))}

          <View style={styles.clinicalDisclaimer}>
            <Ionicons
              name="shield-checkmark"
              size={16}
              color={COLORS.warning}
            />
            <Text style={styles.disclaimerText}>
              SCREENING PURPOSE ONLY • NOT A MEDICAL DIAGNOSIS
            </Text>
          </View>
        </Animated.View>
        <View style={{ height: 100 }} />
      </ScrollView>

      <TouchableOpacity
        style={[styles.backFab, { top: insets.top + 10 }]}
        onPress={onBack}
      >
        <Ionicons name="arrow-back" size={22} color="#FFF" />
      </TouchableOpacity>
    </View>
  );
}

// --- MAIN SCREEN ---

export default function ArticleScreen({ navigation, route }) {
  const [selected, setSelected] = React.useState(
    route?.params?.article ?? null,
  );

  if (selected) {
    return (
      <ArticleDetail article={selected} onBack={() => setSelected(null)} />
    );
  }

  return (
    <View style={styles.root}>
      <BackgroundBubbles />
      <Header title="Clinical Literacy" back />

      <ScrollView
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.listLabel}>LATEST DIAGNOSTIC GUIDES</Text>
        {ARTICLES.map((article) => (
          <ArticleCard
            key={article.id}
            article={article}
            onPress={() => setSelected(article)}
          />
        ))}

        <View style={styles.footerInfo}>
          <Ionicons
            name="information-circle-outline"
            size={14}
            color={COLORS.textLight}
          />
          <Text style={styles.footerInfoText}>
            Medical content is for educational screening purposes.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#F8FAFC" },
  listContent: { padding: SPACING.lg },
  listLabel: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: 16,
    marginLeft: 4,
  },
  articleCard: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    overflow: "hidden",
    marginBottom: SPACING.lg,
    ...SHADOW.md,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  articleCover: { width: "100%", height: 160 },
  articleCardBody: { padding: SPACING.lg },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  categoryBadge: {
    backgroundColor: COLORS.primary + "10",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  categoryText: {
    fontSize: 9,
    fontWeight: "900",
    color: COLORS.primary,
    letterSpacing: 0.5,
  },
  readTime: { fontSize: 12, color: COLORS.textLight, fontWeight: "600" },
  articleTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: COLORS.text,
    marginBottom: 6,
  },
  articleSummary: { fontSize: 13, color: COLORS.textLight, lineHeight: 20 },

  // Detail view
  detailRoot: { flex: 1, backgroundColor: "#F8FAFC" },
  detailCover: { width: "100%", height: 300 },
  detailCard: {
    backgroundColor: "#FFFFFF",
    borderTopLeftRadius: RADIUS.xl,
    borderTopRightRadius: RADIUS.xl,
    marginTop: -40,
    padding: SPACING.lg,
    ...SHADOW.md,
  },
  categoryTag: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.primary,
    backgroundColor: COLORS.primary + "10",
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
  },
  detailTitle: {
    fontSize: 26,
    fontWeight: "900",
    color: COLORS.text,
    marginBottom: 12,
  },
  detailSummary: {
    fontSize: 15,
    color: COLORS.textLight,
    lineHeight: 24,
    marginBottom: 20,
  },
  divider: { height: 1, backgroundColor: "#E2E8F0", marginBottom: 25 },
  sectionHeading: {
    fontSize: 18,
    fontWeight: "900",
    color: COLORS.text,
    marginTop: 24,
    marginBottom: 12,
  },
  paragraph: {
    fontSize: 15,
    color: COLORS.text,
    lineHeight: 24,
    marginBottom: 18,
  },
  backFab: {
    position: "absolute",
    left: 20,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0,0,0,0.4)",
    alignItems: "center",
    justifyContent: "center",
  },

  // Reused Components Style
  dosDontsRow: { flexDirection: "row", gap: 12, marginBottom: 20 },
  dosDontsCard: {
    flex: 1,
    backgroundColor: "#F8FAFC",
    borderRadius: RADIUS.lg,
    padding: 12,
    borderTopWidth: 4,
    borderWidth: 1,
    borderColor: "#E2E8F0",
    ...SHADOW.sm,
  },
  dosDontsHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    marginBottom: 8,
  },
  dosDontsTitle: { fontSize: 13, fontWeight: "800" },
  doItem: { fontSize: 12, color: COLORS.text, lineHeight: 18, marginBottom: 4 },
  dontItem: {
    fontSize: 12,
    color: COLORS.text,
    lineHeight: 18,
    marginBottom: 4,
  },
  contrastList: { gap: 10, marginBottom: 20 },
  contrastRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    backgroundColor: "#FFF",
    padding: 12,
    borderRadius: RADIUS.lg,
    ...SHADOW.sm,
  },
  contrastBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6 },
  contrastRatio: { fontSize: 12, fontWeight: "800", color: "#FFF" },
  contrastLabel: {
    flex: 1,
    fontSize: 14,
    fontWeight: "600",
    color: COLORS.text,
  },
  palettesList: { gap: 12, marginBottom: 20 },
  paletteRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
    backgroundColor: "#FFF",
    padding: 12,
    borderRadius: RADIUS.xl,
    ...SHADOW.sm,
  },
  swatchRow: { flexDirection: "row", gap: 4 },
  swatch: { width: 32, height: 32, borderRadius: 16 },
  paletteName: { fontSize: 15, fontWeight: "800", color: COLORS.text },
  paletteSafe: { fontSize: 12, color: COLORS.textLight, marginTop: 2 },
  toolsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 20,
  },
  toolChip: {
    backgroundColor: COLORS.primary + "08",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.primary + "20",
  },
  toolText: { fontSize: 13, color: COLORS.primary, fontWeight: "700" },
  spectrumWrap: { marginBottom: 24, padding: 4 },
  spectrumBar: {
    height: 10,
    borderRadius: 5,
    backgroundColor: COLORS.primary,
    marginBottom: 8,
    opacity: 0.8,
  }, // Note: Standardizing this bar
  spectrumLabels: { flexDirection: "row", justifyContent: "space-between" },
  spectrumLabel: { fontSize: 11, fontWeight: "700", color: COLORS.textLight },
  cvdTypesList: { gap: 12, marginBottom: 20 },
  cvdTypeCard: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.lg,
    padding: 16,
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  cvdTypeHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 10,
  },
  cvdSwatches: { flexDirection: "row", gap: 4 },
  cvdSwatch: { width: 24, height: 24, borderRadius: 12 },
  cvdTypeName: { fontSize: 15, fontWeight: "800", color: COLORS.text },
  cvdTypeSub: { fontSize: 11, color: COLORS.primary, fontWeight: "800" },
  cvdTypeDesc: { fontSize: 13, color: COLORS.textLight, lineHeight: 20 },
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    backgroundColor: COLORS.primary + "05",
    borderRadius: RADIUS.xl,
    padding: 20,
    marginBottom: 20,
  },
  statItem: { alignItems: "center" },
  statValue: { fontSize: 24, fontWeight: "900", color: COLORS.primary },
  statLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.textLight,
    marginTop: 4,
  },
  coneRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginBottom: 24,
  },
  coneItem: { alignItems: "center", gap: 6 },
  coneCircle: { width: 56, height: 56, borderRadius: 28, ...SHADOW.sm },
  coneLabel: { fontSize: 14, fontWeight: "800", color: COLORS.text },
  coneSub: { fontSize: 11, color: COLORS.textLight },
  causesRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 24,
  },
  causeChip: {
    backgroundColor: "#FFF",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: "#E2E8F0",
  },
  causeText: { fontSize: 13, color: COLORS.text, fontWeight: "700" },
  bentoGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 24,
  },
  bentoCell: {
    width: "48%",
    backgroundColor: "#FFF",
    borderRadius: RADIUS.lg,
    padding: 16,
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  bentoCellTitle: {
    fontSize: 14,
    fontWeight: "800",
    color: COLORS.text,
    marginBottom: 6,
  },
  bentoCellDesc: { fontSize: 12, color: COLORS.textLight, lineHeight: 18 },
  quickWinsList: { gap: 12, marginBottom: 24 },
  quickWinRow: { flexDirection: "row", alignItems: "flex-start", gap: 12 },
  quickWinText: {
    flex: 1,
    fontSize: 14,
    color: COLORS.text,
    lineHeight: 22,
    fontWeight: "600",
  },
  articleList: { gap: 12, marginBottom: 24 },
  listItem: { flexDirection: "row", gap: 12, alignItems: "flex-start" },
  listDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.primary,
    marginTop: 7,
  },
  listItemTitle: { fontSize: 15, fontWeight: "800", color: COLORS.text },
  listItemDesc: { fontSize: 13, color: COLORS.textLight, lineHeight: 20 },

  clinicalDisclaimer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    backgroundColor: "#FFF8E1",
    padding: 16,
    borderRadius: RADIUS.lg,
    marginTop: 30,
    borderWidth: 1,
    borderColor: "#FFE082",
  },
  disclaimerText: {
    fontSize: 10,
    fontWeight: "900",
    color: "#B45309",
    letterSpacing: 0.5,
  },
  footerInfo: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    marginTop: 10,
    opacity: 0.5,
  },
  footerInfoText: { fontSize: 10, color: COLORS.textLight, fontWeight: "700" },
});
