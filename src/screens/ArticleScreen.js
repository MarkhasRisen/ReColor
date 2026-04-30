import { Ionicons } from "@expo/vector-icons";
import { MotiView } from "moti";
import React from "react";
import {
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from "react-native";
import { ARTICLES } from "../data/articles";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

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
                Don't
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

function ArticleCard({ article, onPress }) {
  return (
    <MotiView
      from={{ opacity: 0, translateY: 12 }}
      animate={{ opacity: 1, translateY: 0 }}
      transition={{ type: "spring", damping: 18 }}
    >
      <TouchableOpacity
        style={styles.articleCard}
        onPress={onPress}
        activeOpacity={0.88}
      >
        <Image source={article.coverImage} style={styles.articleCover} />
        <View style={styles.articleCardBody}>
          <View style={styles.metaRow}>
            <Text style={styles.categoryTag}>{article.category}</Text>
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

function ArticleDetail({ article, onBack }) {
  return (
    <ScrollView style={styles.detailRoot} showsVerticalScrollIndicator={false}>
      <Image source={article.coverImage} style={styles.detailCover} />

      <View style={styles.detailContent}>
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

        {/* Disclaimer */}
        <View style={styles.disclaimer}>
          <Ionicons name="warning-outline" size={16} color={COLORS.warning} />
          <Text style={styles.disclaimerText}>
            SCREENING PURPOSE ONLY. NOT A CLINICAL DIAGNOSIS.
          </Text>
        </View>
      </View>

      <TouchableOpacity style={styles.backFab} onPress={onBack}>
        <Ionicons name="arrow-back" size={22} color="#FFF" />
      </TouchableOpacity>
    </ScrollView>
  );
}

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
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={22} color={COLORS.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Learn & Understand</Text>
        <Image
          source={require("../../assets/icon.png")}
          style={styles.headerLogo}
        />
      </View>

      <ScrollView
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
      >
        {ARTICLES.map((article) => (
          <ArticleCard
            key={article.id}
            article={article}
            onPress={() => setSelected(article)}
          />
        ))}

        <View style={styles.footerDisclaimer}>
          <Ionicons name="warning-outline" size={14} color={COLORS.warning} />
          <Text style={styles.footerDisclaimerText}>
            SCREENING PURPOSE ONLY. NOT A CLINICAL DIAGNOSIS.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: SPACING.md,
    paddingTop: 52,
    paddingBottom: SPACING.md,
    backgroundColor: COLORS.card,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  headerTitle: { fontSize: 17, fontWeight: "700", color: COLORS.text },
  headerLogo: { width: 36, height: 36, resizeMode: "contain" },

  listContent: {
    padding: SPACING.md,
    gap: SPACING.md,
    paddingBottom: SPACING.xxl,
  },

  articleCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    overflow: "hidden",
    ...SHADOW.md,
  },
  articleCover: { width: "100%", height: 160, resizeMode: "cover" },
  articleCardBody: { padding: SPACING.md },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    marginBottom: 6,
  },
  categoryTag: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.primary,
    backgroundColor: COLORS.surfaceAlt,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  readTime: { fontSize: 12, color: COLORS.textLight },
  articleTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: COLORS.text,
    marginBottom: 4,
  },
  articleSummary: { fontSize: 13, color: COLORS.textLight, lineHeight: 19 },

  // Detail view
  detailRoot: { flex: 1, backgroundColor: COLORS.background },
  detailCover: { width: "100%", height: 220, resizeMode: "cover" },
  detailContent: { padding: SPACING.md },
  detailTitle: {
    fontSize: 24,
    fontWeight: "800",
    color: COLORS.text,
    marginBottom: 6,
  },
  detailSummary: {
    fontSize: 14,
    color: COLORS.textLight,
    lineHeight: 21,
    marginBottom: SPACING.md,
  },
  divider: {
    height: 1,
    backgroundColor: COLORS.border,
    marginBottom: SPACING.md,
  },

  sectionHeading: {
    fontSize: 16,
    fontWeight: "800",
    color: COLORS.text,
    marginTop: SPACING.md,
    marginBottom: SPACING.sm,
  },
  paragraph: {
    fontSize: 14,
    color: COLORS.text,
    lineHeight: 22,
    marginBottom: SPACING.sm,
  },

  dosDontsRow: {
    flexDirection: "row",
    gap: SPACING.sm,
    marginBottom: SPACING.md,
  },
  dosDontsCard: {
    flex: 1,
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    padding: SPACING.sm,
    borderTopWidth: 3,
    ...SHADOW.sm,
  },
  dosDontsHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    marginBottom: 6,
  },
  dosDontsTitle: { fontSize: 13, fontWeight: "700" },
  doItem: { fontSize: 12, color: COLORS.text, lineHeight: 18 },
  dontItem: { fontSize: 12, color: COLORS.text, lineHeight: 18 },

  contrastList: { gap: SPACING.sm, marginBottom: SPACING.md },
  contrastRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    backgroundColor: COLORS.card,
    padding: SPACING.sm,
    borderRadius: RADIUS.sm,
    ...SHADOW.sm,
  },
  contrastBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  contrastRatio: { fontSize: 12, fontWeight: "700", color: "#FFF" },
  contrastLabel: { flex: 1, fontSize: 13, color: COLORS.text },

  palettesList: { gap: SPACING.sm, marginBottom: SPACING.md },
  paletteRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.md,
    backgroundColor: COLORS.card,
    padding: SPACING.sm,
    borderRadius: RADIUS.md,
    ...SHADOW.sm,
  },
  swatchRow: { flexDirection: "row", gap: 4 },
  swatch: { width: 28, height: 28, borderRadius: 14 },
  paletteName: { fontSize: 14, fontWeight: "700", color: COLORS.text },
  paletteSafe: { fontSize: 12, color: COLORS.textLight },

  toolsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: SPACING.sm,
    marginBottom: SPACING.md,
  },
  toolChip: {
    backgroundColor: COLORS.surfaceAlt,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  toolText: { fontSize: 13, color: COLORS.primary, fontWeight: "600" },

  articleList: { gap: SPACING.sm, marginBottom: SPACING.md },
  listItem: { flexDirection: "row", gap: SPACING.sm, alignItems: "flex-start" },
  listDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.primary,
    marginTop: 6,
  },
  listItemTitle: { fontSize: 14, fontWeight: "700", color: COLORS.text },
  listItemDesc: { fontSize: 13, color: COLORS.textLight, lineHeight: 19 },

  disclaimer: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    backgroundColor: "#FFF8E1",
    padding: SPACING.md,
    borderRadius: RADIUS.md,
    marginTop: SPACING.lg,
    marginBottom: SPACING.xxl,
  },
  disclaimerText: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.warning,
    flex: 1,
  },

  backFab: {
    position: "absolute",
    top: 48,
    left: SPACING.md,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(0,0,0,0.5)",
    alignItems: "center",
    justifyContent: "center",
  },

  footerDisclaimer: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    backgroundColor: "#FFF8E1",
    padding: SPACING.md,
    borderRadius: RADIUS.md,
    marginTop: SPACING.md,
  },
  footerDisclaimerText: {
    fontSize: 11,
    fontWeight: "700",
    color: COLORS.warning,
  },

  // Severity spectrum
  spectrumWrap: { marginBottom: SPACING.md },
  spectrumBar: {
    height: 12,
    borderRadius: 6,
    background: "linear-gradient(to right, #27AE60, #F39C12, #E74C3C)",
    backgroundColor: "#F39C12",
    marginBottom: 6,
    // Simulated gradient via shadow
    shadowColor: "#E74C3C",
    shadowOffset: { width: 4, height: 0 },
    shadowOpacity: 0.4,
  },
  spectrumLabels: { flexDirection: "row", justifyContent: "space-between" },
  spectrumLabel: { fontSize: 12, color: COLORS.textLight },

  // CVD types
  cvdTypesList: { gap: SPACING.sm, marginBottom: SPACING.md },
  cvdTypeCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    padding: SPACING.sm,
    ...SHADOW.sm,
  },
  cvdTypeHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    marginBottom: 6,
  },
  cvdSwatches: { flexDirection: "row", gap: 4 },
  cvdSwatch: { width: 22, height: 22, borderRadius: 11 },
  cvdTypeName: { fontSize: 14, fontWeight: "700", color: COLORS.text },
  cvdTypeSub: { fontSize: 11, color: COLORS.primary, fontWeight: "600" },
  cvdTypeDesc: { fontSize: 13, color: COLORS.textLight, lineHeight: 18 },

  // Stats
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    backgroundColor: COLORS.surfaceAlt,
    borderRadius: RADIUS.md,
    padding: SPACING.md,
    marginBottom: SPACING.md,
  },
  statItem: { alignItems: "center" },
  statValue: { fontSize: 20, fontWeight: "800", color: COLORS.primary },
  statLabel: {
    fontSize: 11,
    color: COLORS.textLight,
    marginTop: 2,
    textAlign: "center",
  },

  // Cone diagram
  coneRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginBottom: SPACING.md,
  },
  coneItem: { alignItems: "center", gap: 4 },
  coneCircle: { width: 48, height: 48, borderRadius: 24 },
  coneLabel: { fontSize: 13, fontWeight: "700", color: COLORS.text },
  coneSub: { fontSize: 11, color: COLORS.textLight },

  // Causes
  causesRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: SPACING.sm,
    marginBottom: SPACING.md,
  },
  causeChip: {
    backgroundColor: COLORS.surfaceAlt,
    borderRadius: 20,
    paddingHorizontal: SPACING.md,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  causeText: { fontSize: 13, color: COLORS.text, fontWeight: "500" },

  // Bento grid
  bentoGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: SPACING.sm,
    marginBottom: SPACING.md,
  },
  bentoCell: {
    width: "47%",
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    padding: SPACING.sm,
    ...SHADOW.sm,
  },
  bentoCellTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: COLORS.text,
    marginBottom: 4,
  },
  bentoCellDesc: { fontSize: 12, color: COLORS.textLight, lineHeight: 17 },

  // Quick wins
  quickWinsList: { gap: SPACING.sm, marginBottom: SPACING.md },
  quickWinRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: SPACING.sm,
  },
  quickWinText: { flex: 1, fontSize: 13, color: COLORS.text, lineHeight: 19 },
});
