import { Ionicons } from '@expo/vector-icons';
import { MotiView } from 'moti';
import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { COLORS, RADIUS, SHADOW, SPACING } from '../theme/colors';

function ScoreRing({ percentage, color }) {
  return (
    <View style={styles.ringWrap}>
      <View style={[styles.ring, { borderColor: color }]}>
        <Text style={[styles.ringPercent, { color }]}>{percentage}%</Text>
        <Text style={styles.ringLabel}>Score</Text>
      </View>
    </View>
  );
}

function InfoCard({ icon, iconColor, bg, title, children }) {
  return (
    <MotiView
      from={{ opacity: 0, translateY: 12 }}
      animate={{ opacity: 1, translateY: 0 }}
      transition={{ type: 'spring', damping: 18 }}
      style={[styles.infoCard, { backgroundColor: bg || COLORS.card }]}
    >
      <View style={styles.infoCardHeader}>
        <Ionicons name={icon} size={22} color={iconColor} />
        <Text style={[styles.infoCardTitle, { color: iconColor }]}>{title}</Text>
      </View>
      {children}
    </MotiView>
  );
}

export default function ResultsScreen({ route, navigation }) {
  const {
    score = 0,
    maxScore,
    total = 38,
    diagnosis = 'Unknown',
    severity = 'N/A',
    percentage = 0,
    shuffledOrder,
  } = route?.params || {};

  const displayScore = maxScore ?? total;
  const isNormal = diagnosis === 'Normal Vision';

  const severityColor =
    severity === 'Severe' ? COLORS.danger : severity === 'Moderate' ? COLORS.warning : COLORS.success;

  const displayDiagnosis = isNormal ? diagnosis : `Likely ${diagnosis}`;

  const getDescription = () => {
    if (isNormal)
      return 'Your colour vision screening appears to be within the normal range. No significant colour confusion patterns were detected during this session.';
    if (diagnosis.includes('Protan'))
      return 'Screening suggests possible red-channel sensitivity reduction (Protanomaly/Protanopia). Red and green may appear similar. This is a screening result only — consult a qualified eye care professional for clinical confirmation.';
    if (diagnosis.includes('Deutan'))
      return 'Screening suggests possible green-channel sensitivity reduction (Deuteranomaly/Deuteranopia). Red and green may appear similar. This is a screening result only — consult a qualified eye care professional for clinical confirmation.';
    if (diagnosis.includes('Tritan'))
      return 'Screening suggests possible blue-channel sensitivity reduction (Tritanomaly/Tritanopia). Blue and green may appear similar. This is a screening result only — consult a qualified eye care professional for clinical confirmation.';
    return 'Screening suggests a possible colour vision variation. This is not a diagnosis — consult a qualified eye care professional for a comprehensive assessment.';
  };

  return (
    <ScrollView style={styles.root} showsVerticalScrollIndicator={false}>
      {/* ── HERO BANNER ── */}
      <MotiView
        from={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', damping: 16 }}
        style={[styles.heroBanner, { backgroundColor: isNormal ? '#E8F5E9' : '#FFF3E0' }]}
      >
        <Ionicons
          name={isNormal ? 'checkmark-circle' : 'alert-circle'}
          size={44}
          color={isNormal ? COLORS.success : COLORS.warning}
        />
        <Text style={styles.heroLabel}>Screening Result</Text>
        <Text style={styles.heroDiagnosis}>{displayDiagnosis}</Text>

        <ScoreRing percentage={percentage} color={severityColor} />

        <Text style={styles.heroScore}>
          {score} / {displayScore} weighted points
        </Text>

        <View style={[styles.severityBadge, { backgroundColor: severityColor }]}>
          <Text style={styles.severityText}>Severity: {severity}</Text>
        </View>
      </MotiView>

      <View style={styles.content}>
        {/* What this means */}
        <InfoCard icon="eye" iconColor="#2196F3" bg="#E3F2FD" title="What This Means">
          <Text style={styles.infoText}>{getDescription()}</Text>
        </InfoCard>

        {/* Recommendations */}
        <InfoCard icon="star" iconColor={COLORS.primary} title="Recommendations">
          <TouchableOpacity
            style={styles.recRow}
            onPress={() => navigation.navigate('CameraEnhance')}
          >
            <View style={[styles.recIcon, { backgroundColor: '#F3E5F5' }]}>
              <Ionicons name="camera" size={18} color="#9C27B0" />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.recTitle}>Colour Enhancement</Text>
              <Text style={styles.recSub}>Real-time camera colour mode</Text>
            </View>
          </TouchableOpacity>

          <View style={styles.recDivider} />

          <View style={styles.recRow}>
            <View style={[styles.recIcon, { backgroundColor: '#E3F2FD' }]}>
              <Ionicons name="medkit" size={18} color="#2196F3" />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.recTitle}>Professional Care</Text>
              <Text style={styles.recSub}>Consult a qualified eye care professional</Text>
            </View>
          </View>
        </InfoCard>

        {/* Test metadata */}
        <InfoCard icon="information-circle" iconColor={COLORS.textLight} title="Test Details">
          <View style={styles.metaGrid}>
            <View style={styles.metaItem}>
              <Text style={styles.metaValue}>{total}</Text>
              <Text style={styles.metaKey}>Total Plates</Text>
            </View>
            <View style={styles.metaItem}>
              <Text style={styles.metaValue}>{displayScore}</Text>
              <Text style={styles.metaKey}>Max Score</Text>
            </View>
            <View style={styles.metaItem}>
              <Text style={styles.metaValue}>{score}</Text>
              <Text style={styles.metaKey}>Weighted Score</Text>
            </View>
          </View>
          {shuffledOrder && (
            <Text style={styles.shuffleNote}>
              Plate order was randomised for this session to prevent memorisation.
            </Text>
          )}
        </InfoCard>

        {/* Actions */}
        <TouchableOpacity
          style={styles.primaryBtn}
          onPress={() => navigation.navigate('MainTabs')}
          activeOpacity={0.85}
        >
          <Text style={styles.primaryBtnText}>Back to Home</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryBtn}
          onPress={() => navigation.replace('IshiharaOnboarding', { testType: 'comprehensive' })}
        >
          <Text style={styles.secondaryBtnText}>Retake Test</Text>
        </TouchableOpacity>

        {/* Legal disclaimer */}
        <View style={styles.disclaimer}>
          <Ionicons name="warning-outline" size={13} color={COLORS.warning} />
          <Text style={styles.disclaimerText}>
            Screening tool only. Not a medical diagnosis. Consult qualified eye care professionals for a comprehensive assessment.
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },

  heroBanner: {
    padding: SPACING.xl,
    alignItems: 'center',
    gap: SPACING.sm,
  },
  heroLabel: { fontSize: 12, color: '#888', fontWeight: '600', letterSpacing: 0.5 },
  heroDiagnosis: { fontSize: 22, fontWeight: '800', color: COLORS.text, textAlign: 'center' },
  heroScore: { fontSize: 13, color: '#666' },

  ringWrap: { marginVertical: SPACING.sm },
  ring: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 6,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.card,
    ...SHADOW.md,
  },
  ringPercent: { fontSize: 26, fontWeight: '800' },
  ringLabel: { fontSize: 10, color: COLORS.textLight, marginTop: -2 },

  severityBadge: {
    paddingHorizontal: SPACING.md,
    paddingVertical: 5,
    borderRadius: 16,
    marginTop: 4,
  },
  severityText: { color: '#FFF', fontSize: 12, fontWeight: '700' },

  content: { padding: SPACING.md, gap: SPACING.md, paddingBottom: SPACING.xxl },

  infoCard: {
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    ...SHADOW.sm,
  },
  infoCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  infoCardTitle: { fontSize: 15, fontWeight: '700' },
  infoText: { fontSize: 14, color: COLORS.text, lineHeight: 21 },

  recRow: { flexDirection: 'row', alignItems: 'center', gap: SPACING.md },
  recIcon: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center' },
  recTitle: { fontSize: 14, fontWeight: '600', color: COLORS.text },
  recSub: { fontSize: 11, color: COLORS.textLight },
  recDivider: { height: 1, backgroundColor: COLORS.border, marginVertical: SPACING.sm },

  metaGrid: { flexDirection: 'row', justifyContent: 'space-around', marginBottom: SPACING.sm },
  metaItem: { alignItems: 'center' },
  metaValue: { fontSize: 22, fontWeight: '800', color: COLORS.primary },
  metaKey: { fontSize: 11, color: COLORS.textLight, marginTop: 2 },
  shuffleNote: { fontSize: 11, color: COLORS.textLight, textAlign: 'center', fontStyle: 'italic' },

  primaryBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.lg,
    paddingVertical: 14,
    alignItems: 'center',
    ...SHADOW.md,
  },
  primaryBtnText: { color: '#FFF', fontWeight: '700', fontSize: 16 },

  secondaryBtn: {
    borderWidth: 1.5,
    borderColor: COLORS.primary,
    borderRadius: RADIUS.lg,
    paddingVertical: 12,
    alignItems: 'center',
  },
  secondaryBtnText: { color: COLORS.primary, fontWeight: '700', fontSize: 15 },

  disclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.sm,
    backgroundColor: '#FFF8E1',
    padding: SPACING.md,
    borderRadius: RADIUS.md,
  },
  disclaimerText: {
    flex: 1,
    fontSize: 11,
    color: COLORS.warning,
    lineHeight: 17,
    fontWeight: '500',
  },
});
