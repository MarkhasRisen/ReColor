import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import { MotiView } from 'moti';
import React, { useRef, useState } from 'react';
import {
  Dimensions,
  FlatList,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { COLORS, RADIUS, SHADOW, SPACING } from '../theme/colors';

const { width } = Dimensions.get('window');

const SLIDES = [
  {
    id: '1',
    icon: 'eye-outline',
    iconColor: COLORS.primary,
    badge: 'WELCOME',
    title: "Let's personalise\nyour view of the world.",
    subtitle:
      'ReColor uses AI and clinical screening to help you understand and enhance your colour perception.',
    cta: 'Get Started',
  },
  {
    id: '2',
    icon: 'color-palette-outline',
    iconColor: '#FF6B6B',
    badge: 'COLOUR VISION',
    title: 'What is your\ncolour vision type?',
    subtitle:
      'Millions of people experience colour differently. Our Ishihara test identifies Deuteranopia, Protanopia, and Tritanopia.',
    cta: 'Next',
    options: [
      'Red Sensitivity (Protanopia/Protanomaly)',
      'Green Sensitivity (Deuteranopia/Deuteranomaly)',
      'Blue Sensitivity (Tritanopia/Tritanomaly)',
      'Not sure yet',
    ],
  },
  {
    id: '3',
    icon: 'phone-portrait-outline',
    iconColor: COLORS.accent,
    badge: 'LIVE PREVIEW',
    title: 'See the difference,\nright now.',
    subtitle:
      'Point your camera anywhere. ReColor identifies colours in real-time and applies adaptive enhancement filters.',
    cta: 'Next',
  },
  {
    id: '4',
    icon: 'camera-outline',
    iconColor: COLORS.warning,
    badge: 'PERMISSIONS',
    title: "We'll need your camera\nto act as your eyes.",
    subtitle:
      'Camera access enables real-time colour identification and CVD simulation. Photo library access lets you enhance saved images.',
    cta: 'Continue',
    perms: ['Camera Access', 'Photo Storage'],
  },
  {
    id: '5',
    icon: 'options-outline',
    iconColor: COLORS.success,
    badge: 'READY',
    title: 'Choose your\nexperience.',
    subtitle: 'Start with the clinical screening test or jump straight into colour enhancement mode.',
    cta: 'Begin →',
    finalCta: true,
  },
];

function SlideItem({ item, onNext, isLast, onFinish }) {
  const [selected, setSelected] = useState(null);

  return (
    <View style={[styles.slide, { width }]}>
      <MotiView
        from={{ opacity: 0, translateY: 30 }}
        animate={{ opacity: 1, translateY: 0 }}
        transition={{ type: 'spring', damping: 18, delay: 100 }}
        style={styles.iconWrap}
      >
        <Ionicons name={item.icon} size={48} color={item.iconColor} />
      </MotiView>

      <MotiView
        from={{ opacity: 0, translateY: 20 }}
        animate={{ opacity: 1, translateY: 0 }}
        transition={{ type: 'spring', damping: 18, delay: 200 }}
      >
        <Text style={styles.badge}>{item.badge}</Text>
        <Text style={styles.title}>{item.title}</Text>
        <Text style={styles.subtitle}>{item.subtitle}</Text>
      </MotiView>

      {item.options && (
        <MotiView
          from={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 350 }}
          style={styles.optionsWrap}
        >
          {item.options.map((opt) => (
            <TouchableOpacity
              key={opt}
              style={[styles.optionChip, selected === opt && styles.optionChipActive]}
              onPress={() => setSelected(opt)}
            >
              <Text style={[styles.optionText, selected === opt && styles.optionTextActive]}>
                {opt}
              </Text>
            </TouchableOpacity>
          ))}
        </MotiView>
      )}

      {item.perms && (
        <MotiView
          from={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 350 }}
          style={styles.permsWrap}
        >
          {item.perms.map((perm) => (
            <View key={perm} style={styles.permRow}>
              <Ionicons name="checkmark-circle" size={20} color={COLORS.success} />
              <Text style={styles.permText}>{perm}</Text>
            </View>
          ))}
        </MotiView>
      )}

      <MotiView
        from={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', damping: 14, delay: 400 }}
        style={styles.ctaWrap}
      >
        <TouchableOpacity
          style={styles.ctaBtn}
          onPress={isLast ? onFinish : onNext}
          activeOpacity={0.85}
        >
          <Text style={styles.ctaText}>{item.cta}</Text>
        </TouchableOpacity>
      </MotiView>
    </View>
  );
}

export default function AppOnboarding({ navigation }) {
  const flatRef = useRef(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  const goNext = () => {
    if (currentIndex < SLIDES.length - 1) {
      flatRef.current?.scrollToIndex({ index: currentIndex + 1, animated: true });
      setCurrentIndex(currentIndex + 1);
    }
  };

  const finish = async () => {
    await AsyncStorage.setItem('@recolor_onboarded', '1').catch(() => {});
    navigation.replace('Login');
  };

  return (
    <View style={styles.container}>
      <FlatList
        ref={flatRef}
        data={SLIDES}
        keyExtractor={(s) => s.id}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        scrollEnabled={false}
        renderItem={({ item, index }) => (
          <SlideItem
            item={item}
            onNext={goNext}
            isLast={index === SLIDES.length - 1}
            onFinish={finish}
          />
        )}
      />

      {/* Dots indicator */}
      <View style={styles.dotsRow}>
        {SLIDES.map((_, i) => (
          <MotiView
            key={i}
            animate={{ width: i === currentIndex ? 24 : 8, opacity: i === currentIndex ? 1 : 0.35 }}
            transition={{ type: 'spring', damping: 18 }}
            style={[styles.dot, { backgroundColor: COLORS.primary }]}
          />
        ))}
      </View>

      <TouchableOpacity style={styles.skipBtn} onPress={finish}>
        <Text style={styles.skipText}>Skip</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  slide: {
    flex: 1,
    paddingHorizontal: SPACING.md,
    paddingTop: 80,
    paddingBottom: 120,
    alignItems: 'center',
  },
  iconWrap: {
    width: 96,
    height: 96,
    borderRadius: RADIUS.xl,
    backgroundColor: COLORS.card,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.lg,
    ...SHADOW.md,
  },
  badge: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.5,
    color: COLORS.primary,
    textAlign: 'center',
    marginBottom: SPACING.sm,
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: COLORS.text,
    textAlign: 'center',
    lineHeight: 36,
    marginBottom: SPACING.md,
  },
  subtitle: {
    fontSize: 15,
    color: COLORS.textLight,
    textAlign: 'center',
    lineHeight: 22,
    paddingHorizontal: SPACING.sm,
  },
  optionsWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: SPACING.sm,
    marginTop: SPACING.lg,
  },
  optionChip: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: 20,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    backgroundColor: COLORS.card,
  },
  optionChipActive: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.surfaceAlt,
  },
  optionText: {
    fontSize: 14,
    color: COLORS.textLight,
    fontWeight: '500',
  },
  optionTextActive: {
    color: COLORS.primary,
    fontWeight: '700',
  },
  permsWrap: {
    marginTop: SPACING.lg,
    alignSelf: 'stretch',
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
  },
  permRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    backgroundColor: COLORS.card,
    padding: SPACING.md,
    borderRadius: RADIUS.md,
    ...SHADOW.sm,
  },
  permText: {
    fontSize: 14,
    color: COLORS.text,
    fontWeight: '500',
  },
  ctaWrap: {
    position: 'absolute',
    bottom: SPACING.xxl,
    left: SPACING.md,
    right: SPACING.md,
  },
  ctaBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: RADIUS.lg,
    paddingVertical: 16,
    alignItems: 'center',
    ...SHADOW.md,
  },
  ctaText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  dotsRow: {
    position: 'absolute',
    bottom: 110,
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 6,
    alignItems: 'center',
  },
  dot: {
    height: 8,
    borderRadius: 4,
  },
  skipBtn: {
    position: 'absolute',
    top: 52,
    right: SPACING.md,
    padding: SPACING.sm,
  },
  skipText: {
    color: COLORS.textLight,
    fontSize: 14,
    fontWeight: '500',
  },
});
