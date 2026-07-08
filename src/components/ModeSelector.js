import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { Text, TouchableOpacity, View, StyleSheet, Dimensions } from 'react-native';
import { MotiView } from 'moti';
import { AppLog } from '../utils/logger';
import { COLORS } from '../theme/colors';

const { width } = Dimensions.get('window');

const MODE_TO_SCREEN = {
  Enhancement: 'CameraEnhance',
  Identifier: 'ColorIdentifier',
  Simulation: 'CVDSimulation',
};

const MODES = [
  {
    key: 'Enhancement',
    screen: 'CameraEnhance',
    title: 'Color Enhancement',
    desc: 'Calibrate and shift camera colors to assist color vision deficiencies.',
    icon: 'color-wand',
    activeColor: '#6C63FF',
    bgColor: '#EEF2FF',
  },
  {
    key: 'Identifier',
    screen: 'ColorIdentifier',
    title: 'Color Identifier',
    desc: 'Pinpoint and name specific color shades, RGB values, and color names.',
    icon: 'eyedrop',
    activeColor: '#EC4899',
    bgColor: '#FDF2F8',
  },
  {
    key: 'Simulation',
    screen: 'CVDSimulation',
    title: 'CVD Simulation',
    desc: 'Experience how people with Protan, Deutan, or Tritan CVD conditions perceive the world.',
    icon: 'eye',
    activeColor: '#3B82F6',
    bgColor: '#EFF6FF',
  },
];

export default function ModeSelector({ visible, onClose, navigation, currentMode }) {
  if (!visible) return null;

  const navigateTo = (screen) => {
    if (screen === MODE_TO_SCREEN[currentMode]) {
      onClose();
      return;
    }
    AppLog.log('ModeSelector', `switching from ${currentMode} to ${screen}`);
    onClose();
    setTimeout(() => navigation.replace(screen), 400);
  };

  return (
    <TouchableOpacity
      activeOpacity={1}
      onPress={onClose}
      style={localStyles.overlay}
    >
      <MotiView
        from={{ opacity: 0, scale: 0.95, translateY: 15 }}
        animate={{ opacity: 1, scale: 1, translateY: 0 }}
        transition={{ type: 'timing', duration: 250 }}
        style={localStyles.modalCard}
      >
        <View style={localStyles.header}>
          <Text style={localStyles.title}>Select Camera Tool</Text>
          <TouchableOpacity onPress={onClose} style={localStyles.closeBtn}>
            <Ionicons name="close" size={20} color="#64748B" />
          </TouchableOpacity>
        </View>

        <View style={localStyles.optionsContainer}>
          {MODES.map((mode) => {
            const isActive = currentMode === mode.key;
            return (
              <TouchableOpacity
                key={mode.key}
                activeOpacity={0.8}
                style={[
                  localStyles.card,
                  isActive && {
                    borderColor: mode.activeColor,
                    backgroundColor: mode.bgColor,
                    borderWidth: 2,
                  },
                ]}
                onPress={() => navigateTo(mode.screen)}
              >
                <View
                  style={[
                    localStyles.iconContainer,
                    { backgroundColor: isActive ? mode.activeColor : '#F1F5F9' },
                  ]}
                >
                  <Ionicons
                    name={mode.icon}
                    size={22}
                    color={isActive ? '#FFF' : '#475569'}
                  />
                </View>
                <View style={localStyles.textContainer}>
                  <Text
                    style={[
                      localStyles.cardTitle,
                      isActive && { color: mode.activeColor, fontWeight: '800' },
                    ]}
                  >
                    {mode.title}
                  </Text>
                  <Text style={localStyles.cardDesc}>{mode.desc}</Text>
                </View>
                {isActive && (
                  <View style={[localStyles.indicator, { backgroundColor: mode.activeColor }]}>
                    <Ionicons name="checkmark" size={12} color="#FFF" />
                  </View>
                )}
              </TouchableOpacity>
            );
          })}
        </View>
      </MotiView>
    </TouchableOpacity>
  );
}

const localStyles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.45)', // Sleek modern slate overlay
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 999,
    paddingHorizontal: 20,
  },
  modalCard: {
    backgroundColor: '#FFF',
    width: width - 40,
    maxWidth: 380,
    borderRadius: 24,
    padding: 20,
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowRadius: 15,
    shadowOffset: { width: 0, height: 8 },
    elevation: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: '800',
    color: '#0F172A',
    letterSpacing: -0.2,
  },
  closeBtn: {
    padding: 4,
    backgroundColor: '#F8FAFC',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  optionsContainer: {
    gap: 12,
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF',
    borderWidth: 1.5,
    borderColor: '#E2E8F0',
    borderRadius: 18,
    padding: 14,
    position: 'relative',
  },
  iconContainer: {
    width: 44,
    height: 44,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  textContainer: {
    flex: 1,
    marginLeft: 14,
    marginRight: 10,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1E293B',
    marginBottom: 3,
  },
  cardDesc: {
    fontSize: 11,
    color: '#64748B',
    lineHeight: 15,
  },
  indicator: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 18,
    height: 18,
    borderRadius: 9,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
