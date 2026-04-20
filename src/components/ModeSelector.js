import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { Text, TouchableOpacity, View } from 'react-native';
import { AppLog } from '../utils/logger';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

const MODE_TO_SCREEN = {
  Enhancement: 'CameraEnhance',
  Identifier: 'ColorIdentifier',
  Simulation: 'CVDSimulation',
};

export default function ModeSelector({ visible, onClose, navigation, currentMode }) {
  if (!visible) return null;

  const navigateTo = (screen) => {
    if (screen === MODE_TO_SCREEN[currentMode]) {
      onClose();
      return;
    }
    AppLog.log('ModeSelector', `switching from ${currentMode} to ${screen}`);
    onClose();
    // Delay lets VisionCamera fully release the device before the new screen acquires it
    setTimeout(() => navigation.replace(screen), 500);
  };

  return (
    <TouchableOpacity activeOpacity={1} onPress={onClose} style={styles.modalOverlay}>
      <View style={styles.modalContent}>
        <Text style={styles.modalTitle}>Select Mode</Text>

        <TouchableOpacity
          style={[styles.modalOption, currentMode === 'Enhancement' && styles.modalOptionActive]}
          onPress={() => navigateTo('CameraEnhance')}
        >
          <Ionicons name="color-wand" size={20} color={currentMode === 'Enhancement' ? '#FFF' : '#333'} />
          <Text style={[styles.modalText, currentMode === 'Enhancement' && { color: '#FFF' }]}>
            Color Enhancement
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modalOption, currentMode === 'Identifier' && styles.modalOptionActive]}
          onPress={() => navigateTo('ColorIdentifier')}
        >
          <Ionicons name="eyedrop" size={20} color={currentMode === 'Identifier' ? '#FFF' : '#333'} />
          <Text style={[styles.modalText, currentMode === 'Identifier' && { color: '#FFF' }]}>
            Color Identifier
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modalOption, currentMode === 'Simulation' && styles.modalOptionActive]}
          onPress={() => navigateTo('CVDSimulation')}
        >
          <Ionicons name="eye" size={20} color={currentMode === 'Simulation' ? '#FFF' : '#333'} />
          <Text style={[styles.modalText, currentMode === 'Simulation' && { color: '#FFF' }]}>
            CVD Simulation
          </Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
}
