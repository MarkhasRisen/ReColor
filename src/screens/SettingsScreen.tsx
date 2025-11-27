import React from 'react';
import { View, Button, Alert, Text, TouchableOpacity } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'ReColor_HasSeenCalibrationWarning';

export const SettingsScreen: React.FC<{ onClose?: () => void }> = ({ onClose }) => {
  const resetWarnings = async () => {
    try {
      await AsyncStorage.removeItem(STORAGE_KEY);
      Alert.alert('Reset Complete', 'Calibration warning will appear next time.');
    } catch (e) {
      Alert.alert('Error', 'Failed to reset calibration warning');
    }
  };

  return (
    <View style={{ flex: 1, padding: 16, backgroundColor: 'white' }}>
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Text style={{ fontSize: 20, fontWeight: '700' }}>Settings</Text>
        <TouchableOpacity onPress={onClose} style={{ paddingHorizontal: 8, paddingVertical: 6 }}>
          <Text style={{ fontSize: 18 }}>Close</Text>
        </TouchableOpacity>
      </View>
      <View style={{ height: 1, backgroundColor: '#eee', marginBottom: 16 }} />
      <Button title="Reset Calibration Warning" onPress={resetWarnings} />
    </View>
  );
};

export default SettingsScreen;
