import React from 'react';
import { View, Text, Modal, StyleSheet, TouchableOpacity } from 'react-native';
import ScreenBrightness from 'react-native-screen-brightness';
import { activateKeepAwake, deactivateKeepAwake } from 'react-native-keep-awake';
import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'ReColor_HasSeenCalibrationWarning';

export const CalibrationGuard: React.FC = () => {
  const [isVisible, setIsVisible] = React.useState(false);
  const [isReady, setIsReady] = React.useState(false);
  const prevRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    let mounted = true;
    const init = async () => {
      activateKeepAwake();

      // Capture and set brightness to max (with guarded permission calls)
      try {
        const current = await ScreenBrightness.getBrightness();
        if (mounted) prevRef.current = typeof current === 'number' ? current : null;

        const hasPerm = typeof (ScreenBrightness as any).hasPermission === 'function'
          ? await (ScreenBrightness as any).hasPermission()
          : true;
        if (!hasPerm && typeof (ScreenBrightness as any).requestPermission === 'function') {
          await (ScreenBrightness as any).requestPermission();
        }
        await ScreenBrightness.setBrightness(1.0);
      } catch (e) {
        console.warn('Brightness failed', e);
      }

      // Decide if we should show the warning modal
      try {
        const hasSeen = await AsyncStorage.getItem(STORAGE_KEY);
        if (hasSeen !== 'true') setIsVisible(true);
      } catch {
        setIsVisible(true);
      } finally {
        if (mounted) setIsReady(true);
      }
    };

    init();

    return () => {
      mounted = false;
      try { deactivateKeepAwake(); } catch {}
      const prev = prevRef.current;
      if (typeof prev === 'number') {
        ScreenBrightness.setBrightness(Math.max(0, Math.min(1, prev))).catch(() => {});
      }
    };
  }, []);

  const handleDismiss = async () => {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, 'true');
    } catch {}
    setIsVisible(false);
  };

  if (!isReady) return null;

  return (
    <Modal animationType="fade" transparent visible={isVisible} onRequestClose={handleDismiss}>
      <View style={styles.centeredView}>
        <View style={styles.modalView}>
          <Text style={styles.title}>⚠️ Color Accuracy Check</Text>
          <View style={styles.checklist}>
            <Text style={styles.checkItem}>✅ Brightness Maxed (Auto)</Text>
            <Text style={styles.checkItem}>❌ Disable True Tone</Text>
            <Text style={styles.checkItem}>❌ Disable Night Shift / Blue Light</Text>
            <Text style={styles.checkItem}>❌ Disable Accessibility color inversion/correction</Text>
          </View>
          <Text style={styles.subtext}>These system filters distort the daltonization output.</Text>
          <TouchableOpacity style={styles.button} onPress={handleDismiss}>
            <Text style={styles.buttonText}>I Understand</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  centeredView: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.8)'
  },
  modalView: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 20,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
    maxWidth: 420,
  },
  title: { fontSize: 18, fontWeight: 'bold', marginBottom: 10, color: '#D32F2F' },
  checklist: { alignItems: 'flex-start', marginBottom: 12 },
  checkItem: { fontSize: 15, marginVertical: 3, fontWeight: '500', color: '#222' },
  subtext: { fontSize: 12, color: '#666', textAlign: 'center', marginBottom: 12 },
  button: { backgroundColor: '#2196F3', borderRadius: 10, paddingVertical: 10, paddingHorizontal: 12, minWidth: 140 },
  buttonText: { color: 'white', fontWeight: 'bold', textAlign: 'center' },
});

export default CalibrationGuard;
