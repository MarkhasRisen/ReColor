import React from 'react';
import { SafeAreaView, View, Text, Pressable, Modal, TouchableOpacity } from 'react-native';
import Slider from '@react-native-community/slider';
import { useSharedValue } from 'react-native-reanimated';
import { DaltonizedCamera } from './src/components/DaltonizedCamera';
import { type CVDType } from './src/constants/matrices';
import CalibrationGuard from './src/components/CalibrationGuard';
import SettingsScreen from './src/screens/SettingsScreen';

export default function App() {
  const strength = useSharedValue(0.6);
  const [cvdType, setCvdType] = React.useState<CVDType>('protan');
  const [showSettings, setShowSettings] = React.useState(false);
  const [compareEnabled, setCompareEnabled] = React.useState(false);
  const [compareSplit, setCompareSplit] = React.useState(0.5);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: 'black' }}>
      <CalibrationGuard />
      <DaltonizedCamera
        strength={strength}
        cvdType={cvdType}
        compare={compareEnabled}
        compareSplit={compareSplit}
      />

      {/* Top bar with app name + gear */}
      <View style={{ position: 'absolute', left: 12, right: 12, top: 12, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text style={{ color: 'white', fontWeight: 'bold', fontSize: 20, textShadowColor: 'rgba(0,0,0,0.75)', textShadowOffset: { width: -1, height: 1 }, textShadowRadius: 8 }}>ReColor</Text>
        <TouchableOpacity onPress={() => setShowSettings(true)} style={{ paddingHorizontal: 10, paddingVertical: 6, backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: 18 }}>
          <Text style={{ fontSize: 20 }}>⚙️</Text>
        </TouchableOpacity>
      </View>

      {/* Bottom controls */}
      <View style={{ position: 'absolute', left: 12, right: 12, bottom: 24 }}>
        <Text style={{ color: 'white', marginBottom: 8 }}>Strength</Text>
        <Slider
          minimumValue={0}
          maximumValue={1}
          value={strength.value}
          step={0.01}
          onValueChange={(v: number) => (strength.value = v)}
          minimumTrackTintColor="#6cf"
          maximumTrackTintColor="#888"
          thumbTintColor="#6cf"
        />

        {/* Single control: Strength (gain) only */}

        {/* Quick CVD selector */}
        <View style={{ flexDirection: 'row', marginTop: 16, gap: 8 }}>
          {(['protan', 'deutan', 'tritan'] as CVDType[]).map((t) => (
            <Pressable
              key={t}
              onPress={() => setCvdType(t)}
              style={{
                paddingVertical: 8,
                paddingHorizontal: 12,
                borderRadius: 8,
                backgroundColor: cvdType === t ? '#6cf' : 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                borderColor: 'rgba(255,255,255,0.2)',
                marginRight: 8,
              }}
            >
              <Text style={{ color: 'white', fontWeight: '600', textTransform: 'capitalize' }}>{t}</Text>
            </Pressable>
          ))}
        </View>

        {/* Compare toggle + split slider */}
        <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 16, justifyContent: 'space-between' }}>
          <Pressable
            onPress={() => setCompareEnabled((v: boolean) => !v)}
            style={{
              paddingVertical: 8,
              paddingHorizontal: 12,
              borderRadius: 8,
              backgroundColor: compareEnabled ? '#6cf' : 'rgba(255,255,255,0.1)',
              borderWidth: 1,
              borderColor: 'rgba(255,255,255,0.2)'
            }}
          >
            <Text style={{ color: 'white', fontWeight: '600' }}>{compareEnabled ? 'Compare: On' : 'Compare: Off'}</Text>
          </Pressable>
          <View style={{ flex: 1, marginLeft: 12 }}>
            <Text style={{ color: 'white', marginBottom: 8 }}>Split</Text>
            <Slider
              minimumValue={0}
              maximumValue={1}
              value={compareSplit}
              step={0.01}
              onValueChange={(v: number) => setCompareSplit(v)}
              minimumTrackTintColor="#6cf"
              maximumTrackTintColor="#888"
              thumbTintColor="#6cf"
              disabled={!compareEnabled}
            />
          </View>
        </View>
      </View>

      {/* Settings modal overlay (keeps camera hot) */}
      <Modal
        visible={showSettings}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowSettings(false)}
      >
        <SettingsScreen onClose={() => setShowSettings(false)} />
      </Modal>
    </SafeAreaView>
  );
}
