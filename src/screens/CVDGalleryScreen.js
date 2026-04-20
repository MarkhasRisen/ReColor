import { Ionicons } from '@expo/vector-icons';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import React, { useState } from 'react';
import { ActivityIndicator, Dimensions, Image, SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { decodeJpegBase64, encodeToDataUri, getCVDRows } from '../../tensorHelper';
import { CAPTURE_SIZE } from '../utils/constants';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

const { width } = Dimensions.get('window');

export default function CVDGalleryScreen({ navigation }) {
  const [originalUri, setOriginalUri] = useState(null);
  const [displayUri, setDisplayUri] = useState(null);
  const [mode, setMode] = useState('Off');
  const [processing, setProcessing] = useState(false);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, allowsEditing: true, quality: 1 });
    if (!result.canceled) {
      const uri = result.assets[0].uri;
      setOriginalUri(uri);
      setDisplayUri(uri);
      setMode('Off');
    }
  };

  const applyFilter = async (cvdMode) => {
    setMode(cvdMode);
    if (cvdMode === 'Off' || !originalUri) { setDisplayUri(originalUri); return; }
    setProcessing(true);
    try {
      const resized = await ImageManipulator.manipulateAsync(originalUri, [{ resize: { width: CAPTURE_SIZE } }], { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 0.9 });
      const rawImage = decodeJpegBase64(resized.base64);
      const { row0, row1, row2 } = getCVDRows(cvdMode);
      const pixels = rawImage.data;
      for (let i = 0; i < pixels.length; i += 4) {
        const r = Math.pow(pixels[i] / 255, 2.2);
        const g = Math.pow(pixels[i + 1] / 255, 2.2);
        const b = Math.pow(pixels[i + 2] / 255, 2.2);
        const sr = Math.max(0, Math.min(1, row0[0] * r + row0[1] * g + row0[2] * b));
        const sg = Math.max(0, Math.min(1, row1[0] * r + row1[1] * g + row1[2] * b));
        const sb = Math.max(0, Math.min(1, row2[0] * r + row2[1] * g + row2[2] * b));
        pixels[i] = Math.round(Math.pow(sr, 1 / 2.2) * 255);
        pixels[i + 1] = Math.round(Math.pow(sg, 1 / 2.2) * 255);
        pixels[i + 2] = Math.round(Math.pow(sb, 1 / 2.2) * 255);
      }
      setDisplayUri(encodeToDataUri(pixels, rawImage.width, rawImage.height));
    } catch (e) {
      console.warn('[CVDGallery] filter error:', e);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>
      <SafeAreaView style={{ flex: 1 }}>
        <View style={styles.camTopBar}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="arrow-back" size={24} color="#FFF" />
          </TouchableOpacity>
          <Text style={{ color: '#FFF', fontWeight: 'bold' }}>Gallery Analysis</Text>
          <TouchableOpacity onPress={pickImage}>
            <Ionicons name="add-circle" size={28} color="#FFF" />
          </TouchableOpacity>
        </View>

        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          {displayUri ? (
            <View style={{ width, height: width * 1.3 }}>
              <Image source={{ uri: displayUri }} style={{ width: '100%', height: '100%', resizeMode: 'contain' }} />
              {processing && (
                <View style={[StyleSheet.absoluteFill, { justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.4)' }]}>
                  <ActivityIndicator size="large" color="#FFF" />
                  <Text style={{ color: '#FFF', marginTop: 10 }}>Applying simulation...</Text>
                </View>
              )}
            </View>
          ) : (
            <TouchableOpacity onPress={pickImage} style={{ alignItems: 'center' }}>
              <Ionicons name="images-outline" size={60} color="#555" />
              <Text style={{ color: '#777', marginTop: 10 }}>Tap to pick an image</Text>
            </TouchableOpacity>
          )}
        </View>

        {originalUri && (
          <View style={{ flexDirection: 'row', justifyContent: 'center', paddingBottom: 30, gap: 10 }}>
            {['Off', 'Protan', 'Deutan', 'Tritan'].map((m) => (
              <TouchableOpacity key={m} onPress={() => applyFilter(m)} disabled={processing} style={{ backgroundColor: mode === m ? COLORS.primary : '#333', padding: 10, borderRadius: 20, paddingHorizontal: 20, opacity: processing ? 0.5 : 1 }}>
                <Text style={{ color: '#FFF', fontWeight: 'bold' }}>{m}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </SafeAreaView>
    </View>
  );
}
