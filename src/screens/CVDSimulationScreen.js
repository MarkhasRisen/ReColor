import { Canvas, Image as SkiaImage, RuntimeShader, useCanvasRef, useImage } from '@shopify/react-native-skia';
import { Ionicons } from '@expo/vector-icons';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, Alert, Dimensions, SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { useIsFocused } from '@react-navigation/native';
import { getCVDRows } from '../../tensorHelper';
import { ScreenErrorBoundary } from '../utils/logger';
import { CVD_EFFECT } from '../utils/constants';
import ModeSelector from '../components/ModeSelector';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

const { width, height: screenHeight } = Dimensions.get('window');

function CVDSimulationScreenInner({ navigation, route }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const isFocused = useIsFocused();
  const [cameraPosition, setCameraPosition] = useState('back');
  const [cvdType, setCvdType] = useState(route?.params?.initialCvdType || 'Protan');
  const [showModal, setShowModal] = useState(false);
  const [frozen, setFrozen] = useState(false);
  const [frozenUri, setFrozenUri] = useState(null);
  const [processing, setProcessing] = useState(false);

  const cameraRef = useRef(null);
  const isMountedRef = useRef(true);
  const canvasRef = useCanvasRef();
  const device = useCameraDevice(cameraPosition);
  const skImage = useImage(frozenUri);
  const cvdUniforms = useMemo(() => getCVDRows(cvdType), [cvdType]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => { isMountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, []);

  useEffect(() => {
    const unsub = navigation.addListener('beforeRemove', () => setFrozen(true));
    return unsub;
  }, [navigation]);

  const handleFreeze = useCallback(async () => {
    if (!cameraRef.current) return;
    setProcessing(true);
    try {
      const photo = await cameraRef.current.takePhoto({ qualityPrioritization: 'quality', enableShutterSound: false });
      if (!photo?.path) throw new Error('takePhoto returned no path');
      const fileUri = `file://${photo.path}`;
      const SIM_MAX = 1040;
      const resized = await ImageManipulator.manipulateAsync(
        fileUri,
        [{ resize: photo.width >= photo.height ? { width: Math.min(SIM_MAX, photo.width) } : { height: Math.min(SIM_MAX, photo.height) } }],
        { format: ImageManipulator.SaveFormat.JPEG, compress: 0.92 },
      );
      if (!isMountedRef.current) return;
      setFrozenUri(resized.uri);
      setFrozen(true);
      setProcessing(false);
    } catch (e) {
      if (isMountedRef.current) { setProcessing(false); Alert.alert('Error', `Capture failed: ${e?.message || 'unknown error'}`); }
    }
  }, []);

  const handleReset = useCallback(() => { setFrozen(false); setFrozenUri(null); setProcessing(false); }, []);

  const handleSave = useCallback(async () => {
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') { Alert.alert('Permission needed', 'Please allow access to save photos.'); return; }
      const snapshot = canvasRef.current?.makeImageSnapshot();
      if (!snapshot) { Alert.alert('Error', 'Nothing to save.'); return; }
      const b64 = snapshot.encodeToBase64();
      const tmpPath = `${FileSystem.cacheDirectory}recolor_sim_${Date.now()}.png`;
      await FileSystem.writeAsStringAsync(tmpPath, b64, { encoding: FileSystem.EncodingType.Base64 });
      await MediaLibrary.saveToLibraryAsync(tmpPath);
      Alert.alert('Saved', 'Photo saved to your gallery.');
    } catch (e) {
      Alert.alert('Error', 'Could not save photo.');
    }
  }, []);

  if (!hasPermission) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text style={{ marginBottom: 20 }}>Camera access is needed for simulation.</Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>
      {device && !frozen && (
        <Camera ref={cameraRef} style={StyleSheet.absoluteFill} device={device} isActive={isFocused && !frozen} photo />
      )}
      {frozen && skImage && (
        <Canvas ref={canvasRef} style={StyleSheet.absoluteFill}>
          <SkiaImage image={skImage} x={0} y={0} width={width} height={screenHeight} fit="cover">
            {cvdType !== 'Off' && CVD_EFFECT && <RuntimeShader source={CVD_EFFECT} uniforms={cvdUniforms} />}
          </SkiaImage>
        </Canvas>
      )}

      {(processing || (frozen && frozenUri && !skImage)) && (
        <View style={{ ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center', zIndex: 10 }}>
          <ActivityIndicator size="large" color="#FFF" />
          <Text style={{ color: '#FFF', fontSize: 14, marginTop: 12, fontWeight: '600' }}>{processing ? 'Capturing...' : 'Loading image...'}</Text>
        </View>
      )}

      <SafeAreaView style={{ flex: 1 }} pointerEvents="box-none">
        <View style={styles.camTopBar}>
          <TouchableOpacity onPress={() => { if (frozen) handleReset(); else navigation.goBack(); }} style={{ padding: 5 }}>
            <Ionicons name={frozen ? 'close' : 'arrow-back'} size={24} color="#FFF" />
          </TouchableOpacity>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Text style={{ color: '#FFF', fontWeight: 'bold', marginRight: 10, textShadowColor: 'rgba(0,0,0,0.75)', textShadowOffset: { width: -1, height: 1 }, textShadowRadius: 10 }}>
              {frozen ? (cvdType === 'Off' ? 'Original' : `${cvdType} Simulation`) : 'CVD Simulation'}
            </Text>
            {!frozen && <TouchableOpacity onPress={() => setShowModal(true)}><Ionicons name="menu" size={28} color="#FFF" /></TouchableOpacity>}
          </View>
        </View>

        <View style={{ position: 'absolute', top: 100, right: 20, alignItems: 'center', zIndex: 2 }}>
          {['Off', 'Protan', 'Deutan', 'Tritan'].map((m) => (
            <TouchableOpacity key={m} onPress={() => setCvdType(m)} disabled={processing} style={[styles.filterBtn, { backgroundColor: cvdType === m ? COLORS.primary : 'rgba(0,0,0,0.5)', marginBottom: 15, opacity: processing ? 0.4 : 1 }]}>
              <Text style={{ color: '#FFF', fontWeight: 'bold', fontSize: 10 }}>{m === 'Off' ? 'Off' : m.charAt(0)}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={{ position: 'absolute', bottom: 30, left: 0, right: 0, zIndex: 2 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' }}>
            {frozen ? (
              <>
                <TouchableOpacity onPress={handleReset}><Ionicons name="refresh" size={30} color="#FFF" /></TouchableOpacity>
                <View style={{ width: 70 }} />
                <TouchableOpacity onPress={handleSave} disabled={!skImage}>
                  <Ionicons name="download-outline" size={30} color={!skImage ? '#666' : '#FFF'} />
                </TouchableOpacity>
              </>
            ) : (
              <>
                <TouchableOpacity onPress={() => setCameraPosition((p) => (p === 'back' ? 'front' : 'back'))}>
                  <Ionicons name="camera-reverse" size={30} color="#FFF" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.shutterBtn} onPress={handleFreeze}>
                  <View style={{ width: 60, height: 60, borderRadius: 30, backgroundColor: '#FFF', justifyContent: 'center', alignItems: 'center' }}>
                    <Ionicons name="snow" size={24} color="#333" />
                  </View>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => navigation.navigate('CVDGallery')}>
                  <Ionicons name="images" size={30} color="#FFF" />
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>

        <ModeSelector visible={showModal} onClose={() => setShowModal(false)} navigation={navigation} currentMode="Simulation" />
      </SafeAreaView>
    </View>
  );
}

export default function CVDSimulationScreen(props) {
  return (
    <ScreenErrorBoundary navigation={props.navigation}>
      <CVDSimulationScreenInner {...props} />
    </ScreenErrorBoundary>
  );
}
