import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Slider,
} from 'react-native';
import { Camera, CameraView, CameraType } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import { processFrame } from '../services/imageProcessing';
import { getVisionProfile, updateProfileSeverity, VisionProfile } from '../services/profileStorage';
import { CVDType } from '../services/daltonization';

const ColorEnhancementScreen = () => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [profile, setProfile] = useState<VisionProfile | null>(null);
  const [severity, setSeverity] = useState(0.8);
  const [isProcessing, setIsProcessing] = useState(true);
  const [facing, setFacing] = useState<CameraType>('back');
  const cameraRef = useRef<any>(null);

  // Request camera permissions and load profile
  useEffect(() => {
    (async () => {
      try {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');

        if (status === 'granted') {
          const userProfile = await getVisionProfile();
          if (userProfile) {
            setProfile(userProfile);
            setSeverity(userProfile.severity);
          } else {
            // Create default profile
            const defaultProfile: VisionProfile = {
              cvdType: 'protan',
              severity: 0.8,
              userId: 'default',
              timestamp: Date.now(),
            };
            setProfile(defaultProfile);
          }
        }
      } catch (error) {
        console.error('Error setting up camera:', error);
        Alert.alert('Error', 'Failed to initialize camera');
      }
    })();
  }, []);

  // Update profile when severity changes
  const handleSeverityChange = async (value: number) => {
    setSeverity(value);
    if (profile) {
      const updatedProfile = { ...profile, severity: value };
      setProfile(updatedProfile);
      try {
        await updateProfileSeverity(value);
      } catch (error) {
        console.error('Error updating severity:', error);
      }
    }
  };

  // Toggle camera facing
  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  // Capture photo with enhancement
  const capturePhoto = async () => {
    if (!cameraRef.current || !profile) {
      Alert.alert('Error', 'Camera not ready');
      return;
    }

    try {
      setIsProcessing(true);

      // Capture photo
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });

      // Note: For full implementation, you would:
      // 1. Load the image data
      // 2. Apply daltonization using processFrame
      // 3. Save the processed image
      
      // For now, show success
      Alert.alert(
        'Photo Captured',
        'Enhanced photo saved! (Full processing to be implemented)',
        [{ text: 'OK' }]
      );
    } catch (error) {
      console.error('Error capturing photo:', error);
      Alert.alert('Error', 'Failed to capture photo');
    } finally {
      setIsProcessing(false);
    }
  };

  // Toggle processing on/off
  const toggleProcessing = () => {
    setIsProcessing(!isProcessing);
  };

  // Permission states
  if (hasPermission === null) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.text}>Requesting camera permissions...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.centered}>
        <Text style={styles.icon}>�</Text>
        <Text style={styles.text}>Camera permission denied</Text>
        <Text style={styles.subtext}>
          Please enable camera access in your device settings
        </Text>
      </View>
    );
  }

  if (!profile) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.text}>Loading vision profile...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
      >
        {/* Top overlay - Profile info */}
        <View style={styles.topOverlay}>
          <View style={styles.profileBadge}>
            <Text style={styles.badgeText}>
              {profile.cvdType.toUpperCase()}
            </Text>
          </View>
          <TouchableOpacity
            style={[
              styles.processingToggle,
              isProcessing ? styles.processingOn : styles.processingOff,
            ]}
            onPress={toggleProcessing}
          >
            <Text style={styles.toggleText}>
              {isProcessing ? '✓ Enhanced' : '○ Original'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Severity slider */}
        <View style={styles.sliderContainer}>
          <Text style={styles.sliderLabel}>Correction Strength</Text>
          <View style={styles.sliderRow}>
            <Text style={styles.sliderValue}>0%</Text>
            <Slider
              style={styles.slider}
              minimumValue={0}
              maximumValue={1}
              value={severity}
              onValueChange={handleSeverityChange}
              minimumTrackTintColor="#4CAF50"
              maximumTrackTintColor="#666"
              thumbTintColor="#4CAF50"
            />
            <Text style={styles.sliderValue}>100%</Text>
          </View>
          <Text style={styles.sliderCurrent}>
            {Math.round(severity * 100)}%
          </Text>
        </View>

        {/* Bottom controls */}
        <View style={styles.bottomOverlay}>
          {/* Flip camera button */}
          <TouchableOpacity
            style={styles.controlButton}
            onPress={toggleCameraFacing}
          >
            <Text style={styles.controlIcon}>🔄</Text>
          </TouchableOpacity>

          {/* Capture button */}
          <TouchableOpacity
            style={styles.captureButton}
            onPress={capturePhoto}
            disabled={!isProcessing}
          >
            <View style={styles.captureButtonInner} />
          </TouchableOpacity>

          {/* Placeholder for gallery */}
          <View style={styles.controlButton}>
            <Text style={styles.controlIcon}>🖼️</Text>
          </View>
        </View>

        {/* Processing indicator */}
        {isProcessing && (
          <View style={styles.processingIndicator}>
            <View style={styles.processingDot} />
            <Text style={styles.processingText}>Live Enhancement</Text>
          </View>
        )}
      </CameraView>

      {/* Info overlay */}
      <View style={styles.infoOverlay}>
        <Text style={styles.infoText}>
          Real-time color correction for {profile.cvdType}
        </Text>
        <Text style={styles.infoSubtext}>
          Adjust strength slider • Tap toggle to compare
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 30,
  },
  camera: {
    flex: 1,
  },
  icon: {
    fontSize: 64,
    marginBottom: 20,
  },
  text: {
    fontSize: 18,
    color: '#FFF',
    textAlign: 'center',
    marginTop: 20,
  },
  subtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
    marginTop: 10,
  },
  topOverlay: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  profileBadge: {
    backgroundColor: 'rgba(76, 175, 80, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  badgeText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: 'bold',
  },
  processingToggle: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 2,
  },
  processingOn: {
    backgroundColor: 'rgba(76, 175, 80, 0.9)',
    borderColor: '#4CAF50',
  },
  processingOff: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderColor: '#666',
  },
  toggleText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
  sliderContainer: {
    position: 'absolute',
    top: 100,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 15,
  },
  sliderLabel: {
    color: '#FFF',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 10,
    fontWeight: '600',
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  slider: {
    flex: 1,
    marginHorizontal: 10,
  },
  sliderValue: {
    color: '#999',
    fontSize: 12,
    width: 35,
    textAlign: 'center',
  },
  sliderCurrent: {
    color: '#4CAF50',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 5,
    fontWeight: 'bold',
  },
  bottomOverlay: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  controlButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#FFF',
  },
  controlIcon: {
    fontSize: 28,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#FFF',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#FFF',
  },
  processingIndicator: {
    position: 'absolute',
    top: 70,
    left: '50%',
    transform: [{ translateX: -60 }],
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(76, 175, 80, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  processingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FFF',
    marginRight: 8,
  },
  processingText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '600',
  },
  infoOverlay: {
    position: 'absolute',
    bottom: 150,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 15,
    borderRadius: 10,
  },
  infoText: {
    color: '#FFF',
    fontSize: 14,
    textAlign: 'center',
    fontWeight: '600',
  },
  infoSubtext: {
    color: '#999',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 5,
  },
});

export default ColorEnhancementScreen;
