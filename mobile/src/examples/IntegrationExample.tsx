/**
 * Integration Example
 * Shows how all the on-device processing components work together
 */

import React, { useEffect, useState } from 'react';
import { View, Text, Button, Image, StyleSheet } from 'react-native';
import { daltonizePixel, identifyColor, CVDType } from '../services/daltonization';
import { processFrame } from '../services/imageProcessing';
import {
  getVisionProfile,
  saveVisionProfile,
  updateProfileSeverity,
  VisionProfile,
} from '../services/profileStorage';

// Example 1: Initialize user profile
export async function initializeUserProfile(userId: string, cvdType: CVDType) {
  const profile: VisionProfile = {
    cvdType,
    severity: 0.8,
    userId,
    timestamp: Date.now(),
  };

  await saveVisionProfile(profile);
  console.log('Profile initialized:', profile);
}

// Example 2: Process a single color
export function processSingleColor() {
  const red = [1.0, 0.0, 0.0]; // Pure red
  
  // Correct for protanopia
  const corrected = daltonizePixel(red, 'protan', 0.8);
  
  console.log('Original:', red);
  console.log('Corrected:', corrected);
  
  // Identify the color
  const colorInfo = identifyColor([255, 0, 0]);
  console.log('Color identified as:', colorInfo.name, colorInfo.hex);
}

// Example 3: Process camera frame
export async function processCameraFrame(
  frameData: Uint8ClampedArray,
  width: number,
  height: number
) {
  // Get user's profile
  const profile = await getVisionProfile();
  
  if (!profile) {
    console.error('No profile found');
    return frameData;
  }

  // Process the frame
  const processed = processFrame(frameData, width, height, profile);
  
  return processed;
}

// Example 4: Update correction strength
export async function adjustCorrectionStrength(newSeverity: number) {
  await updateProfileSeverity(newSeverity);
  console.log('Severity updated to:', newSeverity);
}

// Example 5: Complete integration component
export const IntegrationExample: React.FC = () => {
  const [profile, setProfile] = useState<VisionProfile | null>(null);
  const [severity, setSeverity] = useState(0.8);

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    const userProfile = await getVisionProfile();
    if (userProfile) {
      setProfile(userProfile);
      setSeverity(userProfile.severity);
    } else {
      // Create default profile
      await initializeUserProfile('user123', 'protan');
      loadProfile();
    }
  };

  const handleSeverityChange = async (value: number) => {
    setSeverity(value);
    await updateProfileSeverity(value);
    
    // Reload profile to confirm update
    const updated = await getVisionProfile();
    setProfile(updated);
  };

  const testColorCorrection = () => {
    const testColors = [
      { name: 'Red', rgb: [1.0, 0.0, 0.0] },
      { name: 'Green', rgb: [0.0, 1.0, 0.0] },
      { name: 'Blue', rgb: [0.0, 0.0, 1.0] },
    ];

    testColors.forEach(({ name, rgb }) => {
      const corrected = daltonizePixel(rgb, profile?.cvdType || 'normal', severity);
      console.log(`${name}: [${rgb}] → [${corrected.map(v => v.toFixed(2))}]`);
    });
  };

  if (!profile) {
    return (
      <View style={styles.container}>
        <Text>Loading profile...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Integration Example</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Current Profile</Text>
        <Text>CVD Type: {profile.cvdType}</Text>
        <Text>Severity: {Math.round(profile.severity * 100)}%</Text>
        <Text>User ID: {profile.userId}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Actions</Text>
        
        <Button
          title="Test Color Correction"
          onPress={testColorCorrection}
        />
        
        <Button
          title="Increase Severity"
          onPress={() => handleSeverityChange(Math.min(1.0, severity + 0.1))}
        />
        
        <Button
          title="Decrease Severity"
          onPress={() => handleSeverityChange(Math.max(0.0, severity - 0.1))}
        />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Usage Examples</Text>
        <Text style={styles.code}>
          {`// Initialize profile
await initializeUserProfile('user123', 'protan');

// Process single color
const corrected = daltonizePixel([1,0,0], 'protan', 0.8);

// Process camera frame
const processed = processFrame(frameData, w, h, profile);

// Adjust strength
await updateProfileSeverity(0.9);`}
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#F5F5F5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
  },
  section: {
    backgroundColor: '#FFF',
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  code: {
    fontFamily: 'monospace',
    fontSize: 12,
    backgroundColor: '#F0F0F0',
    padding: 10,
    borderRadius: 4,
    color: '#666',
  },
});

// Quick test function for debugging
export async function quickTest() {
  console.log('🧪 Running Quick Test...\n');

  // Test 1: Profile management
  console.log('Test 1: Profile Management');
  await initializeUserProfile('test-user', 'protan');
  const profile = await getVisionProfile();
  console.log('✓ Profile created:', profile?.cvdType);

  // Test 2: Color correction
  console.log('\nTest 2: Color Correction');
  const red = [1.0, 0.0, 0.0];
  const corrected = daltonizePixel(red, 'protan', 0.8);
  console.log('✓ Red corrected:', corrected.map(v => v.toFixed(2)));

  // Test 3: Color identification
  console.log('\nTest 3: Color Identification');
  const colorInfo = identifyColor([255, 0, 0]);
  console.log('✓ Color identified:', colorInfo.name, colorInfo.hex);

  // Test 4: Frame processing simulation
  console.log('\nTest 4: Frame Processing');
  const testFrame = new Uint8ClampedArray(100 * 100 * 4); // Small test frame
  for (let i = 0; i < testFrame.length; i += 4) {
    testFrame[i] = 255; // Red
    testFrame[i + 3] = 255; // Alpha
  }
  
  if (profile) {
    const processed = processFrame(testFrame, 100, 100, profile);
    console.log('✓ Frame processed:', processed.length, 'bytes');
  }

  console.log('\n✅ All quick tests completed!');
}

export default IntegrationExample;
