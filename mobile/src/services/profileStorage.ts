/**
 * Local storage for user vision profiles
 * Stores CVD type, severity, and calibration data on device
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { CVDType } from './daltonization';

export interface VisionProfile {
  cvdType: CVDType;
  severity: number;
  userId: string;
  timestamp: number;
  calibrationData?: {
    ishiharaScore?: number;
    testType?: 'quick' | 'comprehensive';
    completedAt?: number;
  };
}

const PROFILE_KEY = '@daltonization_vision_profile';
const PROFILES_HISTORY_KEY = '@daltonization_profiles_history';

/**
 * Save the user's vision profile locally
 */
export async function saveVisionProfile(profile: VisionProfile): Promise<void> {
  try {
    await AsyncStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    
    // Also save to history
    const history = await getProfileHistory();
    history.push(profile);
    
    // Keep only last 10 profiles
    const recentHistory = history.slice(-10);
    await AsyncStorage.setItem(PROFILES_HISTORY_KEY, JSON.stringify(recentHistory));
  } catch (error) {
    console.error('Error saving vision profile:', error);
    throw new Error('Failed to save vision profile');
  }
}

/**
 * Get the current vision profile
 */
export async function getVisionProfile(): Promise<VisionProfile | null> {
  try {
    const profileJson = await AsyncStorage.getItem(PROFILE_KEY);
    
    if (!profileJson) {
      return null;
    }
    
    return JSON.parse(profileJson) as VisionProfile;
  } catch (error) {
    console.error('Error getting vision profile:', error);
    return null;
  }
}

/**
 * Get profile history for tracking changes
 */
export async function getProfileHistory(): Promise<VisionProfile[]> {
  try {
    const historyJson = await AsyncStorage.getItem(PROFILES_HISTORY_KEY);
    
    if (!historyJson) {
      return [];
    }
    
    return JSON.parse(historyJson) as VisionProfile[];
  } catch (error) {
    console.error('Error getting profile history:', error);
    return [];
  }
}

/**
 * Clear the current profile
 */
export async function clearVisionProfile(): Promise<void> {
  try {
    await AsyncStorage.removeItem(PROFILE_KEY);
  } catch (error) {
    console.error('Error clearing vision profile:', error);
    throw new Error('Failed to clear vision profile');
  }
}

/**
 * Get or create a default profile
 */
export async function getOrCreateDefaultProfile(userId: string): Promise<VisionProfile> {
  const existing = await getVisionProfile();
  
  if (existing) {
    return existing;
  }
  
  // Create default profile for normal vision
  const defaultProfile: VisionProfile = {
    cvdType: 'normal',
    severity: 0,
    userId,
    timestamp: Date.now(),
  };
  
  await saveVisionProfile(defaultProfile);
  return defaultProfile;
}

/**
 * Update profile severity (e.g., from slider adjustment)
 */
export async function updateProfileSeverity(severity: number): Promise<void> {
  const profile = await getVisionProfile();
  
  if (!profile) {
    throw new Error('No profile found to update');
  }
  
  profile.severity = Math.max(0, Math.min(1, severity)); // Clamp to [0, 1]
  profile.timestamp = Date.now();
  
  await saveVisionProfile(profile);
}

/**
 * Update profile after Ishihara test
 */
export async function updateProfileFromTest(
  cvdType: CVDType,
  severity: number,
  ishiharaScore: number,
  testType: 'quick' | 'comprehensive'
): Promise<void> {
  const profile = await getVisionProfile();
  
  const updatedProfile: VisionProfile = {
    ...profile,
    cvdType,
    severity,
    userId: profile?.userId || 'unknown',
    timestamp: Date.now(),
    calibrationData: {
      ishiharaScore,
      testType,
      completedAt: Date.now(),
    },
  };
  
  await saveVisionProfile(updatedProfile);
}

/**
 * Check if profile exists and is recent (within 30 days)
 */
export async function hasValidProfile(): Promise<boolean> {
  const profile = await getVisionProfile();
  
  if (!profile) {
    return false;
  }
  
  const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
  return profile.timestamp > thirtyDaysAgo;
}

/**
 * Export profile for backup or sharing
 */
export async function exportProfile(): Promise<string> {
  const profile = await getVisionProfile();
  
  if (!profile) {
    throw new Error('No profile to export');
  }
  
  return JSON.stringify(profile, null, 2);
}

/**
 * Import profile from backup
 */
export async function importProfile(profileJson: string): Promise<void> {
  try {
    const profile = JSON.parse(profileJson) as VisionProfile;
    
    // Validate profile structure
    if (!profile.cvdType || typeof profile.severity !== 'number') {
      throw new Error('Invalid profile format');
    }
    
    await saveVisionProfile(profile);
  } catch (error) {
    console.error('Error importing profile:', error);
    throw new Error('Failed to import profile');
  }
}
