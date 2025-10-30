/**
 * Firebase Sync Service
 * Handles synchronization between local AsyncStorage and Firebase Firestore
 * Using Firebase Web SDK v10
 */

import { firestore, auth } from './firebase';
import { doc, setDoc, getDoc, collection, query, orderBy, limit as limitQuery, getDocs, deleteDoc, onSnapshot, serverTimestamp } from 'firebase/firestore';
import { getVisionProfile, saveVisionProfile } from './profileStorage';
import { VisionProfile } from './imageProcessing';

/**
 * Save vision profile to Firestore
 * @param profile Vision profile to save
 * @returns Promise<void>
 */
export async function syncProfileToFirestore(profile: VisionProfile): Promise<void> {
  try {
    const user = auth.currentUser;
    if (!user) {
      console.warn('No authenticated user, skipping Firestore sync');
      return;
    }

    const profileRef = doc(firestore, 'users', user.uid, 'profiles', 'current');
    await setDoc(profileRef, {
      ...profile,
      userId: user.uid,
      syncedAt: serverTimestamp(),
    });

    console.log('Profile synced to Firestore successfully');
  } catch (error) {
    console.error('Error syncing profile to Firestore:', error);
    throw error;
  }
}

/**
 * Load vision profile from Firestore
 * @returns Promise<VisionProfile | null>
 */
export async function loadProfileFromFirestore(): Promise<VisionProfile | null> {
  try {
    const user = auth.currentUser;
    if (!user) {
      console.warn('No authenticated user, cannot load from Firestore');
      return null;
    }

    const profileRef = doc(firestore, 'users', user.uid, 'profiles', 'current');
    const docSnap = await getDoc(profileRef);

    if (!docSnap.exists()) {
      console.log('No profile found in Firestore');
      return null;
    }

    const data = docSnap.data() as VisionProfile;
    console.log('Profile loaded from Firestore:', data);
    return data;
  } catch (error) {
    console.error('Error loading profile from Firestore:', error);
    throw error;
  }
}

/**
 * Save test results to Firestore
 * @param testType Type of test (e.g., 'ishihara-quick', 'ishihara-comprehensive')
 * @param results Test results data
 * @returns Promise<void>
 */
export async function saveTestResults(testType: string, results: any): Promise<void> {
  try {
    const user = auth.currentUser;
    if (!user) {
      console.warn('No authenticated user, skipping test results sync');
      return;
    }

    const resultsRef = collection(firestore, 'users', user.uid, 'testResults');
    await setDoc(doc(resultsRef), {
      testType,
      results,
      timestamp: serverTimestamp(),
      userId: user.uid,
    });

    console.log('Test results saved to Firestore');
  } catch (error) {
    console.error('Error saving test results:', error);
    throw error;
  }
}

/**
 * Get test history from Firestore
 * @param limit Maximum number of results to fetch
 * @returns Promise<any[]>
 */
export async function getTestHistory(limit: number = 10): Promise<any[]> {
  try {
    const user = auth.currentUser;
    if (!user) {
      console.warn('No authenticated user, cannot fetch test history');
      return [];
    }

    const resultsRef = collection(firestore, 'users', user.uid, 'testResults');
    const q = query(resultsRef, orderBy('timestamp', 'desc'), limitQuery(limit));
    const querySnapshot = await getDocs(q);

    return querySnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data(),
    }));
  } catch (error) {
    console.error('Error fetching test history:', error);
    throw error;
  }
}

/**
 * Sync local profile to Firestore on app startup
 * @returns Promise<void>
 */
export async function syncLocalProfileOnStartup(): Promise<void> {
  try {
    const user = auth.currentUser;
    if (!user) {
      console.log('No user authenticated, skipping sync on startup');
      return;
    }

    // Try to load from Firestore first
    const cloudProfile = await loadProfileFromFirestore();
    
    if (cloudProfile) {
      // Use cloud profile if it exists
      await saveVisionProfile(cloudProfile);
      console.log('Loaded profile from cloud on startup');
    } else {
      // Otherwise sync local profile to cloud
      const localProfile = await getVisionProfile();
      if (localProfile) {
        await syncProfileToFirestore(localProfile);
        console.log('Synced local profile to cloud on startup');
      }
    }
  } catch (error) {
    console.error('Error syncing on startup:', error);
  }
}

/**
 * Delete user data from Firestore (for account deletion)
 * @returns Promise<void>
 */
export async function deleteUserData(): Promise<void> {
  try {
    const user = auth.currentUser;
    if (!user) {
      console.warn('No authenticated user');
      return;
    }

    // Delete profile
    const profileRef = doc(firestore, 'users', user.uid, 'profiles', 'current');
    await deleteDoc(profileRef);

    // Delete test results
    const resultsRef = collection(firestore, 'users', user.uid, 'testResults');
    const querySnapshot = await getDocs(resultsRef);
    const deletePromises = querySnapshot.docs.map(doc => deleteDoc(doc.ref));
    await Promise.all(deletePromises);

    console.log('User data deleted from Firestore');
  } catch (error) {
    console.error('Error deleting user data:', error);
    throw error;
  }
}

/**
 * Listen to profile changes in real-time
 * @param callback Function to call when profile changes
 * @returns Unsubscribe function
 */
export function listenToProfileChanges(callback: (profile: VisionProfile | null) => void): () => void {
  const user = auth.currentUser;
  if (!user) {
    console.warn('No authenticated user, cannot listen to profile changes');
    return () => {};
  }

  const profileRef = doc(firestore, 'users', user.uid, 'profiles', 'current');
  
  return onSnapshot(profileRef, (docSnap) => {
    if (docSnap.exists()) {
      callback(docSnap.data() as VisionProfile);
    } else {
      callback(null);
    }
  }, (error) => {
    console.error('Error listening to profile changes:', error);
  });
}
