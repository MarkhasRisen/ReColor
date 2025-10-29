/**
 * Firebase Sync Service
 * Handles synchronization between local AsyncStorage and Firebase Firestore
 */

import { firestore, auth } from './firebase';
import { getVisionProfile, saveVisionProfile } from './profileStorage';
import { VisionProfile } from './imageProcessing';

/**
 * Save vision profile to Firestore
 * @param profile Vision profile to save
 * @returns Promise<void>
 */
export async function syncProfileToFirestore(profile: VisionProfile): Promise<void> {
  try {
    const user = auth().currentUser;
    if (!user) {
      console.warn('No authenticated user, skipping Firestore sync');
      return;
    }

    await firestore()
      .collection('users')
      .doc(user.uid)
      .collection('profiles')
      .doc('current')
      .set({
        ...profile,
        userId: user.uid,
        syncedAt: firestore.FieldValue.serverTimestamp(),
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
    const user = auth().currentUser;
    if (!user) {
      console.warn('No authenticated user, cannot load from Firestore');
      return null;
    }

    const doc = await firestore()
      .collection('users')
      .doc(user.uid)
      .collection('profiles')
      .doc('current')
      .get();

    if (!doc.exists) {
      console.log('No profile found in Firestore');
      return null;
    }

    const data = doc.data();
    return {
      cvdType: data?.cvdType || 'normal',
      severity: data?.severity || 0.7,
      userId: data?.userId || user.uid,
      timestamp: data?.timestamp || Date.now(),
    };
  } catch (error) {
    console.error('Error loading profile from Firestore:', error);
    throw error;
  }
}

/**
 * Save test results to Firestore
 * @param testType Type of test (quick, comprehensive, survey)
 * @param results Test results object
 */
export async function saveTestResults(
  testType: 'quick' | 'comprehensive' | 'survey',
  results: any
): Promise<void> {
  try {
    const user = auth().currentUser;
    if (!user) {
      console.warn('No authenticated user, skipping test results sync');
      return;
    }

    await firestore()
      .collection('users')
      .doc(user.uid)
      .collection('testResults')
      .add({
        testType,
        results,
        timestamp: firestore.FieldValue.serverTimestamp(),
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
 * @param limit Number of results to fetch
 * @returns Promise<any[]>
 */
export async function getTestHistory(limit: number = 10): Promise<any[]> {
  try {
    const user = auth().currentUser;
    if (!user) {
      console.warn('No authenticated user');
      return [];
    }

    const snapshot = await firestore()
      .collection('users')
      .doc(user.uid)
      .collection('testResults')
      .orderBy('timestamp', 'desc')
      .limit(limit)
      .get();

    return snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data(),
    }));
  } catch (error) {
    console.error('Error fetching test history:', error);
    return [];
  }
}

/**
 * Sync local profile to Firestore on app startup
 * Only syncs if user is authenticated
 */
export async function syncLocalProfileOnStartup(): Promise<void> {
  try {
    const user = auth().currentUser;
    if (!user) {
      console.log('No user authenticated, skipping startup sync');
      return;
    }

    // Get local profile
    const localProfile = await getVisionProfile();
    
    if (localProfile) {
      // Sync to Firestore
      await syncProfileToFirestore(localProfile);
      console.log('Local profile synced to Firestore on startup');
    } else {
      // Try to load from Firestore
      const firestoreProfile = await loadProfileFromFirestore();
      if (firestoreProfile) {
        // Save to local storage
        await saveVisionProfile(firestoreProfile);
        console.log('Profile loaded from Firestore to local storage');
      }
    }
  } catch (error) {
    console.error('Error during startup sync:', error);
  }
}

/**
 * Delete user data from Firestore (for account deletion)
 */
export async function deleteUserData(): Promise<void> {
  try {
    const user = auth().currentUser;
    if (!user) {
      throw new Error('No authenticated user');
    }

    // Delete profiles
    const profilesSnapshot = await firestore()
      .collection('users')
      .doc(user.uid)
      .collection('profiles')
      .get();

    const profileDeletePromises = profilesSnapshot.docs.map(doc => doc.ref.delete());

    // Delete test results
    const testResultsSnapshot = await firestore()
      .collection('users')
      .doc(user.uid)
      .collection('testResults')
      .get();

    const testResultsDeletePromises = testResultsSnapshot.docs.map(doc => doc.ref.delete());

    // Delete user document
    await Promise.all([
      ...profileDeletePromises,
      ...testResultsDeletePromises,
      firestore().collection('users').doc(user.uid).delete(),
    ]);

    console.log('User data deleted from Firestore');
  } catch (error) {
    console.error('Error deleting user data:', error);
    throw error;
  }
}

/**
 * Listen to profile changes in Firestore (real-time sync)
 * @param callback Function to call when profile changes
 * @returns Unsubscribe function
 */
export function listenToProfileChanges(
  callback: (profile: VisionProfile | null) => void
): () => void {
  const user = auth().currentUser;
  if (!user) {
    console.warn('No authenticated user for real-time sync');
    return () => {};
  }

  const unsubscribe = firestore()
    .collection('users')
    .doc(user.uid)
    .collection('profiles')
    .doc('current')
    .onSnapshot(
      (doc) => {
        if (doc.exists) {
          const data = doc.data();
          const profile: VisionProfile = {
            cvdType: data?.cvdType || 'normal',
            severity: data?.severity || 0.7,
            userId: data?.userId || user.uid,
            timestamp: data?.timestamp || Date.now(),
          };
          callback(profile);
        } else {
          callback(null);
        }
      },
      (error) => {
        console.error('Error listening to profile changes:', error);
      }
    );

  return unsubscribe;
}
