import { initializeApp, getApps } from 'firebase/app';
import { getAuth, initializeAuth, getReactNativePersistence } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import AsyncStorage from '@react-native-async-storage/async-storage';

const firebaseConfig = {
  apiKey: "AIzaSyAOoHkeZ5ZZtkSX8SkZI9JP1UjZGiwrR78",
  authDomain: "recolor-7d7fd.firebaseapp.com",
  projectId: "recolor-7d7fd",
  storageBucket: "recolor-7d7fd.firebasestorage.app",
  messagingSenderId: "50383676625",
  appId: "1:50383676625:android:91cd747047b39ef8bed1f0",
};

// Initialize Firebase
let app;
if (getApps().length === 0) {
  app = initializeApp(firebaseConfig);
} else {
  app = getApps()[0];
}

// Initialize Auth with AsyncStorage persistence
const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(AsyncStorage)
});

// Initialize Firestore
const firestore = getFirestore(app);

export { auth, firestore };
export default app;
