import AsyncStorage from "@react-native-async-storage/async-storage";
import { initializeApp } from "firebase/app";
import {
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  getReactNativePersistence,
  initializeAuth,
  onAuthStateChanged,
  sendEmailVerification,
  sendPasswordResetEmail,
  signInWithCredential,
  signInWithEmailAndPassword,
  signOut,
} from "firebase/auth";
import {
  addDoc,
  collection,
  initializeFirestore,
  persistentLocalCache,
  serverTimestamp,
} from "firebase/firestore";

// --- 1. NEW RECOLOR-DEV KEYS ---
const firebaseConfig = {
  apiKey: "AIzaSyCc_EYXVkqwSLYm67K5x-JGazLxxa-Vi7M",
  authDomain: "recolor-dev.firebaseapp.com",
  projectId: "recolor-dev",
  storageBucket: "recolor-dev.firebasestorage.app",
  messagingSenderId: "1047364142133",
  appId: "1:1047364142133:web:a0b3deb67836acf727dd7c",
  measurementId: "G-VEPWR9T85C",
};

// --- 2. INITIALIZE APP ---
const app = initializeApp(firebaseConfig);

// --- 3. AUTHENTICATION (With Persistence) ---
// This keeps you logged in even if you close the app—crucial for the "User Profiling" requirement.
const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(AsyncStorage),
});

// --- 4. DATABASE (OFFLINE FIRST) ---
// This allows the app to work without Wi-Fi and sync later—a major requirement for your thesis.
const db = initializeFirestore(app, {
  localCache: persistentLocalCache(),
});

// --- 5. HELPER: DUAL WRITE FOR PRIVACY ---
// This anonymizes data for PERI Researchers (Slide 56).
export const saveExamResult = async (
  userId,
  score,
  diagnosis,
  severity,
  total = 14,
) => {
  try {
    // WRITE 1: Private (Includes UID)[cite: 15]
    await addDoc(collection(db, "users", userId, "history"), {
      score,
      total,
      diagnosis,
      severity,
      date: serverTimestamp(),
    });

    // WRITE 2: Anonymized (NO UID - THIS IS THE "REAL SHIT" Brandon mentioned)[cite: 15, 19]
    await addDoc(collection(db, "research_data_anonymized"), {
      diagnosis,
      severity,
      score,
      total,
      device: "Mobile_Client_Node", // General identifier only
      timestamp: serverTimestamp(),
    });

    return true;
  } catch (error) {
    console.error("Save failed", error);
    return false;
  }
};

export {
  GoogleAuthProvider,
  auth,
  createUserWithEmailAndPassword,
  db,
  onAuthStateChanged,
  sendEmailVerification,
  sendPasswordResetEmail,
  signInWithCredential,
  signInWithEmailAndPassword,
  signOut
};

