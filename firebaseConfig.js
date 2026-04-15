// firebaseConfig.js
import AsyncStorage from '@react-native-async-storage/async-storage';
import { initializeApp } from 'firebase/app';
import {
    createUserWithEmailAndPassword,
    getReactNativePersistence,
    initializeAuth,
    onAuthStateChanged,
    signInWithEmailAndPassword,
    signOut
} from 'firebase/auth';
import {
    addDoc,
    collection,
    initializeFirestore,
    persistentLocalCache,
    serverTimestamp
} from 'firebase/firestore';

// --- 1. PASTE YOUR KEYS HERE ---
// (Get these from Firebase Console > Project Settings > General > Your Apps > SDK Setup)
const firebaseConfig = {
  apiKey: "AIzaSyBf6p-ZwEhzzoCzaxddhR3Of0l_5ws034c",
  authDomain: "recolor-e42b1.firebaseapp.com",
  projectId: "recolor-e42b1",
  storageBucket: "recolor-e42b1.firebasestorage.app",
  messagingSenderId: "912342727884",
  appId: "1:912342727884:web:2917497763f9703971d669"
};

// --- 2. INITIALIZE APP ---
const app = initializeApp(firebaseConfig);

// --- 3. AUTHENTICATION (With Persistence) ---
// This ensures the user stays logged in even if they close the app.
// Required for "User Profiling" (Slide 55).
const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(AsyncStorage)
});

// --- 4. DATABASE (OFFLINE FIRST) ---
// CRITICAL: We use 'persistentLocalCache' to satisfy the 
// "Connectivity and Offline Behavior" requirement in your manuscript.
const db = initializeFirestore(app, {
  localCache: persistentLocalCache()
});

// --- 5. HELPER: DUAL WRITE FOR PRIVACY ---
// This satisfies the Ethical Consideration for Data Privacy (Slide 56).
// It splits data into two streams: Private (User) and Anonymized (Research).
const saveExamResult = async (userId, score, diagnosis, severity, total = 14) => {
  try {
    // WRITE 1: Private User History (Linked to User ID)
    // Visible only to the user in the "History" tab.
    await addDoc(collection(db, "users", userId, "history"), {
      score: score,
      total: total,
      diagnosis: diagnosis,
      severity: severity,
      date: serverTimestamp()
    });

    // WRITE 2: Anonymized Research Data (No Names/Emails)
    // Visible to PERI Researchers via the Admin Dashboard.
    await addDoc(collection(db, "research_data_anonymized"), {
      diagnosis: diagnosis,
      severity: severity,
      score: score,
      total: total,
      device: "Mobile_Client", // Generic device tag
      timestamp: serverTimestamp()
    });
    
    console.log("Dual write successful: Private + Research");
    return true;
  } catch (error) {
    console.error("Save failed", error);
    return false;
  }
};

// Export these to be used in App.js
export {
    auth, createUserWithEmailAndPassword, db, onAuthStateChanged,
    saveExamResult, signInWithEmailAndPassword, signOut
};
