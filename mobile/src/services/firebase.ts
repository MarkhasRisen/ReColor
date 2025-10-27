import firebase from "@react-native-firebase/app";
import auth from "@react-native-firebase/auth";
import firestore from "@react-native-firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAOoHkeZ5ZZtkSX8SkZI9JP1UjZGiwrR78",
  projectId: "recolor-7d7fd",
  storageBucket: "recolor-7d7fd.firebasestorage.app",
  appId: "1:50383676625:android:91cd747047b39ef8bed1f0",
};

if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}

export { auth, firestore };
export default firebase;
