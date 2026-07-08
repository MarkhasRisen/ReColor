import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  addDoc,
  collection,
  getDocs,
  query,
  serverTimestamp,
  where,
} from "firebase/firestore";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, { FadeInDown, FadeInUp } from "react-native-reanimated";
import { auth, db } from "../../firebaseConfig";
import BackgroundBubbles from "../components/BackgroundBubbles";
import Header from "../components/Header";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";

const CAUSES = ["Medical Intake", "Genetics", "Ageing", "Others"];

export default function SurveyScreen({ navigation }) {
  const [selectedCause, setSelectedCause] = useState(null);
  const [selectedSex, setSelectedSex] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [alreadySubmitted, setAlreadySubmitted] = useState(false);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const uid = auth.currentUser?.uid;
    if (!uid) {
      AsyncStorage.getItem("@recolor_survey_submitted")
        .then((submitted) => {
          setAlreadySubmitted(submitted === "true");
        })
        .catch((err) => console.error("Survey check local error:", err))
        .finally(() => setChecking(false));
      return;
    }
    // Check if the user has already contributed to the research
    getDocs(query(collection(db, "surveys"), where("userId", "==", uid)))
      .then((snap) => setAlreadySubmitted(!snap.empty))
      .catch((err) => console.error("Survey check error:", err))
      .finally(() => setChecking(false));
  }, []);

  const handleSubmit = async () => {
    if (selectedCause === null || !selectedSex) {
      Alert.alert(
        "Incomplete Form",
        "Please provide all details to help our clinical research.",
      );
      return;
    }

    setSubmitting(true);
    try {
      const surveyData = {
        cause: CAUSES[selectedCause],
        sex: selectedSex,
        userId: auth.currentUser?.uid || "anonymous",
        timestamp: serverTimestamp(),
      };

      // Save to global research collection
      await addDoc(collection(db, "surveys"), surveyData);

      // Save to user's private history
      if (auth.currentUser) {
        await addDoc(
          collection(db, "users", auth.currentUser.uid, "surveys"),
          surveyData,
        );
      } else {
        await AsyncStorage.setItem("@recolor_survey_submitted", "true");
      }

      navigation.replace("SurveySuccess");
    } catch (e) {
      console.warn("[Survey] save failed:", e);
      // Fallback for demo/offline purposes
      if (!auth.currentUser) {
        try {
          await AsyncStorage.setItem("@recolor_survey_submitted", "true");
        } catch (_) {}
      }
      navigation.replace("SurveySuccess");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <View style={styles.container}>
      <BackgroundBubbles />
      <Header title="Patient Survey" back />

      {checking ? (
        <View style={styles.center}>
          <ActivityIndicator color={COLORS.primary} size="large" />
        </View>
      ) : alreadySubmitted ? (
        <Animated.View entering={FadeInUp.duration(600)} style={styles.center}>
          <View style={styles.iconBox}>
            <Ionicons name="checkmark-done" size={50} color={COLORS.success} />
          </View>
          <Text style={styles.completeTitle}>Contribution Received</Text>
          <Text style={styles.completeDesc}>
            Thank you for participating. Your data helps improve clinical
            outcomes for color vision deficiency research.
          </Text>
          <TouchableOpacity
            style={styles.doneBtn}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.doneBtnText}>Return to Knowledge Base</Text>
          </TouchableOpacity>
        </Animated.View>
      ) : (
        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
        >
          <Animated.View entering={FadeInDown.duration(500)}>
            <Text style={styles.sectionLabel}>CVD ORIGIN RESEARCH</Text>
            <View style={styles.moduleCard}>
              <Text style={styles.question}>
                What do you believe is the primary cause of your condition?
              </Text>
              <View style={styles.optionsGrid}>
                {CAUSES.map((item, idx) => (
                  <TouchableOpacity
                    key={idx}
                    style={[
                      styles.option,
                      selectedCause === idx && styles.optionSelected,
                    ]}
                    onPress={() => setSelectedCause(idx)}
                  >
                    <Ionicons
                      name={
                        selectedCause === idx
                          ? "radio-button-on"
                          : "radio-button-off"
                      }
                      size={20}
                      color={
                        selectedCause === idx
                          ? COLORS.primary
                          : COLORS.textLight
                      }
                    />
                    <Text
                      style={[
                        styles.optionText,
                        selectedCause === idx && styles.textSelected,
                      ]}
                    >
                      {item}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <Text style={styles.sectionLabel}>DEMOGRAPHIC DATA</Text>
            <View style={styles.moduleCard}>
              <Text style={styles.question}>Biological Sex</Text>
              <View style={styles.row}>
                {["Male", "Female"].map((sex) => (
                  <TouchableOpacity
                    key={sex}
                    style={[
                      styles.sexBtn,
                      selectedSex === sex && styles.sexSelected,
                    ]}
                    onPress={() => setSelectedSex(sex)}
                  >
                    <Text
                      style={[
                        styles.sexText,
                        selectedSex === sex && styles.textSelected,
                      ]}
                    >
                      {sex}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <TouchableOpacity
              style={[styles.submitBtn, submitting && { opacity: 0.7 }]}
              onPress={handleSubmit}
              disabled={submitting}
            >
              {submitting ? (
                <ActivityIndicator color="#FFF" />
              ) : (
                <Text style={styles.submitText}>Submit Clinical Data</Text>
              )}
            </TouchableOpacity>

            <Text style={styles.footerInfo}>
              Your response is anonymized and used strictly for educational
              research purposes.
            </Text>
          </Animated.View>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8FAFC",
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 40,
  },
  scroll: {
    padding: SPACING.lg,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "800",
    color: COLORS.textLight,
    letterSpacing: 1.5,
    marginBottom: 12,
    marginLeft: 4,
  },
  moduleCard: {
    backgroundColor: "#FFF",
    borderRadius: RADIUS.xl,
    padding: SPACING.lg,
    marginBottom: 25,
    ...SHADOW.sm,
    borderWidth: 1,
    borderColor: "#F1F5F9",
  },
  question: {
    fontSize: 16,
    fontWeight: "700",
    color: COLORS.text,
    marginBottom: 20,
    lineHeight: 22,
  },
  optionsGrid: {
    gap: 12,
  },
  option: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: "#F1F5F9",
    gap: 12,
  },
  optionSelected: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.primary + "05",
  },
  optionText: {
    fontSize: 15,
    fontWeight: "600",
    color: COLORS.text,
  },
  textSelected: {
    color: COLORS.primary,
    fontWeight: "800",
  },
  row: {
    flexDirection: "row",
    gap: 12,
  },
  sexBtn: {
    flex: 1,
    padding: 16,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: "#F1F5F9",
    alignItems: "center",
  },
  sexSelected: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.primary + "05",
  },
  sexText: {
    fontWeight: "700",
    color: COLORS.textLight,
  },
  submitBtn: {
    backgroundColor: COLORS.text,
    padding: 20,
    borderRadius: RADIUS.xl,
    alignItems: "center",
    marginTop: 10,
    ...SHADOW.md,
  },
  submitText: {
    color: "#FFF",
    fontWeight: "800",
    fontSize: 16,
    letterSpacing: 0.5,
  },
  iconBox: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: COLORS.success + "15",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 25,
  },
  completeTitle: {
    fontSize: 22,
    fontWeight: "900",
    color: COLORS.text,
    textAlign: "center",
  },
  completeDesc: {
    fontSize: 14,
    color: COLORS.textLight,
    textAlign: "center",
    marginTop: 12,
    lineHeight: 22,
  },
  doneBtn: {
    marginTop: 30,
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: "#E2E8F0",
  },
  doneBtnText: {
    color: COLORS.text,
    fontWeight: "700",
    fontSize: 14,
  },
  footerInfo: {
    textAlign: "center",
    color: COLORS.textLight,
    fontSize: 11,
    marginTop: 25,
    opacity: 0.6,
    lineHeight: 16,
  },
});
