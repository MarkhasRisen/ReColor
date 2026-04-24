import { Ionicons } from "@expo/vector-icons";
import * as Brightness from "expo-brightness";
import * as Haptics from "expo-haptics";
import * as Speech from "expo-speech";
import { MotiView } from "moti";
import { useCallback, useEffect, useRef, useState } from "react";
import {
    ActivityIndicator,
    Dimensions,
    Image,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from "react-native";
import { auth } from "../../firebaseConfig";
import { saveTestSession } from "../database/client";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";
import {
    buildTestQueue,
    computeDiagnosis,
    evaluateStage1,
} from "../utils/colorLogic";

const { width } = Dimensions.get("window");
const DISPLAY_TIME = 3; // seconds each plate is shown

export default function TestScreen({ route, navigation }) {
  const { testType = "comprehensive" } = route?.params || {};
  const plateCount = testType === "comprehensive" ? 25 : 14;

  // Build queue once on mount — plate #1 always first, rest randomised
  const [queue] = useState(() => buildTestQueue(plateCount));
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [timeLeft, setTimeLeft] = useState(DISPLAY_TIME);
  const [showImage, setShowImage] = useState(true);
  const [calculating, setCalculating] = useState(false);
  const [stage, setStage] = useState(1); // Track current stage (1 or 2)
  const timerRef = useRef(null);
  const prevBrightnessRef = useRef(null);

  // Set brightness to 80% on mount, restore on unmount
  useEffect(() => {
    Brightness.requestPermissionsAsync().then(({ granted }) => {
      if (!granted) return;
      Brightness.getBrightnessAsync().then((b) => {
        prevBrightnessRef.current = b;
      });
      Brightness.setBrightnessAsync(0.8);
    });
    return () => {
      if (prevBrightnessRef.current !== null) {
        Brightness.setBrightnessAsync(prevBrightnessRef.current).catch(
          () => {},
        );
      }
    };
  }, []);

  const current = queue[index];

  // Timer — resets on each new plate
  useEffect(() => {
    setTimeLeft(DISPLAY_TIME);
    setShowImage(true);

    timerRef.current = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          setShowImage(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timerRef.current);
  }, [index]);

  const handleInput = useCallback((num) => {
    setUserInput((prev) => (prev.length < 3 ? prev + num : prev));
  }, []);

  const handleBackspace = useCallback(() => {
    setUserInput((prev) => prev.slice(0, -1));
  }, []);

  const handleNext = useCallback(
    async (inputOverride) => {
      clearInterval(timerRef.current);

      // Only accept string overrides — TouchableOpacity's onPress passes a
      // GestureResponderEvent object as the first arg, which must NOT be treated
      // as an answer (would score every plate wrong).
      const input = typeof inputOverride === "string" ? inputOverride : userInput;
      const isCorrect =
        current.category === "hidden" ? input === "" : input === current.answer;

      // Haptic feedback
      Haptics.impactAsync(
        isCorrect
          ? Haptics.ImpactFeedbackStyle.Light
          : Haptics.ImpactFeedbackStyle.Medium,
      ).catch(() => {});

      const newAnswers = [
        ...answers,
        { plate: current, userAnswer: input, isCorrect },
      ];

      // Check if we're finishing 21 answers total (end of stage 1: demo + plates 2-21)
      if (newAnswers.length === 21 && stage === 1) {
        // Evaluate stage 1
        const stage1Result = evaluateStage1(newAnswers);

        if (stage1Result.shouldProceedToStage2) {
          // Score < 14: Proceed to stage 2
          setAnswers(newAnswers);
          setUserInput("");
          setStage(2); // Switch to stage 2
          setIndex(index + 1);

          // Announce stage 2
          Speech.speak("Stage 2: Diagnostic plates", {
            rate: 1.1,
            pitch: 1.0,
          });
        } else {
          // Score >= 14: Go directly to results
          setCalculating(true);
          const result = computeDiagnosis(newAnswers);
          const shuffledOrder = queue.map((p) => p.id);

          if (auth.currentUser) {
            await saveTestSession({
              userId: auth.currentUser.uid,
              score: stage1Result.correctCount,
              maxScore: stage1Result.totalStage1,
              diagnosis: result.diagnosis,
              diagnosisCode: result.diagnosisCode,
              severity: result.severity,
              total: stage1Result.totalStage1,
              shuffledOrder,
            });
          }

          setTimeout(() => {
            navigation.replace("IshiharaResult", {
              score: stage1Result.correctCount,
              maxScore: stage1Result.totalStage1,
              total: stage1Result.totalStage1,
              diagnosis: result.diagnosis,
              severity: result.severity,
              percentage: result.percentage,
              shuffledOrder,
            });
          }, 1500);
        }
      } else if (index < queue.length - 1) {
        // Continue normally
        setAnswers(newAnswers);
        setUserInput("");
        setIndex(index + 1);

        const nextNum = index + 2;
        if (index < queue.length - 1) {
          Speech.speak(`Plate ${nextNum}`, { rate: 1.1, pitch: 1.0 });
        }
      } else {
        // Test complete (all plates done, including stage 2)
        const result = computeDiagnosis(newAnswers);
        const shuffledOrder = queue.map((p) => p.id);

        setCalculating(true);

        if (auth.currentUser) {
          await saveTestSession({
            userId: auth.currentUser.uid,
            score: result.score,
            maxScore: result.maxScore,
            diagnosis: result.diagnosis,
            diagnosisCode: result.diagnosisCode,
            severity: result.severity,
            total: queue.length,
            shuffledOrder,
          });
        }

        setTimeout(() => {
          navigation.replace("IshiharaResult", {
            score: result.score,
            maxScore: result.maxScore,
            total: queue.length,
            diagnosis: result.diagnosis,
            severity: result.severity,
            percentage: result.percentage,
            shuffledOrder,
          });
        }, 1500);
      }
    },
    [index, queue, answers, userInput, current, navigation, stage],
  );

  if (!current) return <View style={styles.root} />;

  const isTracingPlate = current?.id >= 26;

  if (calculating) {
    return (
      <View style={styles.calculatingOverlay}>
        <ActivityIndicator size="large" color={COLORS.primary} />
        <Text style={styles.calculatingText}>Analysing your results…</Text>
        <Text style={styles.calculatingHint}>Please wait</Text>
      </View>
    );
  }

  // Calculate progress based on current stage (Stage 1 = plates 1-21, Stage 2 = plates 22-25)
  const stageMaxPlates = stage === 1 ? 21 : 25;
  const progress = ((index + 1) / stageMaxPlates) * 100;
  const stageLabel = stage === 1 ? `Screening` : `Diagnostic`;
  const timerColor =
    timeLeft <= 1
      ? COLORS.danger
      : timeLeft <= 2
        ? COLORS.warning
        : COLORS.primary;

  return (
    <View style={styles.root}>
      {/* ── STICKY DISCLAIMER ── */}
      <View style={styles.disclaimerBanner}>
        <Ionicons name="warning-outline" size={13} color={COLORS.warning} />
        <Text style={styles.disclaimerText}>
          NOT A MEDICAL DIAGNOSIS — SCREENING PURPOSE ONLY
        </Text>
      </View>

      {/* ── HEADER & PROGRESS ── */}
      <View style={styles.header}>
        <Text style={styles.headerLabel}>Ishihara Test</Text>
        <Text style={styles.headerSub}>
          {stageLabel} · Plate {index + 1} of {stageMaxPlates}
        </Text>
      </View>
      <View style={styles.progressTrack}>
        <MotiView
          animate={{ width: `${progress}%` }}
          transition={{ type: "timing", duration: 300 }}
          style={styles.progressFill}
        />
      </View>

      {/* ── PLATE AREA ── */}
      <View style={styles.plateArea}>
        <MotiView
          key={index}
          from={{ opacity: 0, scale: 0.88 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "spring", damping: 16, stiffness: 160 }}
          style={styles.plateCard}
        >
          {showImage ? (
            <Image
              source={current.img}
              style={styles.plateImage}
              resizeMode="contain"
            />
          ) : (
            <View style={styles.hiddenState}>
              <Ionicons name="eye-off-outline" size={56} color="#CCC" />
              <Text style={styles.hiddenLabel}>Image Hidden</Text>
              <Text style={styles.hiddenSub}>Enter what you saw</Text>
            </View>
          )}

          {/* Debug overlay — shows correct answer */}
          <View style={styles.debugOverlay}>
            <Text style={styles.debugLabel}>ANS:</Text>
            <Text style={styles.debugAnswer}>
              {current.category === "hidden" ? "—" : current.answer}
            </Text>
          </View>
        </MotiView>

        {/* Timer */}
        <View style={styles.timerRow}>
          <MotiView
            animate={{ backgroundColor: timerColor }}
            transition={{ type: "timing", duration: 300 }}
            style={styles.timerPill}
          >
            <Ionicons name="timer-outline" size={14} color="#FFF" />
            <Text style={styles.timerText}>
              {showImage ? `${timeLeft}s` : "Time's up"}
            </Text>
          </MotiView>
        </View>
      </View>

      {/* ── INPUT PANEL ── */}
      {isTracingPlate ? (
        <View style={styles.tracingPanel}>
          <Text style={styles.tracingQuestion}>
            Can you see a figure in this plate?
          </Text>
          <View style={styles.tracingBtns}>
            <TouchableOpacity
              style={[styles.tracingBtn, { backgroundColor: COLORS.success }]}
              onPress={() => handleNext(current.answer)}
              activeOpacity={0.85}
            >
              <Ionicons name="eye-outline" size={26} color="#FFF" />
              <Text style={styles.tracingBtnText}>Yes, I see it</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.tracingBtn, { backgroundColor: COLORS.danger }]}
              onPress={() => handleNext("")}
              activeOpacity={0.85}
            >
              <Ionicons name="eye-off-outline" size={26} color="#FFF" />
              <Text style={styles.tracingBtnText}>I can't see it</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View style={styles.inputPanel}>
          {/* Answer display */}
          <View style={styles.answerDisplay}>
            <Text style={styles.answerLabel}>YOUR ANSWER</Text>
            <Text style={styles.answerValue}>{userInput || "—"}</Text>
          </View>

          {/* Numpad */}
          <View style={styles.numpad}>
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
              <NumKey
                key={n}
                label={String(n)}
                onPress={() => handleInput(String(n))}
              />
            ))}
            <NumKey label="⌫" onPress={handleBackspace} variant="delete" />
            <NumKey label="0" onPress={() => handleInput("0")} />
            <NumKey
              label="→"
              onPress={handleNext}
              variant="submit"
              disabled={userInput.length === 0 && current.category !== "hidden"}
            />
          </View>

          {/* Skip for hidden plates */}
          {current.category === "hidden" && userInput.length === 0 && (
            <TouchableOpacity style={styles.skipBtn} onPress={handleNext}>
              <Text style={styles.skipText}>I see nothing → Skip</Text>
            </TouchableOpacity>
          )}
        </View>
      )}
    </View>
  );
}

function NumKey({ label, onPress, variant, disabled }) {
  const [pressed, setPressed] = useState(false);
  const keyWidth = (width - SPACING.md * 2 - SPACING.sm * 2) / 3;

  const bg =
    variant === "delete"
      ? "#FFEBEE"
      : variant === "submit"
        ? disabled
          ? "#E0E0E0"
          : COLORS.primary
        : pressed
          ? COLORS.surfaceAlt
          : COLORS.card;

  const textColor =
    variant === "delete"
      ? COLORS.danger
      : variant === "submit"
        ? "#FFF"
        : COLORS.text;

  return (
    <MotiView
      animate={{ scale: pressed ? 0.93 : 1 }}
      transition={{ type: "spring", damping: 14, stiffness: 300 }}
      style={[
        styles.numKey,
        { width: keyWidth, backgroundColor: bg },
        SHADOW.sm,
      ]}
    >
      <TouchableOpacity
        style={styles.numKeyInner}
        onPress={onPress}
        onPressIn={() => setPressed(true)}
        onPressOut={() => setPressed(false)}
        disabled={disabled}
        activeOpacity={1}
      >
        <Text style={[styles.numKeyText, { color: textColor }]}>{label}</Text>
      </TouchableOpacity>
    </MotiView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.background },

  disclaimerBanner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.xs,
    backgroundColor: "#FFF8E1",
    paddingVertical: 6,
    paddingTop: 42,
  },
  disclaimerText: {
    fontSize: 10,
    fontWeight: "700",
    color: COLORS.warning,
    letterSpacing: 0.5,
  },

  header: {
    backgroundColor: COLORS.card,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  headerLabel: { fontSize: 17, fontWeight: "800", color: COLORS.text },
  headerSub: { fontSize: 12, color: COLORS.textLight, marginTop: 1 },

  progressTrack: {
    height: 3,
    backgroundColor: COLORS.border,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    backgroundColor: COLORS.primary,
    borderRadius: 2,
  },

  plateArea: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: SPACING.md,
  },
  plateCard: {
    width: width * 0.82,
    aspectRatio: 1,
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.xl,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    ...SHADOW.lg,
  },
  plateImage: { width: "90%", height: "90%" },
  hiddenState: { alignItems: "center", gap: SPACING.sm },
  hiddenLabel: { fontSize: 16, fontWeight: "700", color: "#AAA" },
  hiddenSub: { fontSize: 12, color: "#CCC" },

  debugOverlay: {
    position: "absolute",
    top: SPACING.md,
    right: SPACING.md,
    backgroundColor: "rgba(255, 193, 7, 0.9)",
    paddingHorizontal: SPACING.sm,
    paddingVertical: 4,
    borderRadius: RADIUS.md,
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    ...SHADOW.md,
  },
  debugLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: "#333",
    letterSpacing: 0.5,
  },
  debugAnswer: {
    fontSize: 14,
    fontWeight: "900",
    color: "#333",
    letterSpacing: 1,
  },

  timerRow: { alignItems: "center" },
  timerPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: SPACING.md,
    paddingVertical: 6,
    borderRadius: 20,
  },
  timerText: { fontSize: 13, fontWeight: "700", color: "#FFF" },

  inputPanel: {
    backgroundColor: COLORS.card,
    borderTopLeftRadius: RADIUS.xl,
    borderTopRightRadius: RADIUS.xl,
    paddingHorizontal: SPACING.md,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.xl,
    ...SHADOW.lg,
  },

  answerDisplay: { alignItems: "center", marginBottom: SPACING.md },
  answerLabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 1.5,
    color: COLORS.textLight,
    marginBottom: 4,
  },
  answerValue: {
    fontSize: 34,
    fontWeight: "800",
    color: COLORS.text,
    letterSpacing: 4,
  },

  numpad: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: SPACING.sm,
    justifyContent: "center",
  },
  numKey: {
    height: 50,
    borderRadius: RADIUS.md,
    overflow: "hidden",
  },
  numKeyInner: { flex: 1, alignItems: "center", justifyContent: "center" },
  numKeyText: { fontSize: 20, fontWeight: "700" },

  skipBtn: {
    alignSelf: "center",
    marginTop: SPACING.sm,
    paddingVertical: SPACING.sm,
  },
  skipText: { fontSize: 13, color: COLORS.textLight, fontWeight: "500" },

  calculatingOverlay: {
    flex: 1,
    backgroundColor: COLORS.background,
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.md,
  },
  calculatingText: { fontSize: 18, fontWeight: "700", color: COLORS.text },
  calculatingHint: { fontSize: 13, color: COLORS.textLight },

  tracingPanel: {
    backgroundColor: COLORS.card,
    borderTopLeftRadius: RADIUS.xl,
    borderTopRightRadius: RADIUS.xl,
    paddingHorizontal: SPACING.md,
    paddingTop: SPACING.lg,
    paddingBottom: SPACING.xl,
    ...SHADOW.lg,
  },
  tracingQuestion: {
    textAlign: "center",
    fontSize: 15,
    fontWeight: "600",
    color: COLORS.text,
    marginBottom: SPACING.md,
  },
  tracingBtns: {
    flexDirection: "row",
    gap: SPACING.md,
  },
  tracingBtn: {
    flex: 1,
    borderRadius: RADIUS.lg,
    paddingVertical: 22,
    alignItems: "center",
    gap: SPACING.sm,
    ...SHADOW.md,
  },
  tracingBtnText: {
    color: "#FFF",
    fontWeight: "700",
    fontSize: 14,
  },
});
