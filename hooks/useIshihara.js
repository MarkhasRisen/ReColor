import { useEffect, useState } from "react";
import { RAW_PLATES } from "../constants/theme";
import { auth, saveExamResult } from "../firebaseConfig";

export const useIshihara = (testType, navigation) => {
  const [testState, setTestState] = useState(() => {
    // Shuffling and Slicing logic moved here
    const count = testType === "comprehensive" ? 38 : 14;
    const shuffled = [...RAW_PLATES]
      .sort(() => 0.5 - Math.random())
      .slice(0, count);

    return {
      queue: shuffled,
      current: shuffled[0],
      index: 0,
      score: 0,
      userInput: "",
    };
  });

  const [timeLeft, setTimeLeft] = useState(5);
  const [showImage, setShowImage] = useState(true);

  // --- TIMER LOGIC ---
  useEffect(() => {
    if (!testState.current) return;
    setTimeLeft(5);
    setShowImage(true);

    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          setShowImage(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [testState.index]);

  // --- NAVIGATION & SAVING LOGIC ---
  const handleNext = async () => {
    const { queue, index, score, userInput, current } = testState;
    const isCorrect = userInput === current.answer;
    const newScore = isCorrect ? score + 1 : score;

    if (index < queue.length - 1) {
      setTestState((prev) => ({
        ...prev,
        index: index + 1,
        current: queue[index + 1],
        score: newScore,
        userInput: "",
      }));
    } else {
      // Logic for results and Firebase saving
      let diagnosis = "Normal Vision";
      let severity = "None";
      const percentage = (newScore / queue.length) * 100;

      if (percentage < 80) {
        diagnosis = "Deuteranomaly (Simulated)";
        severity = percentage < 40 ? "Severe" : "Moderate";
      }

      if (auth.currentUser) {
        await saveExamResult(
          auth.currentUser.uid,
          newScore,
          diagnosis,
          severity,
        );
      }

      navigation.replace("IshiharaResult", {
        score: newScore,
        total: queue.length,
        type: testType,
      });
    }
  };

  const handleInput = (num) => {
    if (testState.userInput.length < 3) {
      setTestState((prev) => ({ ...prev, userInput: prev.userInput + num }));
    }
  };

  const handleBackspace = () => {
    setTestState((prev) => ({
      ...prev,
      userInput: prev.userInput.slice(0, -1),
    }));
  };

  return {
    testState,
    timeLeft,
    showImage,
    handleNext,
    handleInput,
    handleBackspace,
  };
};
