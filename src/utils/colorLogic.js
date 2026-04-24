import { ISHIHARA_PLATES } from "../data/ishiharaData";

// Fisher-Yates shuffle — O(n), unbiased
export function fisherYatesShuffle(arr) {
  const shuffled = [...arr];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Build test queue: Plate #1 (demo) is always first; the rest are randomised.
// For comprehensive test (count=25): demo + 20 stage1 plates + 4 diagnostic plates
// Stage 1 positions (1-20): screening, vanishing, hidden plates only (randomised)
// Stage 2 positions (21-24): diagnostic plates (22-25) in order
export function buildTestQueue(count = 25) {
  const plate1 = ISHIHARA_PLATES.find((p) => p.id === 1);

  // For comprehensive test of 25 plates:
  if (count === 25) {
    // Get all non-diagnostic, non-demo plates (2-21)
    const stage1Plates = ISHIHARA_PLATES.filter((p) => p.id >= 2 && p.id <= 21);
    // Get all diagnostic plates (22-25)
    const diagnosticPlates = ISHIHARA_PLATES.filter(
      (p) => p.id >= 22 && p.id <= 25,
    );

    // Shuffle stage 1 plates and build final queue
    const shuffledStage1 = fisherYatesShuffle(stage1Plates);
    return [plate1, ...shuffledStage1, ...diagnosticPlates];
  }

  // For other counts (e.g. 14-plate quick test), use simple shuffle logic.
  const rest = ISHIHARA_PLATES.filter((p) => p.id !== 1);
  const shuffled = fisherYatesShuffle(rest).slice(0, count - 1);
  return [plate1, ...shuffled];
}

// Weighted scoring:  Score = Σ (Correct × Weight)
// Plates 22-25 (diagnostic) carry weight 2 for Protan/Deutan differentiation.
// Hidden-category plates (18-21) are excluded from scoring (normal answer is '').
export function calculateWeightedScore(answers) {
  let weightedScore = 0;
  let maxScore = 0;

  answers.forEach(({ plate, isCorrect }) => {
    if (plate.category === "demo" || plate.category === "hidden") return;
    const w = plate.weight ?? 1;
    maxScore += w;
    if (isCorrect) weightedScore += w;
  });

  return { weightedScore, maxScore };
}

// Stage 1 evaluation: Count correct answers in the FIRST 21 answers
// (demo plate + 20 screening/vanishing/hidden plates = plates 1-21).
// Thresholds per clinical spec:
//   17-21 correct → Normal vision      (end test, no Stage 2)
//   14-16 correct → Indeterminate      (end test, no Stage 2)
//    0-13 correct → Proceed to Stage 2 (diagnostic plates 22-25)
export function evaluateStage1(answers) {
  const stage1Answers = answers.slice(0, 21);

  let correctCount = 0;
  stage1Answers.forEach(({ isCorrect }) => {
    if (isCorrect) correctCount++;
  });

  const shouldProceedToStage2 = correctCount < 14;

  return {
    correctCount,
    totalStage1: stage1Answers.length, // 21
    shouldProceedToStage2,
  };
}

// Protan/Deutan differentiation based on diagnostic plates (22-25).
// Checks if ≥3 of 4 plates match protan or deutan pattern.
// Returns 'Protan', 'Deutan', or null if insufficient data.
function detectProtanDeutan(diagnosticAnswers) {
  // Only process diagnostic plates (22-25)
  const diagnostic = diagnosticAnswers.filter(
    ({ plate }) => plate.category === "diagnostic",
  );

  if (diagnostic.length < 3) return null; // Need at least 3 diagnostic plates

  let protanHits = 0;
  let deutanHits = 0;

  diagnostic.forEach(({ plate, userAnswer }) => {
    if (plate.protanAnswer && userAnswer === plate.protanAnswer) protanHits++;
    if (plate.deutanAnswer && userAnswer === plate.deutanAnswer) deutanHits++;
  });

  // Rule: If ≥3 of 4 match protan pattern → Protan
  // If ≥3 of 4 match deutan pattern → Deutan
  // Otherwise → null (indeterminate)
  if (protanHits >= 3) return "Protan";
  if (deutanHits >= 3) return "Deutan";
  return null;
}

// Full diagnosis from a completed answer set (two-stage process)
// Returns { diagnosis, severity, diagnosisCode, weightedScore, maxScore, percentage }
export function computeDiagnosis(answers) {
  const stage1 = evaluateStage1(answers);

  if (stage1.correctCount >= 17) {
    return {
      diagnosis: "Normal Vision",
      severity: "None",
      diagnosisCode: "N",
      score: stage1.correctCount,
      maxScore: stage1.totalStage1,
      percentage: Math.round((stage1.correctCount / stage1.totalStage1) * 100),
      stage: 1,
    };
  } else if (stage1.correctCount >= 14) {
    return {
      diagnosis: "Indeterminate Result",
      severity: "Borderline",
      diagnosisCode: "I",
      score: stage1.correctCount,
      maxScore: stage1.totalStage1,
      percentage: Math.round((stage1.correctCount / stage1.totalStage1) * 100),
      stage: 1,
    };
  } else {
    // Stage 2: diagnostic plates are positions 21-24 in the queue
    const stage2Answers = answers.slice(21);

    // Total-correct across ALL answered plates (stage 1 + stage 2)
    let totalCorrect = stage1.correctCount;
    stage2Answers.forEach(({ isCorrect }) => {
      if (isCorrect) totalCorrect++;
    });
    const totalAnswered = stage1.totalStage1 + stage2Answers.length;

    if (stage2Answers.length > 0) {
      const subtype = detectProtanDeutan(stage2Answers);

      if (subtype === "Protan") {
        return {
          diagnosis: "Protanomaly",
          severity: "Mild",
          diagnosisCode: "P",
          score: totalCorrect,
          maxScore: totalAnswered,
          percentage: Math.round((totalCorrect / totalAnswered) * 100),
          stage: 2,
        };
      } else if (subtype === "Deutan") {
        return {
          diagnosis: "Deuteranomaly",
          severity: "Mild",
          diagnosisCode: "D",
          score: totalCorrect,
          maxScore: totalAnswered,
          percentage: Math.round((totalCorrect / totalAnswered) * 100),
          stage: 2,
        };
      } else {
        return {
          diagnosis: "Indeterminate Result",
          severity: "Borderline",
          diagnosisCode: "I",
          score: totalCorrect,
          maxScore: totalAnswered,
          percentage: Math.round((totalCorrect / totalAnswered) * 100),
          stage: 2,
        };
      }
    } else {
      return {
        diagnosis: "Incomplete",
        severity: "N/A",
        diagnosisCode: "X",
        score: stage1.correctCount,
        maxScore: stage1.totalStage1,
        percentage: Math.round(
          (stage1.correctCount / stage1.totalStage1) * 100,
        ),
        stage: 1,
      };
    }
  }
}
