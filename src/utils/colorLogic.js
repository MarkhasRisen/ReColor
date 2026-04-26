import { ISHIHARA_PLATES } from "../data/ishiharaData";
import { QUICK_ISHIHARA_PLATES } from "../data/quickIshiharaData";

// ─────────────────────────────────────────────────────────────
// Shared utilities
// ─────────────────────────────────────────────────────────────

// Fisher-Yates shuffle — O(n), unbiased
export function fisherYatesShuffle(arr) {
  const shuffled = [...arr];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Mode-specific constants.
// Comprehensive: 25 plates, Stage 1 = 21, Stage 2 = 4 diagnostic, ≥3/4 classification.
// Quick:         14 plates, Stage 1 = 11, Stage 2 = 3 diagnostic, ≥2/3 classification.
const MODE_CONFIG = {
  comprehensive: {
    totalPlates: 25,
    stage1Length: 21, // demo + plates 2-21
    stage2Length: 4,  // plates 22-25
    normalThreshold: 17,       // ≥17 → Normal
    indeterminateThreshold: 14, // 14-16 → Indeterminate, <14 → Stage 2
    classificationHits: 3,      // ≥3 of 4 diagnostic matches
  },
  quick: {
    totalPlates: 14,
    stage1Length: 11, // demo + plates 2-11
    stage2Length: 3,  // plates 12-14
    normalThreshold: 10,       // ≥10 → Normal (perfect)
    indeterminateThreshold: 8,  // 8-9 → Borderline, <8 → Stage 2
    classificationHits: 2,      // ≥2 of 3 diagnostic matches
  },
};

function getConfig(testType) {
  return MODE_CONFIG[testType] ?? MODE_CONFIG.comprehensive;
}

// ─────────────────────────────────────────────────────────────
// Queue construction
// ─────────────────────────────────────────────────────────────

// Build test queue: Plate #1 (demo) always first; Stage 1 plates randomised;
// Stage 2 (diagnostic) plates kept in clinical order at the tail.
export function buildTestQueue(testType = "comprehensive") {
  if (testType === "quick") return buildQuickQueue();
  return buildComprehensiveQueue();
}

function buildComprehensiveQueue() {
  const plate1 = ISHIHARA_PLATES.find((p) => p.id === 1);
  const stage1Plates = ISHIHARA_PLATES.filter((p) => p.id >= 2 && p.id <= 21);
  const diagnosticPlates = ISHIHARA_PLATES.filter(
    (p) => p.id >= 22 && p.id <= 25,
  );
  const shuffledStage1 = fisherYatesShuffle(stage1Plates);
  return [plate1, ...shuffledStage1, ...diagnosticPlates];
}

function buildQuickQueue() {
  const plate1 = QUICK_ISHIHARA_PLATES.find((p) => p.id === 1);
  const stage1Plates = QUICK_ISHIHARA_PLATES.filter(
    (p) => p.id >= 2 && p.id <= 11,
  );
  const diagnosticPlates = QUICK_ISHIHARA_PLATES.filter(
    (p) => p.id >= 12 && p.id <= 14,
  );
  const shuffledStage1 = fisherYatesShuffle(stage1Plates);
  return [plate1, ...shuffledStage1, ...diagnosticPlates];
}

// ─────────────────────────────────────────────────────────────
// Stage 1 evaluation
// ─────────────────────────────────────────────────────────────

// Counts correct answers in the Stage-1 slice for the given mode and
// decides whether Stage 2 should run.
//
// Quick mode excludes the demo plate from scoring per the clinical spec
// ("Plate 1 is excluded from scoring", Ishihara Concise Edition 1960).
// Comprehensive mode counts all 21 Stage-1 answers (demo included) per
// the ≥17/21 rule the user verified earlier.
export function evaluateStage1(answers, testType = "comprehensive") {
  const { stage1Length, indeterminateThreshold } = getConfig(testType);
  const stage1Answers = answers.slice(0, stage1Length);

  const scoredAnswers =
    testType === "quick"
      ? stage1Answers.filter(({ plate }) => plate.category !== "demo")
      : stage1Answers;

  let correctCount = 0;
  scoredAnswers.forEach(({ isCorrect }) => {
    if (isCorrect) correctCount++;
  });

  return {
    correctCount,
    totalStage1: scoredAnswers.length, // 10 for quick, 21 for comprehensive
    shouldProceedToStage2: correctCount < indeterminateThreshold,
  };
}

// ─────────────────────────────────────────────────────────────
// Stage 2 classification (Protan vs Deutan)
// ─────────────────────────────────────────────────────────────

// Checks if ≥ minHits of the diagnostic plates match the protan or deutan pattern.
// Returns 'Protan', 'Deutan', or null if inconclusive.
function detectProtanDeutan(diagnosticAnswers, minHits) {
  const diagnostic = diagnosticAnswers.filter(
    ({ plate }) => plate.category === "diagnostic",
  );

  if (diagnostic.length === 0) return null;

  let protanHits = 0;
  let deutanHits = 0;

  diagnostic.forEach(({ plate, userAnswer }) => {
    if (plate.protanAnswer && userAnswer === plate.protanAnswer) protanHits++;
    if (plate.deutanAnswer && userAnswer === plate.deutanAnswer) deutanHits++;
  });

  if (protanHits >= minHits) return "Protan";
  if (deutanHits >= minHits) return "Deutan";
  return null;
}

// ─────────────────────────────────────────────────────────────
// Full diagnosis (Stage 1 + optional Stage 2)
// ─────────────────────────────────────────────────────────────

export function computeDiagnosis(answers, testType = "comprehensive") {
  const config = getConfig(testType);
  const stage1 = evaluateStage1(answers, testType);

  if (stage1.correctCount >= config.normalThreshold) {
    return {
      diagnosis: "Normal Vision",
      severity: "None",
      diagnosisCode: "N",
      score: stage1.correctCount,
      maxScore: stage1.totalStage1,
      percentage: Math.round((stage1.correctCount / stage1.totalStage1) * 100),
      stage: 1,
    };
  }

  if (stage1.correctCount >= config.indeterminateThreshold) {
    return {
      diagnosis: "Indeterminate Result",
      severity: "Borderline",
      diagnosisCode: "I",
      score: stage1.correctCount,
      maxScore: stage1.totalStage1,
      percentage: Math.round((stage1.correctCount / stage1.totalStage1) * 100),
      stage: 1,
    };
  }

  // Stage 2: diagnostic plates begin at position stage1Length
  const stage2Answers = answers.slice(config.stage1Length);

  let totalCorrect = stage1.correctCount;
  stage2Answers.forEach(({ isCorrect }) => {
    if (isCorrect) totalCorrect++;
  });
  const totalAnswered = stage1.totalStage1 + stage2Answers.length;

  if (stage2Answers.length === 0) {
    return {
      diagnosis: "Incomplete",
      severity: "N/A",
      diagnosisCode: "X",
      score: stage1.correctCount,
      maxScore: stage1.totalStage1,
      percentage: Math.round((stage1.correctCount / stage1.totalStage1) * 100),
      stage: 1,
    };
  }

  const subtype = detectProtanDeutan(stage2Answers, config.classificationHits);
  const base = {
    score: totalCorrect,
    maxScore: totalAnswered,
    percentage: Math.round((totalCorrect / totalAnswered) * 100),
    stage: 2,
  };

  if (subtype === "Protan") {
    return { ...base, diagnosis: "Protanomaly", severity: "Mild", diagnosisCode: "P" };
  }
  if (subtype === "Deutan") {
    return { ...base, diagnosis: "Deuteranomaly", severity: "Mild", diagnosisCode: "D" };
  }
  return { ...base, diagnosis: "Indeterminate Result", severity: "Borderline", diagnosisCode: "I" };
}
