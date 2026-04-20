import { ISHIHARA_PLATES } from '../data/ishiharaData';

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
// count: total plates to include (default 38 for comprehensive, 14 for quick).
export function buildTestQueue(count = 38) {
  const plate1 = ISHIHARA_PLATES.find((p) => p.id === 1);
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
    if (plate.category === 'demo' || plate.category === 'hidden') return;
    const w = plate.weight ?? 1;
    maxScore += w;
    if (isCorrect) weightedScore += w;
  });

  return { weightedScore, maxScore };
}

// Protan/Deutan differentiation based on answers to plates 22-25.
// Returns 'Protan', 'Deutan', or null if insufficient data.
function detectProtanDeutan(answers) {
  const diagnosticAnswers = answers.filter(
    ({ plate }) => plate.category === 'diagnostic'
  );
  if (diagnosticAnswers.length === 0) return null;

  let protanHits = 0;
  let deutanHits = 0;

  diagnosticAnswers.forEach(({ plate, userAnswer }) => {
    if (plate.protanAnswer && userAnswer === plate.protanAnswer) protanHits++;
    if (plate.deutanAnswer && userAnswer === plate.deutanAnswer) deutanHits++;
  });

  if (protanHits === 0 && deutanHits === 0) return null;
  return protanHits >= deutanHits ? 'Protan' : 'Deutan';
}

// Full diagnosis from a completed answer set.
// Returns { diagnosis, severity, diagnosisCode, weightedScore, maxScore, percentage }
export function computeDiagnosis(answers) {
  const { weightedScore, maxScore } = calculateWeightedScore(answers);
  const percentage = maxScore > 0 ? (weightedScore / maxScore) * 100 : 0;

  let diagnosis = 'Normal Vision';
  let severity = 'None';
  let diagnosisCode = 'N';

  if (percentage >= 80) {
    diagnosis = 'Normal Vision';
    severity = 'None';
    diagnosisCode = 'N';
  } else {
    const subtype = detectProtanDeutan(answers);
    const subtypeLabel = subtype === 'Protan' ? 'Protanomaly' : 'Deuteranomaly';
    diagnosisCode = subtype === 'Protan' ? 'P' : 'D';

    if (percentage >= 60) {
      severity = 'Mild';
      diagnosis = `Mild ${subtypeLabel ?? 'Red-Green Deficiency'}`;
    } else if (percentage >= 40) {
      severity = 'Moderate';
      diagnosis = `Moderate ${subtypeLabel ?? 'Red-Green Deficiency'}`;
    } else {
      severity = 'Severe';
      diagnosis = `Severe ${subtypeLabel ?? 'Red-Green Deficiency'}`;
    }
  }

  return {
    diagnosis,
    severity,
    diagnosisCode,
    weightedScore,
    maxScore,
    percentage: Math.round(percentage),
  };
}
