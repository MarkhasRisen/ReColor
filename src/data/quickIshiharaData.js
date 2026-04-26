// 14-plate Ishihara Quick Mode (Concise Edition, Ishihara 1960).
// Plate 1 = demonstration (unscored). Plates 2-11 = Stage 1 screening (10 scored).
// Plates 12-14 = Stage 2 classification (protan vs deutan), only shown if Stage 1 ≤ 7.
//
// Source mapping (14-plate # → 38-plate Ishihara source):
//   1-11 → plates 1-11 (same)
//   12   → plate 24
//   13   → plate 25
//   14   → plate 26 (any tracing plate from 26-38)
//
// Thresholds (Stage 1, 10 scored plates, demo excluded):
//   ≥10/10 → Normal Colour Vision   (end test)
//   8-9/10 → Borderline             (end test)
//   ≤ 7/10 → Red-Green CVD Detected (proceed to Stage 2)
//
// Stage 2 Classification Rule (≥2 of 3):
//   ≥2 of 3 protan matches → Protanomaly
//   ≥2 of 3 deutan matches → Deuteranomaly
//   Otherwise → Indeterminate
//
// inputType:
//   "numeric"        — numpad (default)
//   "tracing-yesno"  — Yes / No buttons (plate 11)
//   "tracing-multi"  — Both / Purple only / Red only buttons (plate 14)

export const QUICK_ISHIHARA_PLATES = [
  // ── DEMONSTRATION (Source plate 1) ────────────────────────────
  {
    id: 1,
    img: require("../../assets/1.png"),
    answer: "12",
    category: "demo",
    inputType: "numeric",
    description: 'Demonstration plate – everyone should read "12".',
  },

  // ── STAGE 1: TRANSFORMATION (Source plates 2-5) ───────────────
  {
    id: 2,
    img: require("../../assets/2.png"),
    answer: "8",
    category: "transformation",
    inputType: "numeric",
    description: "Normal: 8 | CVD: 3",
  },
  {
    id: 3,
    img: require("../../assets/3.png"),
    answer: "5",
    category: "transformation",
    inputType: "numeric",
    description: "Normal: 5 | CVD: 2",
  },
  {
    id: 4,
    img: require("../../assets/4.png"),
    answer: "29",
    category: "transformation",
    inputType: "numeric",
    description: "Normal: 29 | CVD: 70",
  },
  {
    id: 5,
    img: require("../../assets/5.png"),
    answer: "74",
    category: "transformation",
    inputType: "numeric",
    description: "Normal: 74 | CVD: 21",
  },

  // ── STAGE 1: VANISHING (Source plates 6, 7, 8) ────────────────
  {
    id: 6,
    img: require("../../assets/6.png"),
    answer: "7",
    category: "vanishing",
    inputType: "numeric",
    description: "Normal: 7 | CVD: none",
  },
  {
    id: 7,
    img: require("../../assets/7.png"),
    answer: "45",
    category: "vanishing",
    inputType: "numeric",
    description: "Normal: 45 | CVD: none",
  },
  {
    id: 8,
    img: require("../../assets/8.png"),
    answer: "2",
    category: "vanishing",
    inputType: "numeric",
    description: "Normal: 2 | CVD: none",
  },

  // ── STAGE 1: HIDDEN DIGIT (Source plate 9, reversed) ──────────
  // Normal vision sees nothing; CVD sees "2". Correct answer = empty ("").
  {
    id: 9,
    img: require("../../assets/9.png"),
    answer: "",
    category: "hidden",
    inputType: "numeric",
    description: "Normal: none | CVD: 2 (reversed plate)",
  },

  // ── STAGE 1: VANISHING (Source plate 10) ──────────────────────
  {
    id: 10,
    img: require("../../assets/10.png"),
    answer: "16",
    category: "vanishing",
    inputType: "numeric",
    description: "Normal: 16 | CVD: none",
  },

  // ── STAGE 1: TRACING (Source plate 11) ────────────────────────
  // Normal traces the blue-green line; CVD cannot or traces wrong.
  // Correct answer = "yes" (can trace).
  {
    id: 11,
    img: require("../../assets/30.png"),
    answer: "yes",
    category: "tracing",
    inputType: "tracing-yesno",
    description: "Normal: traces line | CVD: cannot",
  },

  // ── STAGE 2: CLASSIFICATION (Source plates 24, 25, 26) ────────
  {
    id: 12,
    img: require("../../assets/24.png"),
    answer: "35",
    category: "diagnostic",
    inputType: "numeric",
    protanAnswer: "5",
    deutanAnswer: "3",
    description: "Protan: 5 | Deutan: 3 | Normal: 35 (Source plate 24)",
  },
  {
    id: 13,
    img: require("../../assets/25.png"),
    answer: "96",
    category: "diagnostic",
    inputType: "numeric",
    protanAnswer: "6",
    deutanAnswer: "9",
    description: "Protan: 6 | Deutan: 9 | Normal: 96 (Source plate 25)",
  },
  {
    id: 14,
    img: require("../../assets/26.png"),
    answer: "both",
    category: "diagnostic",
    inputType: "tracing-multi",
    protanAnswer: "purple",
    deutanAnswer: "red",
    description:
      "Normal: both lines | Protan: purple only | Deutan: red only (Source plate 26)",
  },
];
