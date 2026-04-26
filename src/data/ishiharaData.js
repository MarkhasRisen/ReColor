// 25-plate Ishihara colour vision test dataset.
// Plate #1 is the demonstration plate — MUST always be shown first.
// Categories:
//   'demo'         — shown to everyone, verifies the test is working
//   'screening'    — detects presence of red-green CVD (weight 1)
//   'vanishing'    — visible only to normal vision (weight 1)
//   'hidden'       — visible only to CVD (weight 1)
//   'diagnostic'   — differentiates Protan vs Deutan (weight 2)
//   'tracing'      — tracing plates for qualitative assessment (weight 1)

export const ISHIHARA_PLATES = [
  // ── DEMONSTRATION ──────────────────────────────────────────────
  {
    id: 1,
    img: require("../../assets/1.png"),
    answer: "12",
    category: "demo",
    weight: 0,
    description: 'Demonstration plate – everyone should read "12".',
  },

  // ── SCREENING (Plates 2–9) ──────────────────────────────────────
  {
    id: 2,
    img: require("../../assets/2.png"),
    answer: "8",
    category: "screening",
    weight: 1,
    description: "Normal: 8 | CVD: 3",
  },
  {
    id: 3,
    img: require("../../assets/3.png"),
    answer: "6",
    category: "screening",
    weight: 1,
    description: "Normal: 6 | CVD: 5",
  },
  {
    id: 4,
    img: require("../../assets/4.png"),
    answer: "29",
    category: "screening",
    weight: 1,
    description: "Normal: 29 | CVD: 70",
  },
  {
    id: 5,
    img: require("../../assets/5.png"),
    answer: "57",
    category: "screening",
    weight: 1,
    description: "Normal: 57 | CVD: 35",
  },
  {
    id: 6,
    img: require("../../assets/6.png"),
    answer: "5",
    category: "screening",
    weight: 1,
    description: "Normal: 5 | CVD: 2",
  },
  {
    id: 7,
    img: require("../../assets/7.png"),
    answer: "3",
    category: "screening",
    weight: 1,
    description: "Normal: 3 | CVD: 5",
  },
  {
    id: 8,
    img: require("../../assets/8.png"),
    answer: "15",
    category: "screening",
    weight: 1,
    description: "Normal: 15 | CVD: 17",
  },
  {
    id: 9,
    img: require("../../assets/9.png"),
    answer: "74",
    category: "screening",
    weight: 1,
    description: "Normal: 74 | CVD: 21",
  },

  // ── VANISHING (Plates 10–17, visible only to normal vision) ────
  {
    id: 10,
    img: require("../../assets/10.png"),
    answer: "2",
    category: "vanishing",
    weight: 1,
    description: "Normal: 2 | CVD: none",
  },
  {
    id: 11,
    img: require("../../assets/11.png"),
    answer: "6",
    category: "vanishing",
    weight: 1,
    description: "Normal: 6 | CVD: none",
  },
  {
    id: 12,
    img: require("../../assets/12.png"),
    answer: "97",
    category: "vanishing",
    weight: 1,
    description: "Normal: 97 | CVD: none",
  },
  {
    id: 13,
    img: require("../../assets/13.png"),
    answer: "45",
    category: "vanishing",
    weight: 1,
    description: "Normal: 45 | CVD: none",
  },
  {
    id: 14,
    img: require("../../assets/14.png"),
    answer: "5",
    category: "vanishing",
    weight: 1,
    description: "Normal: 5 | CVD: none",
  },
  {
    id: 15,
    img: require("../../assets/15.png"),
    answer: "7",
    category: "vanishing",
    weight: 1,
    description: "Normal: 7 | CVD: none",
  },
  {
    id: 16,
    img: require("../../assets/16.png"),
    answer: "16",
    category: "vanishing",
    weight: 1,
    description: "Normal: 16 | CVD: none",
  },
  {
    id: 17,
    img: require("../../assets/17.png"),
    answer: "73",
    category: "vanishing",
    weight: 1,
    description: "Normal: 73 | CVD: none",
  },

  // ── HIDDEN (Plates 18–21, visible only to CVD) ─────────────────
  {
    id: 18,
    img: require("../../assets/18.png"),
    answer: "",
    category: "hidden",
    weight: 1,
    description: "Normal: none | CVD: 5",
  },
  {
    id: 19,
    img: require("../../assets/19.png"),
    answer: "",
    category: "hidden",
    weight: 1,
    description: "Normal: none | CVD: 45",
  },
  {
    id: 20,
    img: require("../../assets/20.png"),
    answer: "",
    category: "hidden",
    weight: 1,
    description: "Normal: none | CVD: 73",
  },
  {
    id: 21,
    img: require("../../assets/21.png"),
    answer: "",
    category: "hidden",
    weight: 1,
    description: "Normal: none | CVD: number",
  },

  // ── DIAGNOSTIC Protan/Deutan (Plates 22–25, weight = 2) ────────
  {
    id: 22,
    img: require("../../assets/22.png"),
    answer: "26",
    category: "diagnostic",
    weight: 2,
    protanAnswer: "6",
    deutanAnswer: "2",
    description: "Protan: 6 | Deutan: 2 | Normal: 26",
  },
  {
    id: 23,
    img: require("../../assets/23.png"),
    answer: "42",
    category: "diagnostic",
    weight: 2,
    protanAnswer: "2",
    deutanAnswer: "4",
    description: "Protan: 2 | Deutan: 4 | Normal: 42",
  },
  {
    id: 24,
    img: require("../../assets/24.png"),
    answer: "35",
    category: "diagnostic",
    weight: 2,
    protanAnswer: "5",
    deutanAnswer: "3",
    description: "Protan: 5 | Deutan: 3 | Normal: 35",
  },
  {
    id: 25,
    img: require("../../assets/25.png"),
    answer: "96",
    category: "diagnostic",
    weight: 2,
    protanAnswer: "6",
    deutanAnswer: "9",
    description: "Protan: 6 | Deutan: 9 | Normal: 96",
  },
];
