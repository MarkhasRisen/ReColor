const tintColorLight = "#6C63FF"; // Updated to ReColor Purple
const tintColorDark = "#fff";

export const Colors = {
  light: {
    text: "#11181C",
    background: "#F8F9FA", // Updated to your app background
    tint: tintColorLight,
    icon: "#687076",
    tabIconDefault: "#A4B0BE",
    tabIconSelected: tintColorLight,
    // --- ReColor Custom Palette ---
    primary: "#6C63FF",
    secondary: "#FF4081",
    accent: "#00D2D3",
    card: "#FFFFFF",
    success: "#2ECC71",
    warning: "#FF9F43",
    danger: "#FF6B6B",
    darkOverlay: "rgba(0,0,0,0.6)",
  },
  dark: {
    // Keep your existing dark mode or map ReColor colors to darker variants
    text: "#ECEDEE",
    background: "#151718",
    tint: tintColorDark,
    icon: "#9BA1A6",
    tabIconDefault: "#9BA1A6",
    tabIconSelected: tintColorDark,
    card: "#1E1E1E",
  },
};

// ... keep your existing Fonts object ...

// --- The Brain: Ishihara Plate Data ---
export const RAW_PLATES = [
  { id: 1, img: require("../assets/plate_1.png"), answer: "9" },
  { id: 2, img: require("../assets/plate_2.png"), answer: "8" },
  { id: 3, img: require("../assets/plate_3.png"), answer: "12" },
  { id: 4, img: require("../assets/plate_4.png"), answer: "42" },
  { id: 5, img: require("../assets/plate_5.png"), answer: "6" },
  { id: 6, img: require("../assets/plate_6.png"), answer: "2" },
  { id: 7, img: require("../assets/plate_7.png"), answer: "74" },
  { id: 8, img: require("../assets/plate_8.png"), answer: "6" },
  { id: 9, img: require("../assets/plate_9.png"), answer: "16" },
  { id: 10, img: require("../assets/plate_10.png"), answer: "2" },
  { id: 11, img: require("../assets/plate_11.png"), answer: "29" },
  { id: 12, img: require("../assets/plate_12.png"), answer: "7" },
  { id: 13, img: require("../assets/plate_13.png"), answer: "45" },
  { id: 14, img: require("../assets/plate_14.png"), answer: "5" },
  { id: 15, img: require("../assets/plate_15.png"), answer: "97" },
  { id: 16, img: require("../assets/plate_16.png"), answer: "8" },
  { id: 17, img: require("../assets/plate_17.png"), answer: "42" },
  { id: 18, img: require("../assets/plate_18.png"), answer: "3" },
  // ... and so on up to 38
];
