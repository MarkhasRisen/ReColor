import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useIsFocused } from "@react-navigation/native";
import * as ImageManipulator from "expo-image-manipulator";
import * as Speech from "expo-speech"; // Integrated for Objective 2
import { useEffect, useRef, useState } from "react";
import {
  Dimensions,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from "react-native-vision-camera";
import { decodeJpegBase64, identifyColor } from "../../tensorHelper";
import ModeSelector from "../components/ModeSelector";
import { styles } from "../theme/styles";
import { ScreenErrorBoundary } from "../utils/logger";

const { width, height: screenHeight } = Dimensions.get("window");

function ColorIdentifierScreenInner({ navigation }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const isFocused = useIsFocused();
  const [cameraPosition, setCameraPosition] = useState("back");
  const [audio, setAudio] = useState(false); // Default to false
  const [showModal, setShowModal] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({
    x: width / 2,
    y: screenHeight / 2,
  });
  const [identifiedColor, setIdentifiedColor] = useState({
    name: "Ready to Scan",
    hex: "#333",
    conf: "",
  });
  const [isDetecting, setIsDetecting] = useState(false);

  const cameraRef = useRef(null);
  const isProcessingRef = useRef(false);
  const isMountedRef = useRef(true);
  const device = useCameraDevice(cameraPosition);

  // Sync with global settings on mount
  useEffect(() => {
    const syncAudio = async () => {
      const saved = await AsyncStorage.getItem("audio_feedback_enabled");
      setAudio(saved === "true");
    };
    syncAudio();
  }, [isFocused]);

  useEffect(() => {
    isMountedRef.current = true;
    const timer = setTimeout(() => {
      if (isMountedRef.current) setCameraReady(true);
    }, 700);
    return () => {
      isMountedRef.current = false;
      clearTimeout(timer);
    };
  }, []);

  if (!hasPermission) {
    return (
      <View
        style={[
          styles.container,
          { justifyContent: "center", alignItems: "center" },
        ]}
      >
        <Text style={{ marginBottom: 20 }}>Camera access is needed.</Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const runDetection = async (cx, cy) => {
    if (!cameraRef.current || isProcessingRef.current) return;
    isProcessingRef.current = true;
    setIsDetecting(true);

    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: "speed",
        enableShutterSound: false,
      });
      const fileUri = `file://${photo.path}`;

      const resized = await ImageManipulator.manipulateAsync(
        fileUri,
        [{ resize: { width: 640 } }],
        {
          base64: true,
          format: ImageManipulator.SaveFormat.JPEG,
          compress: 0.95,
        },
      );

      const decoded = decodeJpegBase64(resized.base64);
      const imgW = resized.width;
      const imgH = resized.height;

      // Maintain your original aspect ratio mapping logic
      const screenAspect = width / screenHeight;
      const photoAspect = imgW / imgH;
      let pixX, pixY;

      if (photoAspect > screenAspect) {
        const visibleW = screenAspect * imgH;
        const offsetX = (imgW - visibleW) / 2;
        pixX = Math.round(offsetX + (cx / width) * visibleW);
        pixY = Math.round((cy / screenHeight) * imgH);
      } else {
        const visibleH = imgW / screenAspect;
        const offsetY = (imgH - visibleH) / 2;
        pixX = Math.round((cx / width) * imgW);
        pixY = Math.round(offsetY + (cy / screenHeight) * visibleH);
      }

      // Pixel averaging logic
      const half = 5;
      const x0 = Math.max(0, pixX - half),
        y0 = Math.max(0, pixY - half);
      const x1 = Math.min(imgW, pixX + half),
        y1 = Math.min(imgH, pixY + half);
      let avgR = 0,
        avgG = 0,
        avgB = 0,
        count = 0;

      for (let py = y0; py < y1; py++) {
        for (let px = x0; px < x1; px++) {
          const idx = (py * imgW + px) * 4;
          avgR += decoded.data[idx];
          avgG += decoded.data[idx + 1];
          avgB += decoded.data[idx + 2];
          count++;
        }
      }
      avgR = Math.round(avgR / count);
      avgG = Math.round(avgG / count);
      avgB = Math.round(avgB / count);

      const result = identifyColor(avgR, avgG, avgB);
      const sampledHex =
        "#" +
        [avgR, avgG, avgB].map((c) => c.toString(16).padStart(2, "0")).join("");

      if (isMountedRef.current) {
        setIdentifiedColor({
          name: result.className,
          hex: sampledHex,
          conf: `${result.confidence}%`,
        });

        // Trigger Audio Feedback if enabled
        if (audio) {
          Speech.stop();
          Speech.speak(result.className, { rate: 1.0 });
        }
      }
    } catch (e) {
      console.log("[ColorID] detection error:", e);
    } finally {
      isProcessingRef.current = false;
      if (isMountedRef.current) setIsDetecting(false);
    }
  };

  const toggleAudio = async () => {
    const newVal = !audio;
    setAudio(newVal);
    await AsyncStorage.setItem("audio_feedback_enabled", newVal.toString());
  };

  return (
    <View style={{ flex: 1, backgroundColor: "#000" }}>
      {device && cameraReady && (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isFocused}
          photo
          enableShutterSound={false}
        />
      )}
      <View
        style={StyleSheet.absoluteFill}
        onStartShouldSetResponder={() => true}
        onResponderMove={(evt) =>
          setCursorPosition({
            x: evt.nativeEvent.locationX,
            y: evt.nativeEvent.locationY,
          })
        }
        onResponderRelease={(evt) => {
          setCursorPosition({
            x: evt.nativeEvent.locationX,
            y: evt.nativeEvent.locationY,
          });
          runDetection(evt.nativeEvent.locationX, evt.nativeEvent.locationY);
        }}
      />

      <SafeAreaView style={{ flex: 1 }} pointerEvents="box-none">
        <View style={styles.camTopBar}>
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <TouchableOpacity
              onPress={() => navigation.goBack()}
              style={{ marginRight: 10 }}
            >
              <Ionicons name="arrow-back" size={24} color="#FFF" />
            </TouchableOpacity>
            <View style={styles.camPill}>
              <Text style={{ color: "#FFF", fontSize: 12, fontWeight: "bold" }}>
                Color Identifier
              </Text>
            </View>
          </View>
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <TouchableOpacity onPress={toggleAudio} style={{ marginRight: 15 }}>
              <Ionicons
                name={audio ? "volume-high" : "volume-mute"}
                size={24}
                color="#FFF"
              />
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setShowModal(true)}>
              <Ionicons name="menu" size={28} color="#FFF" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Reticle */}
        <View
          pointerEvents="none"
          style={[
            localStyles.cursor,
            { top: cursorPosition.y - 50, left: cursorPosition.x - 50 },
          ]}
        >
          <View style={localStyles.reticleLineV} />
          <View style={localStyles.reticleLineH} />
          <View style={localStyles.reticleCircle} />
        </View>

        {/* Result Card */}
        <View style={localStyles.resultCardContainer}>
          <View style={localStyles.resultCard}>
            <View
              style={[
                localStyles.colorPreview,
                { backgroundColor: identifiedColor.hex },
              ]}
            />
            <Text style={localStyles.colorName}>{identifiedColor.name}</Text>
            <Text style={localStyles.calculatingText}>
              {isDetecting ? "Calculating..." : ""}
            </Text>
          </View>
        </View>

        <ModeSelector
          visible={showModal}
          onClose={() => setShowModal(false)}
          navigation={navigation}
          currentMode="Identifier"
        />
      </SafeAreaView>
    </View>
  );
}

const localStyles = StyleSheet.create({
  cursor: {
    position: "absolute",
    width: 100,
    height: 100,
    justifyContent: "center",
    alignItems: "center",
  },
  reticleLineV: {
    width: 2,
    height: 50,
    backgroundColor: "rgba(255,255,255,0.8)",
    position: "absolute",
  },
  reticleLineH: {
    width: 50,
    height: 2,
    backgroundColor: "rgba(255,255,255,0.8)",
    position: "absolute",
  },
  reticleCircle: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: "#FFF",
  },
  resultCardContainer: {
    position: "absolute",
    top: "60%",
    alignSelf: "center",
    pointerEvents: "none",
  },
  resultCard: {
    backgroundColor: "rgba(0,0,0,0.85)",
    padding: 15,
    borderRadius: 12,
    alignItems: "center",
    minWidth: 150,
  },
  colorPreview: {
    width: 30,
    height: 30,
    borderRadius: 15,
    marginBottom: 5,
    borderWidth: 2,
    borderColor: "#FFF",
  },
  colorName: { color: "#FFF", fontWeight: "bold", fontSize: 16 },
  calculatingText: { color: "#CCC", fontSize: 12 },
});

// DEFAULT EXPORT (Must match your navigator import)
export default function ColorIdentifierScreen(props) {
  return (
    <ScreenErrorBoundary navigation={props.navigation}>
      <ColorIdentifierScreenInner {...props} />
    </ScreenErrorBoundary>
  );
}
