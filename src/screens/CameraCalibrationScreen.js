/**
 * CameraCalibrationScreen
 *
 * One-tap white-balance calibration. User points the camera at a plain white
 * surface (paper, wall, fabric) so the centred guide rectangle is filled with
 * that surface, taps "Calibrate", and the app captures the framed patch,
 * computes per-channel scalars, and saves them to AsyncStorage.
 *
 * Saved value is then used by ColorIdentifier and CameraEnhance to correct
 * cross-device color drift before any classification or enhancement runs.
 */

import { Ionicons } from "@expo/vector-icons";
import { useIsFocused } from "@react-navigation/native";
import * as ImageManipulator from "expo-image-manipulator";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from "react-native-vision-camera";
import { decodeJpegBase64 } from "../../tensorHelper";
import Header from "../components/Header";
import { COLORS, RADIUS, SHADOW, SPACING } from "../theme/colors";
import {
  clearCalibration,
  deriveScalarsFromWhitePatch,
  loadCalibration,
  saveCalibration,
} from "../utils/cameraCalibration";

const { width: SW, height: SH } = Dimensions.get("window");
const GUIDE_SIZE = Math.min(SW, SH) * 0.5; // centred square guide

export default function CameraCalibrationScreen({ navigation }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const isFocused = useIsFocused();
  const device = useCameraDevice("back");
  const cameraRef = useRef(null);
  const isMountedRef = useRef(true);

  const [cameraReady, setCameraReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [existing, setExisting] = useState(null);
  const [preview, setPreview] = useState(null); // {avgR, avgG, avgB, scalars}

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    loadCalibration().then(setExisting);
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const handleCalibrate = async () => {
    if (!cameraRef.current || busy) return;
    setBusy(true);
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

      // Sample the centred square that corresponds to the on-screen guide.
      // Cover-mode mapping: the photo aspect may differ from the screen, so
      // the image fills one axis and crops the other. We sample the centre,
      // which is always visible regardless of the cover crop direction.
      const sampleSize = Math.min(imgW, imgH) * 0.45;
      const cx = imgW / 2;
      const cy = imgH / 2;
      const x0 = Math.max(0, Math.floor(cx - sampleSize / 2));
      const y0 = Math.max(0, Math.floor(cy - sampleSize / 2));
      const x1 = Math.min(imgW, Math.floor(cx + sampleSize / 2));
      const y1 = Math.min(imgH, Math.floor(cy + sampleSize / 2));

      let sumR = 0,
        sumG = 0,
        sumB = 0,
        count = 0;
      for (let py = y0; py < y1; py += 2) {
        for (let px = x0; px < x1; px += 2) {
          const idx = (py * imgW + px) * 4;
          sumR += decoded.data[idx];
          sumG += decoded.data[idx + 1];
          sumB += decoded.data[idx + 2];
          count++;
        }
      }
      const avgR = Math.round(sumR / count);
      const avgG = Math.round(sumG / count);
      const avgB = Math.round(sumB / count);

      // Sanity check: if the patch is very dark, the user is not pointing at
      // a white reference. Refuse rather than silently saving a wrong value.
      const luma = 0.299 * avgR + 0.587 * avgG + 0.114 * avgB;
      if (luma < 100) {
        Alert.alert(
          "Too Dark",
          "The framed area is too dark to use as a white reference. Point the camera at a brighter white surface (paper, wall, fabric) in good lighting and try again.",
        );
        return;
      }

      const scalars = deriveScalarsFromWhitePatch(avgR, avgG, avgB);
      if (isMountedRef.current) {
        setPreview({ avgR, avgG, avgB, scalars });
      }
    } catch (e) {
      console.warn("[Calibration] capture failed", e);
      Alert.alert("Capture Failed", "Could not capture the calibration sample. Please try again.");
    } finally {
      if (isMountedRef.current) setBusy(false);
    }
  };

  const handleSave = async () => {
    if (!preview) return;
    const saved = await saveCalibration(
      preview.scalars.rScale,
      preview.scalars.gScale,
      preview.scalars.bScale,
    );
    setExisting(saved);
    setPreview(null);
    Alert.alert(
      "Calibrated",
      "Camera calibration saved. The Color Identifier and Camera Enhancement will now use this calibration.",
      [{ text: "Done", onPress: () => navigation.goBack() }],
    );
  };

  const handleRetry = () => setPreview(null);

  const handleReset = () => {
    Alert.alert(
      "Reset Calibration",
      "This removes your saved calibration. The cameras will fall back to automatic white-balance.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset",
          style: "destructive",
          onPress: async () => {
            await clearCalibration();
            setExisting(null);
            setPreview(null);
          },
        },
      ],
    );
  };

  // ── Permission gate ──
  if (!hasPermission) {
    return (
      <View style={styles.permissionWrap}>
        <Header title="Camera Calibration" back />
        <View style={styles.permissionBody}>
          <Ionicons name="camera-outline" size={48} color={COLORS.textLight} />
          <Text style={styles.permissionText}>
            Camera access is required to calibrate.
          </Text>
          <TouchableOpacity
            style={styles.primaryBtn}
            onPress={requestPermission}
          >
            <Text style={styles.primaryBtnText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.permissionWrap}>
        <Header title="Camera Calibration" back />
        <View style={styles.permissionBody}>
          <ActivityIndicator color={COLORS.primary} />
          <Text style={styles.permissionText}>Loading camera…</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isFocused && !preview}
        photo={true}
        onInitialized={() => setCameraReady(true)}
      />

      {/* Top header — opaque so it's readable over camera feed */}
      <View style={styles.topBar}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.iconBtn}>
          <Ionicons name="chevron-back" size={28} color="#FFF" />
        </TouchableOpacity>
        <Text style={styles.title}>Camera Calibration</Text>
        <View style={styles.iconBtn} />
      </View>

      {/* Centred guide rectangle */}
      {!preview && (
        <View pointerEvents="none" style={styles.guideOverlay}>
          <View style={styles.guideBox}>
            <View style={[styles.corner, styles.cornerTL]} />
            <View style={[styles.corner, styles.cornerTR]} />
            <View style={[styles.corner, styles.cornerBL]} />
            <View style={[styles.corner, styles.cornerBR]} />
          </View>
        </View>
      )}

      {/* Bottom panel */}
      <View style={styles.bottomPanel}>
        {preview ? (
          <PreviewPanel
            preview={preview}
            onSave={handleSave}
            onRetry={handleRetry}
          />
        ) : (
          <InstructionPanel
            existing={existing}
            cameraReady={cameraReady}
            busy={busy}
            onCalibrate={handleCalibrate}
            onReset={handleReset}
          />
        )}
      </View>
    </View>
  );
}

function InstructionPanel({ existing, cameraReady, busy, onCalibrate, onReset }) {
  return (
    <>
      <Text style={styles.instructionTitle}>How to Calibrate</Text>
      <Text style={styles.instructionText}>
        Hold a plain <Text style={{ fontWeight: "700" }}>white</Text> sheet of
        paper (or a similar white surface) so it fills the framed area in your
        normal lighting. Then tap Calibrate.
      </Text>

      {existing && (
        <View style={styles.statusPill}>
          <Ionicons name="checkmark-circle" size={14} color="#FFF" />
          <Text style={styles.statusText}>
            Currently calibrated · R{existing.rScale.toFixed(2)} G
            {existing.gScale.toFixed(2)} B{existing.bScale.toFixed(2)}
          </Text>
        </View>
      )}

      <TouchableOpacity
        style={[styles.primaryBtn, (!cameraReady || busy) && styles.btnDisabled]}
        onPress={onCalibrate}
        disabled={!cameraReady || busy}
      >
        {busy ? (
          <ActivityIndicator color="#FFF" />
        ) : (
          <>
            <Ionicons name="color-wand" size={20} color="#FFF" />
            <Text style={styles.primaryBtnText}>Calibrate</Text>
          </>
        )}
      </TouchableOpacity>

      {existing && (
        <TouchableOpacity style={styles.secondaryBtn} onPress={onReset}>
          <Text style={styles.secondaryBtnText}>Reset Calibration</Text>
        </TouchableOpacity>
      )}
    </>
  );
}

function PreviewPanel({ preview, onSave, onRetry }) {
  const { avgR, avgG, avgB, scalars } = preview;
  const sampledHex =
    "#" + [avgR, avgG, avgB].map((c) => c.toString(16).padStart(2, "0")).join("");
  const correctedR = Math.min(255, Math.round(avgR * scalars.rScale));
  const correctedG = Math.min(255, Math.round(avgG * scalars.gScale));
  const correctedB = Math.min(255, Math.round(avgB * scalars.bScale));
  const correctedHex =
    "#" +
    [correctedR, correctedG, correctedB]
      .map((c) => c.toString(16).padStart(2, "0"))
      .join("");

  return (
    <>
      <Text style={styles.instructionTitle}>Sample Captured</Text>
      <View style={styles.swatchRow}>
        <View style={styles.swatchCell}>
          <View style={[styles.swatch, { backgroundColor: sampledHex }]} />
          <Text style={styles.swatchLabel}>Captured</Text>
          <Text style={styles.swatchSub}>
            {avgR}, {avgG}, {avgB}
          </Text>
        </View>
        <Ionicons name="arrow-forward" size={20} color={COLORS.textLight} />
        <View style={styles.swatchCell}>
          <View style={[styles.swatch, { backgroundColor: correctedHex }]} />
          <Text style={styles.swatchLabel}>Corrected</Text>
          <Text style={styles.swatchSub}>
            {correctedR}, {correctedG}, {correctedB}
          </Text>
        </View>
      </View>
      <Text style={styles.scalarLine}>
        R×{scalars.rScale.toFixed(3)}  G×{scalars.gScale.toFixed(3)}  B×
        {scalars.bScale.toFixed(3)}
      </Text>

      <TouchableOpacity style={styles.primaryBtn} onPress={onSave}>
        <Ionicons name="save" size={20} color="#FFF" />
        <Text style={styles.primaryBtnText}>Save Calibration</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.secondaryBtn} onPress={onRetry}>
        <Text style={styles.secondaryBtnText}>Try Again</Text>
      </TouchableOpacity>
    </>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#000" },

  permissionWrap: { flex: 1, backgroundColor: COLORS.background },
  permissionBody: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.md,
    padding: SPACING.lg,
  },
  permissionText: { color: COLORS.text, fontSize: 14, textAlign: "center" },

  topBar: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    paddingTop: 48,
    paddingHorizontal: SPACING.md,
    paddingBottom: SPACING.sm,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "rgba(0,0,0,0.45)",
    zIndex: 10,
  },
  iconBtn: { width: 36, height: 36, alignItems: "center", justifyContent: "center" },
  title: { color: "#FFF", fontWeight: "800", fontSize: 16 },

  guideOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
  },
  guideBox: {
    width: GUIDE_SIZE,
    height: GUIDE_SIZE,
    borderWidth: 2,
    borderColor: "rgba(255,255,255,0.85)",
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.05)",
  },
  corner: { position: "absolute", width: 22, height: 22, borderColor: "#FFF" },
  cornerTL: { top: -2, left: -2, borderTopWidth: 4, borderLeftWidth: 4, borderTopLeftRadius: 12 },
  cornerTR: { top: -2, right: -2, borderTopWidth: 4, borderRightWidth: 4, borderTopRightRadius: 12 },
  cornerBL: { bottom: -2, left: -2, borderBottomWidth: 4, borderLeftWidth: 4, borderBottomLeftRadius: 12 },
  cornerBR: { bottom: -2, right: -2, borderBottomWidth: 4, borderRightWidth: 4, borderBottomRightRadius: 12 },

  bottomPanel: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.lg,
    paddingBottom: SPACING.xl,
    backgroundColor: "rgba(20,20,20,0.92)",
    borderTopLeftRadius: RADIUS.xl,
    borderTopRightRadius: RADIUS.xl,
    gap: SPACING.sm,
  },
  instructionTitle: { color: "#FFF", fontWeight: "800", fontSize: 16 },
  instructionText: { color: "#DDD", fontSize: 13, lineHeight: 19 },

  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 6,
    backgroundColor: COLORS.success,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 14,
    marginTop: 2,
  },
  statusText: { color: "#FFF", fontSize: 11, fontWeight: "700" },

  primaryBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: COLORS.primary,
    paddingVertical: 14,
    borderRadius: RADIUS.md,
    marginTop: SPACING.sm,
    ...SHADOW.sm,
  },
  primaryBtnText: { color: "#FFF", fontWeight: "800", fontSize: 15 },
  btnDisabled: { opacity: 0.5 },

  secondaryBtn: { paddingVertical: 12, alignItems: "center" },
  secondaryBtnText: { color: "#FFF", opacity: 0.7, fontSize: 13 },

  swatchRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.md,
    marginVertical: SPACING.sm,
  },
  swatchCell: { alignItems: "center", gap: 4 },
  swatch: {
    width: 64,
    height: 64,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },
  swatchLabel: { color: "#FFF", fontSize: 11, fontWeight: "700" },
  swatchSub: { color: "#AAA", fontSize: 10 },
  scalarLine: { color: "#AAA", fontSize: 11, textAlign: "center" },
});
