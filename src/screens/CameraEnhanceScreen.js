import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";
import { useIsFocused } from "@react-navigation/native";
import * as FileSystem from "expo-file-system";
import * as Haptics from "expo-haptics";
import * as ImageManipulator from "expo-image-manipulator";
import * as MediaLibrary from "expo-media-library";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
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
import { auth } from "../../firebaseConfig"; // Added Firebase Auth import
import {
  applyDaltonization,
  applyHueRotation,
  decodeJpegBase64,
  encodeToDataUri,
} from "../../tensorHelper";
import ModeSelector from "../components/ModeSelector";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";
import { AppLog, ScreenErrorBoundary } from "../utils/logger";

function CameraEnhanceScreenInner({ navigation }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const isFocused = useIsFocused();
  const [cameraPosition, setCameraPosition] = useState("back");
  const [cvdType, setCvdType] = useState("Protan");
  const [algorithm, setAlgorithm] = useState("daltonization");
  const [showModal, setShowModal] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [frozen, setFrozen] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState("");
  const [resultUri, setResultUri] = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);

  const cameraRef = useRef(null);
  const isMountedRef = useRef(true);
  const decodedRef = useRef(null);
  const frozenUriRef = useRef(null);
  const intensityRef = useRef(100);
  const [intensity, setIntensity] = useState(100);
  const device = useCameraDevice(cameraPosition);

  useEffect(() => {
    AsyncStorage.getItem("@recolor_intensity").then((val) => {
      if (val !== null) {
        const v = Number(val);
        intensityRef.current = v;
        setIntensity(v);
      }
    });
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (isMountedRef.current) setCameraReady(true);
    }, 700);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const unsub = navigation.addListener("beforeRemove", () =>
      setCameraReady(false),
    );
    return unsub;
  }, [navigation]);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, []);

  const runEnhancement = useCallback((cvd, algo) => {
    if (!decodedRef.current) return null;
    if (cvd === "Off") return null;
    const { data, width: w, height: h } = decodedRef.current;
    const src = { data, width: w, height: h };
    const enhanced =
      algo === "hue_rotation"
        ? applyHueRotation(src, cvd)
        : applyDaltonization(src, null, cvd);

    const blend = intensityRef.current / 100;
    if (blend >= 1) return encodeToDataUri(enhanced, w, h);

    const blended = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i += 4) {
      blended[i] = Math.round(data[i] * (1 - blend) + enhanced[i] * blend);
      blended[i + 1] = Math.round(
        data[i + 1] * (1 - blend) + enhanced[i + 1] * blend,
      );
      blended[i + 2] = Math.round(
        data[i + 2] * (1 - blend) + enhanced[i + 2] * blend,
      );
      blended[i + 3] = data[i + 3];
    }
    return encodeToDataUri(blended, w, h);
  }, []);

  const handleIntensityChange = useCallback((val) => {
    intensityRef.current = val;
    setIntensity(val);
  }, []);

  const handleIntensityCommit = useCallback(
    (val) => {
      intensityRef.current = val;
      setIntensity(val);
      AsyncStorage.setItem("@recolor_intensity", String(val));
      if (!frozen || !decodedRef.current) return;
      setProcessing(true);
      setProgress("Adjusting...");
      setTimeout(() => {
        const uri = runEnhancement(cvdType, algorithm);
        if (isMountedRef.current) {
          setResultUri(uri || frozenUriRef.current);
          setProcessing(false);
          setProgress("");
        }
      }, 50);
    },
    [frozen, cvdType, algorithm, runEnhancement],
  );

  const handleFreeze = useCallback(async () => {
    if (!cameraRef.current) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    setProcessing(true);
    setProgress("Capturing...");
    AppLog.log("CameraEnhance", "freeze: capturing photo");

    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: "quality",
        enableShutterSound: false,
      });
      if (!photo?.path) throw new Error("takePhoto returned no path");
      const fileUri = `file://${photo.path}`;

      // Save original to gallery
      const { status: mlStatus } = await MediaLibrary.requestPermissionsAsync();
      if (mlStatus === "granted") {
        MediaLibrary.saveToLibraryAsync(fileUri).catch(() => {});
      }

      setProgress("Resizing...");
      const SIM_MAX = 1040;
      const resized = await ImageManipulator.manipulateAsync(
        fileUri,
        [
          {
            resize:
              photo.width >= photo.height
                ? { width: Math.min(SIM_MAX, photo.width) }
                : { height: Math.min(SIM_MAX, photo.height) },
          },
        ],
        {
          base64: true,
          format: ImageManipulator.SaveFormat.JPEG,
          compress: 0.92,
        },
      );
      if (!isMountedRef.current) return;

      frozenUriRef.current = resized.uri;
      setProgress("Decoding...");
      const decoded = decodeJpegBase64(resized.base64);
      decodedRef.current = decoded;
      AppLog.log(
        "CameraEnhance",
        `decoded: ${decoded.width}x${decoded.height}`,
      );

      setProgress("Enhancing...");
      await new Promise((r) => setTimeout(r, 0));
      const uri = runEnhancement(cvdType, algorithm);

      if (isMountedRef.current) {
        setResultUri(uri || frozenUriRef.current);
        setFrozen(true);
        setProcessing(false);
        setProgress("");
      }
    } catch (e) {
      AppLog.log("CameraEnhance", `freeze error: ${e?.message || e}`);
      if (isMountedRef.current) {
        setProcessing(false);
        setProgress("");
        Alert.alert(
          "Error",
          `Processing failed: ${e?.message || "unknown error"}`,
        );
      }
    }
  }, [cvdType, algorithm, runEnhancement]);

  const handleCvdChange = useCallback(
    (newType) => {
      if (newType === cvdType) return;
      setCvdType(newType);
      if (!frozen || !decodedRef.current) return;
      setProcessing(true);
      setProgress("Re-enhancing...");
      setTimeout(() => {
        const uri = runEnhancement(newType, algorithm);
        if (isMountedRef.current) {
          setResultUri(uri || frozenUriRef.current);
          setProcessing(false);
          setProgress("");
        }
      }, 50);
    },
    [cvdType, frozen, algorithm, runEnhancement],
  );

  const handleAlgorithmChange = useCallback(
    (newAlgo) => {
      if (newAlgo === algorithm) return;
      setAlgorithm(newAlgo);
      if (!frozen || !decodedRef.current) return;
      setProcessing(true);
      setProgress("Re-enhancing...");
      setTimeout(() => {
        const uri = runEnhancement(cvdType, newAlgo);
        if (isMountedRef.current) {
          setResultUri(uri || frozenUriRef.current);
          setProcessing(false);
          setProgress("");
        }
      }, 50);
    },
    [algorithm, frozen, cvdType, runEnhancement],
  );

  const handleReset = useCallback(() => {
    setFrozen(false);
    setResultUri(null);
    setProcessing(false);
    setProgress("");
    setShowOriginal(false);
    decodedRef.current = null;
    frozenUriRef.current = null;
    AppLog.log("CameraEnhance", "reset to live preview");
  }, []);

  const handleSave = useCallback(async () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(
      () => {},
    );
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission needed", "Please allow access to save photos.");
        return;
      }
      if (!resultUri) {
        Alert.alert("Error", "Nothing to save.");
        return;
      }

      const b64 = resultUri.split(",")[1];
      const tmpPath = `${FileSystem.documentDirectory}recolor_enhance_${Date.now()}.jpg`;
      await FileSystem.writeAsStringAsync(tmpPath, b64, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Dynamic Album Name based on the current user
      const userName = auth.currentUser?.email?.split("@")[0] || "Guest";
      const albumName = `ReColor_${userName}`;

      const asset = await MediaLibrary.createAssetAsync(tmpPath);
      await MediaLibrary.createAlbumAsync(albumName, asset, false);

      Alert.alert("Saved", `Enhanced photo saved to your ${albumName} album.`);
    } catch (e) {
      AppLog.log("CameraEnhance", `save error: ${e?.message || e}`);
      Alert.alert("Error", "Could not save photo.");
    }
  }, [resultUri]);

  if (!hasPermission) {
    return (
      <View
        style={[
          styles.container,
          { justifyContent: "center", alignItems: "center" },
        ]}
      >
        <Text style={{ textAlign: "center", marginBottom: 20 }}>
          We need camera access for color enhancement.
        </Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const displayUri = frozen
    ? showOriginal
      ? frozenUriRef.current
      : resultUri
    : null;

  return (
    <View style={{ flex: 1, backgroundColor: "#000" }}>
      {device && cameraReady && !frozen && (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isFocused && !frozen}
          photo
          enableShutterSound={false}
        />
      )}
      {frozen && displayUri && (
        <Image
          source={{ uri: displayUri }}
          style={StyleSheet.absoluteFill}
          resizeMode="cover"
        />
      )}

      {processing && (
        <View
          style={{
            ...StyleSheet.absoluteFillObject,
            backgroundColor: "rgba(0,0,0,0.7)",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 10,
          }}
        >
          <ActivityIndicator size="large" color="#FFF" />
          <Text
            style={{
              color: "#FFF",
              fontSize: 14,
              marginTop: 12,
              fontWeight: "600",
            }}
          >
            {progress || "Processing..."}
          </Text>
        </View>
      )}

      {frozen && !processing && (
        <View
          style={{
            position: "absolute",
            top: 120,
            left: 80,
            right: 80,
            bottom: 140,
            zIndex: 1,
          }}
          onStartShouldSetResponder={() => true}
          onResponderGrant={() => setShowOriginal(true)}
          onResponderRelease={() => setShowOriginal(false)}
          onResponderTerminate={() => setShowOriginal(false)}
        />
      )}

      <SafeAreaView style={{ flex: 1 }} pointerEvents="box-none">
        <View style={styles.camTopBar}>
          <TouchableOpacity
            onPress={() => {
              if (frozen) handleReset();
              else navigation.goBack();
            }}
            style={{ padding: 5 }}
          >
            <Ionicons
              name={frozen ? "close" : "arrow-back"}
              size={24}
              color="#FFF"
            />
          </TouchableOpacity>
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <Text
              style={{
                color: "#FFF",
                fontWeight: "bold",
                marginRight: 10,
                textShadowColor: "rgba(0,0,0,0.75)",
                textShadowOffset: { width: -1, height: 1 },
                textShadowRadius: 10,
              }}
            >
              {frozen
                ? cvdType === "Off"
                  ? "Original"
                  : `${cvdType} · ${algorithm === "hue_rotation" ? "Hue Rotate" : "Daltonize"}`
                : "Color Enhancement"}
            </Text>
            {!frozen && (
              <TouchableOpacity onPress={() => setShowModal(true)}>
                <Ionicons name="menu" size={28} color="#FFF" />
              </TouchableOpacity>
            )}
          </View>
        </View>

        {frozen && (
          <View
            style={{
              position: "absolute",
              top: 100,
              right: 20,
              alignItems: "center",
              zIndex: 2,
            }}
          >
            {["Off", "Protan", "Deutan", "Tritan"].map((m) => (
              <TouchableOpacity
                key={m}
                onPress={() => handleCvdChange(m)}
                disabled={processing}
                style={[
                  styles.filterBtn,
                  {
                    backgroundColor:
                      cvdType === m ? COLORS.primary : "rgba(0,0,0,0.5)",
                    marginBottom: 15,
                    opacity: processing ? 0.4 : 1,
                  },
                ]}
              >
                <Text
                  style={{ color: "#FFF", fontWeight: "bold", fontSize: 10 }}
                >
                  {m === "Off" ? "Off" : m.charAt(0)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {frozen && (
          <View
            style={{
              position: "absolute",
              top: 100,
              left: 20,
              alignItems: "center",
              zIndex: 2,
            }}
          >
            {[
              { key: "daltonization", label: "DAL" },
              { key: "hue_rotation", label: "HUE" },
            ].map((a) => (
              <TouchableOpacity
                key={a.key}
                onPress={() => handleAlgorithmChange(a.key)}
                disabled={processing || cvdType === "Off"}
                style={[
                  styles.filterBtn,
                  {
                    backgroundColor:
                      algorithm === a.key ? COLORS.primary : "rgba(0,0,0,0.5)",
                    marginBottom: 15,
                    opacity: processing || cvdType === "Off" ? 0.4 : 1,
                    minWidth: 42,
                  },
                ]}
              >
                <Text
                  style={{ color: "#FFF", fontWeight: "bold", fontSize: 10 }}
                >
                  {a.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {frozen && !processing && (
          <View
            style={{
              position: "absolute",
              top: "55%",
              alignSelf: "center",
              pointerEvents: "none",
            }}
          >
            <View
              style={{
                backgroundColor: "rgba(0,0,0,0.6)",
                paddingHorizontal: 12,
                paddingVertical: 6,
                borderRadius: 10,
              }}
            >
              <Text style={{ color: "#AAA", fontSize: 11 }}>
                {showOriginal ? "Showing original" : "Long-press for original"}
              </Text>
            </View>
          </View>
        )}

        <View
          style={{
            position: "absolute",
            bottom: 30,
            left: 0,
            right: 0,
            zIndex: 2,
          }}
        >
          {frozen && !processing && cvdType !== "Off" && (
            <View
              style={{
                alignItems: "center",
                marginBottom: 12,
                paddingHorizontal: 30,
              }}
            >
              <Text
                style={{
                  color: "rgba(255,255,255,0.75)",
                  fontSize: 11,
                  marginBottom: 2,
                }}
              >
                Intensity: {Math.round(intensity)}%
              </Text>
              <Slider
                style={{ width: "80%", height: 36 }}
                minimumValue={0}
                maximumValue={100}
                step={1}
                value={intensity}
                minimumTrackTintColor={COLORS.primary}
                maximumTrackTintColor="rgba(255,255,255,0.3)"
                thumbTintColor="#FFF"
                onValueChange={handleIntensityChange}
                onSlidingComplete={handleIntensityCommit}
              />
            </View>
          )}
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-around",
              alignItems: "center",
            }}
          >
            {frozen ? (
              <>
                <TouchableOpacity onPress={handleReset}>
                  <Ionicons name="refresh" size={30} color="#FFF" />
                </TouchableOpacity>
                <View style={{ width: 70 }} />
                <TouchableOpacity
                  onPress={handleSave}
                  disabled={!resultUri || processing}
                >
                  <Ionicons
                    name="download-outline"
                    size={30}
                    color={!resultUri || processing ? "#666" : "#FFF"}
                  />
                </TouchableOpacity>
              </>
            ) : (
              <>
                <TouchableOpacity
                  onPress={() =>
                    setCameraPosition((p) => (p === "back" ? "front" : "back"))
                  }
                >
                  <Ionicons name="camera-reverse" size={30} color="#FFF" />
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.shutterBtn}
                  onPress={handleFreeze}
                >
                  <View
                    style={{
                      width: 60,
                      height: 60,
                      borderRadius: 30,
                      backgroundColor: "#FFF",
                      justifyContent: "center",
                      alignItems: "center",
                    }}
                  >
                    <Ionicons name="snow" size={24} color="#333" />
                  </View>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => navigation.navigate("CVDGallery")}
                >
                  <Ionicons name="images" size={30} color="#FFF" />
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>

        <ModeSelector
          visible={showModal}
          onClose={() => setShowModal(false)}
          navigation={navigation}
          currentMode="Enhancement"
        />
      </SafeAreaView>
    </View>
  );
}

export default function CameraEnhanceScreen(props) {
  return (
    <ScreenErrorBoundary navigation={props.navigation}>
      <CameraEnhanceScreenInner {...props} />
    </ScreenErrorBoundary>
  );
}
