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
import { auth } from "../../firebaseConfig";
import {
  applyDaltonization,
  applyHueRotation,
  decodeJpegBase64,
  encodeToDataUri,
} from "../../tensorHelper";
import ModeSelector from "../components/ModeSelector";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";
import { ScreenErrorBoundary } from "../utils/logger";

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
  }, [hasPermission]);

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

    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: "quality",
        enableShutterSound: false,
      });
      const fileUri = `file://${photo.path}`;

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
      const decoded = decodeJpegBase64(resized.base64);
      decodedRef.current = decoded;

      const uri = runEnhancement(cvdType, algorithm);

      if (isMountedRef.current) {
        setResultUri(uri || frozenUriRef.current);
        setFrozen(true);
        setProcessing(false);
        setProgress("");
      }
    } catch (e) {
      if (isMountedRef.current) {
        setProcessing(false);
        setProgress("");
        Alert.alert("Error", "Processing failed.");
      }
    }
  }, [cvdType, algorithm, runEnhancement]);

  const handleSave = useCallback(async () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(
      () => {},
    );
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== "granted") {
        Alert.alert(
          "Permission needed",
          "Please allow gallery access in settings.",
        );
        return;
      }
      if (!resultUri) return;

      const b64 = resultUri.split(",")[1];
      const tmpPath = `${FileSystem.documentDirectory}recolor_enhance_${Date.now()}.jpg`;
      await FileSystem.writeAsStringAsync(tmpPath, b64, { encoding: "base64" });

      const asset = await MediaLibrary.createAssetAsync(tmpPath);
      const userName = auth.currentUser?.email?.split("@")[0] || "Guest";
      const albumName = `ReColor_${userName}`;

      const album = await MediaLibrary.getAlbumAsync(albumName);
      if (!album) {
        await MediaLibrary.createAlbumAsync(albumName, asset, false);
      } else {
        await MediaLibrary.addAssetsToAlbumAsync([asset], album, false);
      }

      Alert.alert("Saved", `Photo added to your ${albumName} album.`);
    } catch (e) {
      Alert.alert("Error", "Could not save photo.");
    }
  }, [resultUri]);

  const handleReset = useCallback(() => {
    setFrozen(false);
    setResultUri(null);
    setProcessing(false);
    setProgress("");
    setShowOriginal(false);
    decodedRef.current = null;
    frozenUriRef.current = null;
  }, []);

  if (!hasPermission) {
    return (
      <View
        style={[
          styles.container,
          { justifyContent: "center", alignItems: "center" },
        ]}
      >
        <Text style={{ textAlign: "center", marginBottom: 20 }}>
          Camera access is needed.
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
        />
      )}

      <SafeAreaView style={{ flex: 1 }} pointerEvents="box-none">
        <View style={styles.camTopBar}>
          <TouchableOpacity
            onPress={() => (frozen ? handleReset() : navigation.goBack())}
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
              style={{ color: "#FFF", fontWeight: "bold", marginRight: 10 }}
            >
              {frozen
                ? cvdType === "Off"
                  ? "Original"
                  : `${cvdType} Enhanced`
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
                onPress={() => setCvdType(m)}
                style={[
                  styles.filterBtn,
                  {
                    backgroundColor:
                      cvdType === m ? COLORS.primary : "rgba(0,0,0,0.5)",
                    marginBottom: 15,
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
              <Text style={{ color: "rgba(255,255,255,0.75)", fontSize: 11 }}>
                Intensity: {Math.round(intensity)}%
              </Text>
              <Slider
                style={{ width: "80%", height: 36 }}
                minimumValue={0}
                maximumValue={100}
                value={intensity}
                onValueChange={handleIntensityChange}
                onSlidingComplete={handleIntensityCommit}
                minimumTrackTintColor={COLORS.primary}
                thumbTintColor="#FFF"
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
                <TouchableOpacity onPress={handleSave} disabled={!resultUri}>
                  <Ionicons
                    name="download-outline"
                    size={30}
                    color={!resultUri ? "#666" : "#FFF"}
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
                  <Ionicons name="snow" size={24} color="#333" />
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => navigation.navigate("EnhanceGallery")}
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
