import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";
import { useIsFocused } from "@react-navigation/native";
import * as FileSystem from "expo-file-system/legacy";
import * as Haptics from "expo-haptics";
import * as ImageManipulator from "expo-image-manipulator";
import * as MediaLibrary from "expo-media-library";
import { collection, getDocs, limit, orderBy, query } from "firebase/firestore";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Dimensions,
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
import { auth, db } from "../../firebaseConfig";
import {
  applyDaltonization,
  applyHueRotation,
  decodeJpegBase64,
  encodeToDataUri,
} from "../../tensorHelper";
import ModeSelector from "../components/ModeSelector";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";
import {
  applyCalibrationToBuffer,
  resolveCalibration,
} from "../utils/cameraCalibration";
import { ScreenErrorBoundary } from "../utils/logger";

const { width } = Dimensions.get("window");

function CameraEnhanceScreenInner({ navigation }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const isFocused = useIsFocused();
  const [cameraPosition, setCameraPosition] = useState("back");
  const [cvdType, setCvdType] = useState("Off");
  const [algorithm, setAlgorithm] = useState("daltonization");
  const [showModal, setShowModal] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [frozen, setFrozen] = useState(false);
  const [processing, setProcessing] = useState(false);
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
        intensityRef.current = Number(val);
        setIntensity(Number(val));
      }
    });

    const fetchCVDDefault = async () => {
      try {
        let diag = await AsyncStorage.getItem("@recolor_latest_diagnosis");

        if (!diag && auth.currentUser) {
          const q = query(
            collection(db, "users", auth.currentUser.uid, "history"),
            orderBy("date", "desc"),
            limit(1),
          );
          const snap = await getDocs(q);
          if (!snap.empty) {
            diag = snap.docs[0].data().diagnosis || "";
            await AsyncStorage.setItem("@recolor_latest_diagnosis", diag);
          }
        }

        if (diag) {
          if (diag.includes("Protan")) setCvdType("Protan");
          else if (diag.includes("Deutan")) setCvdType("Deutan");
          else if (diag.includes("Tritan")) setCvdType("Tritan");
          else setCvdType("Off");
        } else {
          setCvdType("Off");
        }
      } catch (e) {
        console.warn("Failed to fetch CVD default", e);
      }
    };

    const unsubscribe = navigation.addListener("focus", () => {
      fetchCVDDefault();
    });

    fetchCVDDefault();

    isMountedRef.current = true;
    const timer = setTimeout(() => {
      if (isMountedRef.current) setCameraReady(true);
    }, 700);
    return () => {
      isMountedRef.current = false;
      clearTimeout(timer);
      unsubscribe();
    };
  }, [navigation]);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission]);

  const runEnhancement = useCallback((cvd, algo) => {
    if (!decodedRef.current || cvd === "Off") return null;
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

  const handleIntensityCommit = (val) => {
    intensityRef.current = val;
    setIntensity(val);
    AsyncStorage.setItem("@recolor_intensity", String(val));
    if (frozen && decodedRef.current) {
      setProcessing(true);
      setTimeout(() => {
        setResultUri(
          runEnhancement(cvdType, algorithm) || frozenUriRef.current,
        );
        setProcessing(false);
      }, 50);
    }
  };

  const handleFreeze = async () => {
    if (!cameraRef.current) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    setProcessing(true);
    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: "quality",
        enableShutterSound: false,
      });
      const resized = await ImageManipulator.manipulateAsync(
        `file://${photo.path}`,
        [{ resize: { width: 1040 } }],
        {
          base64: true,
          format: ImageManipulator.SaveFormat.JPEG,
          compress: 0.92,
        },
      );
      if (!isMountedRef.current) return;
      frozenUriRef.current = resized.uri;
      decodedRef.current = decodeJpegBase64(resized.base64);

      const calib = await resolveCalibration(decodedRef.current.data);
      applyCalibrationToBuffer(decodedRef.current.data, calib);

      setResultUri(runEnhancement(cvdType, algorithm) || resized.uri);
      setFrozen(true);
    } catch (e) {
      Alert.alert("Error", "Processing failed.");
    } finally {
      setProcessing(false);
    }
  };

  const handleSave = async () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(
      () => {},
    );
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== "granted") {
        return Alert.alert(
          "Permission needed",
          "Allow gallery access in settings.",
        );
      }

      const targetUri = showOriginal ? frozenUriRef.current : resultUri;
      if (!targetUri) return;

      let b64;
      if (targetUri.startsWith("data:")) {
        b64 = targetUri.split(",")[1];
      } else {
        b64 = await FileSystem.readAsStringAsync(targetUri, {
          encoding: "base64",
        });
      }

      const tmpPath = `${FileSystem.documentDirectory}recolor_final_save_${Date.now()}.jpg`;
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

      await FileSystem.deleteAsync(tmpPath, { idempotent: true });
      Alert.alert("Saved", "Photo added to your ReColor album.");
    } catch (e) {
      Alert.alert("Error", "Could not save photo.");
    }
  };

  const handleReset = () => {
    setFrozen(false);
    setResultUri(null);
    decodedRef.current = null;
    frozenUriRef.current = null;
  };

  const showInfo = () => {
    Alert.alert(
      "How to use Enhancement",
      "BEFORE CAPTURE:\nFrame your subject and tap the shutter button.\n\nAFTER CAPTURE:\nYour diagnosed filter is automatically applied. You can change the filter type or adjust the intensity slider to enhance color distinguishability.",
    );
  };

  const displayUri = showOriginal ? frozenUriRef.current : resultUri;

  return (
    <View style={{ flex: 1, backgroundColor: "#000" }}>
      {!frozen ? (
        device &&
        cameraReady && (
          <Camera
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={isFocused}
            photo
          />
        )
      ) : (
        <Image
          source={{ uri: displayUri }}
          style={StyleSheet.absoluteFill}
          resizeMode="cover"
        />
      )}

      {processing && (
        <View
          style={[
            StyleSheet.absoluteFill,
            {
              backgroundColor: "rgba(0,0,0,0.7)",
              justifyContent: "center",
              alignItems: "center",
              zIndex: 10,
            },
          ]}
        >
          <ActivityIndicator size="large" color="#FFF" />
          <Text style={{ color: "#FFF", marginTop: 12 }}>Processing...</Text>
        </View>
      )}

      {frozen && !processing && (
        <View
          style={{
            position: "absolute",
            top: 120,
            bottom: 200,
            left: 50,
            right: 50,
            zIndex: 5,
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
          >
            <Ionicons
              name={frozen ? "close" : "arrow-back"}
              size={26}
              color="#FFF"
            />
          </TouchableOpacity>
          <Text style={{ color: "#FFF", fontWeight: "bold" }}>
            {frozen
              ? showOriginal
                ? "Original"
                : `${cvdType} Enhanced`
              : "Color Enhancement"}
          </Text>
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <TouchableOpacity onPress={showInfo} style={{ marginRight: 15 }}>
              <Ionicons
                name="information-circle-outline"
                size={26}
                color="#FFF"
              />
            </TouchableOpacity>
            {!frozen && (
              <TouchableOpacity onPress={() => setShowModal(true)}>
                <Ionicons name="menu" size={28} color="#FFF" />
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* ALGORITHM BUTTONS (Visible ONLY after capture) */}
        {frozen && (
          <View
            style={{
              position: "absolute",
              top: 110,
              left: 20,
              gap: 10,
              zIndex: 10,
            }}
          >
            {["daltonization", "hue_rotation"].map((a) => (
              <TouchableOpacity
                key={a}
                onPress={() => {
                  setAlgorithm(a);
                  setProcessing(true);
                  setTimeout(() => {
                    setResultUri(runEnhancement(cvdType, a));
                    setProcessing(false);
                  }, 50);
                }}
                style={[
                  styles.filterBtn,
                  {
                    backgroundColor:
                      algorithm === a ? COLORS.secondary : "rgba(0,0,0,0.5)",
                    width: 80,
                    height: 40,
                    borderRadius: 10,
                  },
                ]}
              >
                <Text
                  style={{ color: "#FFF", fontSize: 10, fontWeight: "bold" }}
                >
                  {a === "daltonization" ? "ADAPT" : "HUE"}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {/* CVD TYPE BUTTONS (Visible ONLY after capture) */}
        {frozen && (
          <View
            style={{
              position: "absolute",
              top: 110,
              right: 20,
              alignItems: "center",
              zIndex: 10,
            }}
          >
            {["Off", "Protan", "Deutan", "Tritan"].map((m) => (
              <TouchableOpacity
                key={m}
                onPress={() => {
                  setCvdType(m);
                  setProcessing(true);
                  setTimeout(() => {
                    setResultUri(runEnhancement(m, algorithm));
                    setProcessing(false);
                  }, 50);
                }}
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

        <View style={{ position: "absolute", bottom: 40, left: 0, right: 0 }}>
          {frozen && cvdType !== "Off" && (
            <View style={{ alignItems: "center", marginBottom: 20 }}>
              <Text style={{ color: "#FFF", fontSize: 12 }}>
                Intensity: {Math.round(intensity)}%
              </Text>
              <Slider
                style={{ width: "80%", height: 40 }}
                minimumValue={0}
                maximumValue={100}
                value={intensity}
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
                  <Ionicons name="refresh" size={32} color="#FFF" />
                </TouchableOpacity>
                <TouchableOpacity onPress={handleSave}>
                  <Ionicons name="download-outline" size={32} color="#FFF" />
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
      </SafeAreaView>
      <ModeSelector
        visible={showModal}
        onClose={() => setShowModal(false)}
        navigation={navigation}
        currentMode="Enhancement"
      />
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
