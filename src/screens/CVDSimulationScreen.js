import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useIsFocused } from "@react-navigation/native";
import {
  Canvas,
  RuntimeShader,
  Image as SkiaImage,
  useCanvasRef,
  useImage,
} from "@shopify/react-native-skia";
import * as FileSystem from "expo-file-system/legacy";
import * as Haptics from "expo-haptics";
import * as ImageManipulator from "expo-image-manipulator";
import * as MediaLibrary from "expo-media-library";
import { collection, getDocs, limit, orderBy, query } from "firebase/firestore";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Dimensions,
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
import { getCVDRows } from "../../tensorHelper";
import ModeSelector from "../components/ModeSelector";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";
import { loadCalibration } from "../utils/cameraCalibration";
import { CVD_EFFECT } from "../utils/constants";
import { ScreenErrorBoundary } from "../utils/logger";

const { width, height: screenHeight } = Dimensions.get("window");

function CVDSimulationScreenInner({ navigation, route }) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const isFocused = useIsFocused();
  const [cameraPosition, setCameraPosition] = useState("back");
  const [cvdType, setCvdType] = useState(
    route?.params?.initialCvdType || "Off",
  );
  const [showModal, setShowModal] = useState(false);
  const [frozen, setFrozen] = useState(false);
  const [frozenUri, setFrozenUri] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [calib, setCalib] = useState({ rScale: 1, gScale: 1, bScale: 1 });

  const cameraRef = useRef(null);
  const isMountedRef = useRef(true);
  const canvasRef = useCanvasRef();
  const device = useCameraDevice(cameraPosition);
  const skImage = useImage(frozenUri);
  const cvdUniforms = useMemo(() => {
    const rows = getCVDRows(cvdType);
    return {
      ...rows,
      calib: [calib.rScale, calib.gScale, calib.bScale],
    };
  }, [cvdType, calib]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission]);

  useEffect(() => {
    const apply = (saved) => {
      if (saved && isMountedRef.current) {
        setCalib({
          rScale: saved.rScale,
          gScale: saved.gScale,
          bScale: saved.bScale,
        });
      }
    };
    loadCalibration().then(apply);
    const unsub = navigation.addListener("focus", () =>
      loadCalibration().then(apply),
    );
    return unsub;
  }, [navigation]);

  useEffect(() => {
    if (!route?.params?.initialCvdType) {
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
          }
        } catch (e) {
          console.warn("Failed to fetch CVD default", e);
        }
      };
      fetchCVDDefault();
    }
  }, [route?.params?.initialCvdType]);

  const handleFreeze = useCallback(async () => {
    if (!cameraRef.current) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    setProcessing(true);
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
        { format: ImageManipulator.SaveFormat.JPEG, compress: 0.92 },
      );
      if (!isMountedRef.current) return;
      setFrozenUri(resized.uri);
      setFrozen(true);
      setProcessing(false);
    } catch (e) {
      if (isMountedRef.current) {
        setProcessing(false);
        Alert.alert("Error", "Capture failed.");
      }
    }
  }, []);

  const handleSave = useCallback(async () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(
      () => {},
    );
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission needed", "Allow gallery access in settings.");
        return;
      }

      const snapshot = canvasRef.current?.makeImageSnapshot();
      if (!snapshot) return;

      const b64 = snapshot.encodeToBase64();
      const tmpPath = `${FileSystem.documentDirectory}recolor_sim_${Date.now()}.png`;

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
      Alert.alert("Saved", `Simulation saved to ${albumName}.`);
    } catch (e) {
      Alert.alert("Error", "Could not save photo.");
    }
  }, [cvdType]);

  const handleReset = useCallback(() => {
    setFrozen(false);
    setFrozenUri(null);
    setProcessing(false);
  }, []);

  const showInfo = () => {
    if (frozen) {
      Alert.alert(
        "CVD Simulation Guide",
        "• PROTAN (Red-Blindness):\nSimulates reduced red-cone sensitivity. Reds appear darker and blend with greens; warm colors look olive-brown.\n\n• DEUTAN (Green-Blindness):\nSimulates reduced green-cone sensitivity. Reds and greens blend; colors shift to a yellow-brown spectrum.\n\n• TRITAN (Blue-Blindness):\nSimulates blue-cone deficiency (rare). Blues look greenish, and yellows blend with pinks/violets.\n\n• OFF:\nDisplays the original un-simulated color spectrum.",
      );
    } else {
      Alert.alert(
        "How to use Simulation",
        "BEFORE CAPTURE:\nFrame your subject and tap the shutter button.\n\nAFTER CAPTURE:\nYour screening result is simulated automatically. You can select other types to see how different color vision deficiencies perceive the image.",
      );
    }
  };

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

  return (
    <View style={{ flex: 1, backgroundColor: "#000" }}>
      {device && !frozen && (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isFocused && !frozen}
          photo
        />
      )}
      {frozen && skImage && (
        <Canvas ref={canvasRef} style={StyleSheet.absoluteFill}>
          <SkiaImage
            image={skImage}
            x={0}
            y={0}
            width={width}
            height={screenHeight}
            fit="cover"
          >
            {cvdType !== "Off" && CVD_EFFECT && (
              <RuntimeShader source={CVD_EFFECT} uniforms={cvdUniforms} />
            )}
          </SkiaImage>
        </Canvas>
      )}

      {(processing || (frozen && frozenUri && !skImage)) && (
        <View
          style={{
            ...StyleSheet.absoluteFillObject,
            backgroundColor: "rgba(0,0,0,0.6)",
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
            {processing ? "Capturing..." : "Loading image..."}
          </Text>
        </View>
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
          <Text style={{ color: "#FFF", fontWeight: "bold" }}>
            {frozen ? `${cvdType} Simulation` : "ColorBlind Simulation"}
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

        {/* Safety Disclaimer overlay */}
        <View style={{
          backgroundColor: 'rgba(0,0,0,0.6)',
          paddingVertical: 6,
          paddingHorizontal: 12,
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'row',
          gap: 6
        }}>
          <Ionicons name="warning" size={11} color="#FFF5F5" style={{ opacity: 0.8 }} />
          <Text style={{ color: '#FFF', fontSize: 9, fontWeight: '700', letterSpacing: 0.5, opacity: 0.8 }}>
            SIMULATION ONLY • ACCURACY DEPENDS ON DEVICE SENSOR
          </Text>
        </View>

        {/* CVD TYPE BUTTONS (Visible ONLY after capture) */}
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
                <TouchableOpacity onPress={handleSave} disabled={!skImage}>
                  <Ionicons
                    name="download-outline"
                    size={30}
                    color={!skImage ? "#666" : "#FFF"}
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
          currentMode="Simulation"
        />
      </SafeAreaView>
    </View>
  );
}

export default function CVDSimulationScreen(props) {
  return (
    <ScreenErrorBoundary navigation={props.navigation}>
      <CVDSimulationScreenInner {...props} />
    </ScreenErrorBoundary>
  );
}
