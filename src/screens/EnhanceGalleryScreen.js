import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Slider from "@react-native-community/slider";
import * as FileSystem from "expo-file-system";
import * as ImageManipulator from "expo-image-manipulator";
import * as ImagePicker from "expo-image-picker";
import * as MediaLibrary from "expo-media-library";
import { useEffect, useState } from "react";
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
import { auth } from "../../firebaseConfig";
import {
    applyDaltonization,
    applyHueRotation,
    decodeJpegBase64,
    encodeToDataUri,
} from "../../tensorHelper";
import { COLORS } from "../theme/colors";
import { styles } from "../theme/styles";

const { width } = Dimensions.get("window");

export default function EnhanceGalleryScreen({ navigation }) {
  const [originalUri, setOriginalUri] = useState(null);
  const [displayUri, setDisplayUri] = useState(null);
  const [mode, setMode] = useState("Off");
  const [algo, setAlgo] = useState("daltonization");
  const [intensity, setIntensity] = useState(100);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem("@recolor_intensity").then(
      (val) => val && setIntensity(Number(val)),
    );
  }, []);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 1,
    });
    if (!result.canceled) {
      setOriginalUri(result.assets[0].uri);
      setDisplayUri(result.assets[0].uri);
      setMode("Off");
    }
  };

  const processFilter = async (cvdMode, currentAlgo, currentInt) => {
    if (cvdMode === "Off" || !originalUri) {
      setDisplayUri(originalUri);
      return;
    }
    setProcessing(true);
    try {
      const resized = await ImageManipulator.manipulateAsync(
        originalUri,
        [{ resize: { width: 1040 } }],
        { base64: true },
      );
      const decoded = decodeJpegBase64(resized.base64);
      const src = {
        data: decoded.data,
        width: decoded.width,
        height: decoded.height,
      };

      const enhanced =
        currentAlgo === "hue_rotation"
          ? applyHueRotation(src, cvdMode)
          : applyDaltonization(src, null, cvdMode);

      const blend = currentInt / 100;
      const blended = new Uint8Array(decoded.data.length);
      for (let i = 0; i < decoded.data.length; i += 4) {
        blended[i] = Math.round(
          decoded.data[i] * (1 - blend) + enhanced[i] * blend,
        );
        blended[i + 1] = Math.round(
          decoded.data[i + 1] * (1 - blend) + enhanced[i + 1] * blend,
        );
        blended[i + 2] = Math.round(
          decoded.data[i + 2] * (1 - blend) + enhanced[i + 2] * blend,
        );
        blended[i + 3] = decoded.data[i + 3];
      }
      setDisplayUri(encodeToDataUri(blended, decoded.width, decoded.height));
    } catch (e) {
      console.warn(e);
    } finally {
      setProcessing(false);
    }
  };

  const handleSave = async () => {
    if (!displayUri) return;
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== "granted") return Alert.alert("Permission needed");

      // SAVE FIX: Support both Data URIs and local file paths
      let b64;
      if (displayUri.startsWith("data:")) {
        b64 = displayUri.split(",")[1];
      } else {
        b64 = await FileSystem.readAsStringAsync(displayUri, {
          encoding: "base64",
        });
      }

      const savePath = `${FileSystem.documentDirectory}save_${Date.now()}.jpg`;
      await FileSystem.writeAsStringAsync(savePath, b64, {
        encoding: "base64",
      });
      const asset = await MediaLibrary.createAssetAsync(savePath);
      const albumName = `ReColor_${auth.currentUser?.email.split("@")[0] || "Guest"}`;
      const album = await MediaLibrary.getAlbumAsync(albumName);

      if (!album) await MediaLibrary.createAlbumAsync(albumName, asset, false);
      else await MediaLibrary.addAssetsToAlbumAsync([asset], album, false);

      Alert.alert("Saved", "Added to ReColor album.");
    } catch (e) {
      Alert.alert("Error", "Save failed");
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: "#000" }}>
      <SafeAreaView style={{ flex: 1 }}>
        <View style={styles.camTopBar}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="arrow-back" size={24} color="#FFF" />
          </TouchableOpacity>
          <Text style={{ color: "#FFF", fontWeight: "bold" }}>
            Enhance Gallery
          </Text>
          <TouchableOpacity onPress={pickImage}>
            <Ionicons name="add-circle" size={28} color="#FFF" />
          </TouchableOpacity>
        </View>
        <View style={{ flex: 1, justifyContent: "center" }}>
          {displayUri ? (
            <Image
              source={{ uri: displayUri }}
              style={{ width, height: width * 1.3, resizeMode: "contain" }}
            />
          ) : (
            <TouchableOpacity
              onPress={pickImage}
              style={{ alignSelf: "center" }}
            >
              <Ionicons name="images-outline" size={60} color="#555" />
            </TouchableOpacity>
          )}
          {processing && (
            <ActivityIndicator
              size="large"
              color="#FFF"
              style={StyleSheet.absoluteFill}
            />
          )}
        </View>

        {originalUri && (
          <View style={{ paddingBottom: 20 }}>
            {/* Algorithm & CVD Selectors */}
            <View
              style={{
                flexDirection: "row",
                justifyContent: "center",
                gap: 10,
                marginBottom: 15,
              }}
            >
              {["daltonization", "hue_rotation"].map((a) => (
                <TouchableOpacity
                  key={a}
                  onPress={() => {
                    setAlgo(a);
                    processFilter(mode, a, intensity);
                  }}
                  style={{
                    backgroundColor: algo === a ? COLORS.secondary : "#222",
                    padding: 8,
                    borderRadius: 10,
                  }}
                >
                  <Text
                    style={{ color: "#FFF", fontSize: 10, fontWeight: "bold" }}
                  >
                    {a === "daltonization" ? "DALTO" : "HUE"}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            <View
              style={{
                flexDirection: "row",
                justifyContent: "center",
                gap: 8,
                marginBottom: 15,
              }}
            >
              {["Off", "Protan", "Deutan", "Tritan"].map((m) => (
                <TouchableOpacity
                  key={m}
                  onPress={() => {
                    setMode(m);
                    processFilter(m, algo, intensity);
                  }}
                  style={{
                    backgroundColor: mode === m ? COLORS.primary : "#333",
                    padding: 10,
                    borderRadius: 20,
                  }}
                >
                  <Text style={{ color: "#FFF", fontSize: 11 }}>{m}</Text>
                </TouchableOpacity>
              ))}
            </View>
            <Slider
              style={{ width: "80%", alignSelf: "center", height: 40 }}
              minimumValue={0}
              maximumValue={100}
              value={intensity}
              onSlidingComplete={(v) => {
                setIntensity(v);
                processFilter(mode, algo, v);
              }}
              minimumTrackTintColor={COLORS.primary}
              thumbTintColor="#FFF"
            />
            <TouchableOpacity
              onPress={handleSave}
              style={{
                alignSelf: "center",
                backgroundColor: COLORS.primary,
                padding: 12,
                borderRadius: 24,
                marginTop: 10,
              }}
            >
              <Text style={{ color: "#FFF", fontWeight: "700" }}>
                Save Enhanced Image
              </Text>
            </TouchableOpacity>
          </View>
        )}
      </SafeAreaView>
    </View>
  );
}
