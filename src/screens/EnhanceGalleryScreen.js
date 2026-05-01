import { Ionicons } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system";
import * as ImageManipulator from "expo-image-manipulator";
import * as ImagePicker from "expo-image-picker";
import * as MediaLibrary from "expo-media-library";
import { useState } from "react";
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
  const [processing, setProcessing] = useState(false);

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

  const handleSave = async () => {
    if (!displayUri) return;
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== "granted") return Alert.alert("Permission needed");

      const b64 = displayUri.split(",")[1];
      const savePath = `${FileSystem.documentDirectory}enhance_${Date.now()}.jpg`;
      await FileSystem.writeAsStringAsync(savePath, b64, {
        encoding: "base64",
      });

      const asset = await MediaLibrary.createAssetAsync(savePath);
      const albumName = `ReColor_${auth.currentUser?.email.split("@")[0] || "Guest"}`;
      const album = await MediaLibrary.getAlbumAsync(albumName);

      if (!album) await MediaLibrary.createAlbumAsync(albumName, asset, false);
      else await MediaLibrary.addAssetsToAlbumAsync([asset], album, false);

      Alert.alert("Saved", `Enhanced photo added to ${albumName}`);
    } catch (e) {
      Alert.alert("Error", "Save failed");
    }
  };

  const applyFilter = async (cvdMode) => {
    setMode(cvdMode);
    if (cvdMode === "Off" || !originalUri) return setDisplayUri(originalUri);
    setProcessing(true);
    try {
      const resized = await ImageManipulator.manipulateAsync(
        originalUri,
        [{ resize: { width: 1040 } }],
        { base64: true },
      );
      const decoded = decodeJpegBase64(resized.base64);
      const enhanced = applyDaltonization(
        { data: decoded.data, width: decoded.width, height: decoded.height },
        null,
        cvdMode,
      );
      setDisplayUri(encodeToDataUri(enhanced, decoded.width, decoded.height));
    } catch (e) {
      console.warn(e);
    } finally {
      setProcessing(false);
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
            Enhance Analysis
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
          <View style={{ paddingBottom: 30 }}>
            <View
              style={{
                flexDirection: "row",
                justifyContent: "center",
                gap: 10,
                marginBottom: 14,
              }}
            >
              {["Off", "Protan", "Deutan", "Tritan"].map((m) => (
                <TouchableOpacity
                  key={m}
                  onPress={() => applyFilter(m)}
                  style={{
                    backgroundColor: mode === m ? COLORS.primary : "#333",
                    padding: 10,
                    borderRadius: 20,
                  }}
                >
                  <Text style={{ color: "#FFF" }}>{m}</Text>
                </TouchableOpacity>
              ))}
            </View>
            <TouchableOpacity
              onPress={handleSave}
              style={{
                alignSelf: "center",
                backgroundColor: COLORS.primary,
                paddingHorizontal: 28,
                paddingVertical: 12,
                borderRadius: 24,
              }}
            >
              <Text style={{ color: "#FFF", fontWeight: "700" }}>
                Save to Account Album
              </Text>
            </TouchableOpacity>
          </View>
        )}
      </SafeAreaView>
    </View>
  );
}
