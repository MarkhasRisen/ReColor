import React, { useCallback, useState } from "react";
import { Button, Image, SafeAreaView, StyleSheet, View } from "react-native";
import * as ImagePicker from "expo-image-picker";

import { submitImage } from "../services/api";

export function LivePreviewScreen() {
  const [resultImage, setResultImage] = useState<string | null>(null);

  const handlePickImage = useCallback(async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      return;
    }

    const selection = await ImagePicker.launchImageLibraryAsync({ base64: true });
    if (selection.canceled || !selection.assets?.[0]?.base64) {
      return;
    }

    const userId = "demo-user";
    const response = await submitImage(userId, selection.assets[0].base64);
    setResultImage(`data:${response.content_type};base64,${response.data}`);
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.preview}>
        {resultImage && <Image style={styles.image} source={{ uri: resultImage }} />}
      </View>
      <Button title="Select Image" onPress={handlePickImage} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    padding: 16,
  },
  preview: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
    marginBottom: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  image: {
    width: "100%",
    height: "100%",
    resizeMode: "contain",
  },
});
