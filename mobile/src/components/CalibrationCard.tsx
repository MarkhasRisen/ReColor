import React from "react";
import { StyleSheet, Text, View } from "react-native";

type Props = {
  plateId: string;
  instruction: string;
};

export function CalibrationCard({ plateId, instruction }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>{`Plate ${plateId}`}</Text>
      <Text style={styles.instruction}>{instruction}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    borderRadius: 12,
    backgroundColor: "#1f2933",
    marginVertical: 8,
  },
  title: {
    color: "#f5f7fa",
    fontWeight: "600",
    marginBottom: 8,
  },
  instruction: {
    color: "#d9e2ec",
  },
});
