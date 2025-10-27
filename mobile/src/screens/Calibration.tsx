import React, { useState } from "react";
import { Button, SafeAreaView, ScrollView, StyleSheet, Text, View } from "react-native";
import { CalibrationCard } from "../components/CalibrationCard";
import { submitCalibration } from "../services/api";

const ISHIHARA_PLATES = [
  { id: "p1", instruction: "What number do you see? (Protan test plate)" },
  { id: "p2", instruction: "What number do you see? (Protan test plate)" },
  { id: "p3", instruction: "What number do you see? (Protan test plate)" },
  { id: "d1", instruction: "What number do you see? (Deutan test plate)" },
  { id: "d2", instruction: "What number do you see? (Deutan test plate)" },
  { id: "d3", instruction: "What number do you see? (Deutan test plate)" },
  { id: "t1", instruction: "What number do you see? (Tritan test plate)" },
  { id: "t2", instruction: "What number do you see? (Tritan test plate)" },
];

export function CalibrationScreen() {
  const [responses, setResponses] = useState<Record<string, "correct" | "incorrect" | "skipped">>({});
  const [result, setResult] = useState<any>(null);

  const handleResponse = (plateId: string, response: "correct" | "incorrect" | "skipped") => {
    setResponses((prev) => ({ ...prev, [plateId]: response }));
  };

  const handleSubmit = async () => {
    try {
      const data = await submitCalibration({ userId: "demo-user", responses });
      setResult(data);
    } catch (error) {
      console.error("Calibration failed:", error);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scroll}>
        {ISHIHARA_PLATES.map((plate) => (
          <View key={plate.id}>
            <CalibrationCard plateId={plate.id} instruction={plate.instruction} />
            <View style={styles.buttons}>
              <Button title="Correct" onPress={() => handleResponse(plate.id, "correct")} />
              <Button title="Incorrect" onPress={() => handleResponse(plate.id, "incorrect")} />
              <Button title="Skip" onPress={() => handleResponse(plate.id, "skipped")} />
            </View>
          </View>
        ))}
        <Button title="Submit Calibration" onPress={handleSubmit} />
        {result && (
          <View style={styles.result}>
            <Text style={styles.resultText}>Deficiency: {result.deficiency}</Text>
            <Text style={styles.resultText}>Severity: {result.severity}</Text>
            <Text style={styles.resultText}>Confidence: {result.confidence}</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0d1117",
  },
  scroll: {
    padding: 16,
  },
  buttons: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginVertical: 12,
  },
  result: {
    marginTop: 24,
    padding: 16,
    backgroundColor: "#1f2933",
    borderRadius: 8,
  },
  resultText: {
    color: "#f5f7fa",
    fontSize: 16,
    marginVertical: 4,
  },
});
