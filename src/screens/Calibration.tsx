import React, { useEffect, useState } from "react";
import { 
  ActivityIndicator, 
  Button, 
  Image, 
  SafeAreaView, 
  ScrollView, 
  StyleSheet, 
  Text, 
  TextInput, 
  View 
} from "react-native";
import { getIshiharaPlates, evaluateIshiharaTest, type IshiharaPlate } from "../services/api";

export function CalibrationScreen() {
  const [loading, setLoading] = useState(true);
  const [plates, setPlates] = useState<IshiharaPlate[]>([]);
  const [responses, setResponses] = useState<Record<number, string>>({});
  const [currentPlateIndex, setCurrentPlateIndex] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    loadPlates();
  }, []);

  const loadPlates = async () => {
    try {
      setLoading(true);
      const data = await getIshiharaPlates("quick"); // Use quick test (14 plates)
      setPlates(data.plates);
    } catch (error) {
      console.error("Failed to load Ishihara plates:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitAnswer = () => {
    if (!inputValue.trim()) return;

    const currentPlate = plates[currentPlateIndex];
    setResponses((prev) => ({
      ...prev,
      [currentPlate.plate_number]: inputValue.trim(),
    }));
    setInputValue("");

    // Move to next plate or finish
    if (currentPlateIndex < plates.length - 1) {
      setCurrentPlateIndex(currentPlateIndex + 1);
    }
  };

  const handleSkip = () => {
    const currentPlate = plates[currentPlateIndex];
    setResponses((prev) => ({
      ...prev,
      [currentPlate.plate_number]: "",
    }));

    if (currentPlateIndex < plates.length - 1) {
      setCurrentPlateIndex(currentPlateIndex + 1);
    }
  };

  const handleEvaluate = async () => {
    try {
      setLoading(true);
      const data = await evaluateIshiharaTest(
        "demo-user", // TODO: Use actual user ID from authentication
        responses,
        "quick",
        true // Save profile to Firebase
      );
      setResult(data);
    } catch (error) {
      console.error("Evaluation failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleRestart = () => {
    setResponses({});
    setCurrentPlateIndex(0);
    setResult(null);
    setInputValue("");
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.loadingText}>Loading Ishihara test...</Text>
      </View>
    );
  }

  if (result) {
    return (
      <SafeAreaView style={styles.container}>
        <ScrollView style={styles.scroll}>
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>Test Results</Text>
            
            <View style={styles.resultSection}>
              <Text style={styles.resultLabel}>Color Vision Type:</Text>
              <Text style={styles.resultValue}>{result.diagnosis.cvd_type.toUpperCase()}</Text>
            </View>

            <View style={styles.resultSection}>
              <Text style={styles.resultLabel}>Severity:</Text>
              <Text style={styles.resultValue}>
                {(result.diagnosis.severity * 100).toFixed(0)}%
              </Text>
            </View>

            <View style={styles.resultSection}>
              <Text style={styles.resultLabel}>Confidence:</Text>
              <Text style={styles.resultValue}>
                {(result.diagnosis.confidence * 100).toFixed(0)}%
              </Text>
            </View>

            <View style={styles.resultSection}>
              <Text style={styles.resultLabel}>Interpretation:</Text>
              <Text style={styles.interpretationText}>
                {result.diagnosis.interpretation}
              </Text>
            </View>

            <View style={styles.resultSection}>
              <Text style={styles.resultLabel}>Score Details:</Text>
              <Text style={styles.scoreText}>
                Correct (Normal): {result.result.correct_normal}/{result.result.total_plates}
              </Text>
              <Text style={styles.scoreText}>
                Correct (Protan): {result.result.correct_protan}
              </Text>
              <Text style={styles.scoreText}>
                Correct (Deutan): {result.result.correct_deutan}
              </Text>
            </View>

            {result.profile_saved && (
              <Text style={styles.savedText}>✓ Profile saved to your account</Text>
            )}

            <Button title="Start New Test" onPress={handleRestart} />
          </View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  const currentPlate = plates[currentPlateIndex];
  const isLastPlate = currentPlateIndex === plates.length - 1;
  const hasAnsweredAll = Object.keys(responses).length === plates.length;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scroll}>
        <View style={styles.header}>
          <Text style={styles.title}>Ishihara Color Vision Test</Text>
          <Text style={styles.progress}>
            Plate {currentPlateIndex + 1} of {plates.length}
          </Text>
        </View>

        {currentPlate && (
          <View style={styles.plateContainer}>
            <Image
              source={{ uri: `http://10.0.2.2:8000${currentPlate.image_url}` }}
              style={styles.plateImage}
              resizeMode="contain"
            />
            
            {currentPlate.is_control && (
              <Text style={styles.controlLabel}>⚠️ Control Plate</Text>
            )}

            <View style={styles.inputContainer}>
              <Text style={styles.instruction}>
                What number or pattern do you see?
              </Text>
              <TextInput
                style={styles.input}
                value={inputValue}
                onChangeText={setInputValue}
                placeholder="Enter your answer"
                placeholderTextColor="#666"
                keyboardType="default"
                autoCapitalize="none"
              />
            </View>

            <View style={styles.buttonRow}>
              <View style={styles.buttonWrapper}>
                <Button
                  title="Skip"
                  onPress={handleSkip}
                  color="#888"
                />
              </View>
              <View style={styles.buttonWrapper}>
                <Button
                  title={isLastPlate ? "Submit Answer" : "Next"}
                  onPress={handleSubmitAnswer}
                  disabled={!inputValue.trim()}
                />
              </View>
            </View>
          </View>
        )}

        {hasAnsweredAll && (
          <View style={styles.evaluateContainer}>
            <Text style={styles.completeText}>
              All plates answered! Ready to evaluate your results.
            </Text>
            <Button
              title="Get Results"
              onPress={handleEvaluate}
              color="#4CAF50"
            />
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
  centerContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#0d1117",
  },
  scroll: {
    flex: 1,
  },
  header: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#30363d",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#f5f7fa",
    marginBottom: 8,
  },
  progress: {
    fontSize: 16,
    color: "#8b949e",
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: "#8b949e",
  },
  plateContainer: {
    padding: 16,
  },
  plateImage: {
    width: "100%",
    height: 300,
    marginBottom: 16,
    borderRadius: 8,
  },
  controlLabel: {
    textAlign: "center",
    color: "#f85149",
    fontSize: 14,
    marginBottom: 12,
    fontWeight: "600",
  },
  inputContainer: {
    marginVertical: 16,
  },
  instruction: {
    fontSize: 16,
    color: "#f5f7fa",
    marginBottom: 12,
    textAlign: "center",
  },
  input: {
    backgroundColor: "#161b22",
    borderWidth: 1,
    borderColor: "#30363d",
    borderRadius: 6,
    padding: 12,
    fontSize: 18,
    color: "#f5f7fa",
    textAlign: "center",
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 16,
  },
  buttonWrapper: {
    flex: 1,
    marginHorizontal: 8,
  },
  evaluateContainer: {
    padding: 16,
    marginTop: 24,
  },
  completeText: {
    fontSize: 16,
    color: "#4CAF50",
    textAlign: "center",
    marginBottom: 16,
    fontWeight: "600",
  },
  resultContainer: {
    padding: 16,
  },
  resultTitle: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#f5f7fa",
    marginBottom: 24,
    textAlign: "center",
  },
  resultSection: {
    marginBottom: 20,
    padding: 16,
    backgroundColor: "#161b22",
    borderRadius: 8,
  },
  resultLabel: {
    fontSize: 14,
    color: "#8b949e",
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#4CAF50",
  },
  interpretationText: {
    fontSize: 16,
    color: "#f5f7fa",
    lineHeight: 24,
    marginTop: 8,
  },
  scoreText: {
    fontSize: 14,
    color: "#f5f7fa",
    marginTop: 4,
  },
  savedText: {
    fontSize: 14,
    color: "#4CAF50",
    textAlign: "center",
    marginVertical: 16,
  },
});
