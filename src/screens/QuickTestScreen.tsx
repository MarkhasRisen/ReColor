import React, { useState } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  TextInput,
  ScrollView,
} from 'react-native';
import { QUICK_PLATES, evaluateIshiharaTest } from '../services/ishihara/ishihara';

const QuickTestScreen = ({ navigation }: any) => {
  const [currentPlate, setCurrentPlate] = useState(1);
  const [response, setResponse] = useState('');
  const [responses, setResponses] = useState<{ [key: number]: string }>({});

  const totalPlates = QUICK_PLATES.length;

  const handleNext = () => {
    if (response.trim()) {
      setResponses({ ...responses, [currentPlate]: response });
      setResponse('');

      if (currentPlate < totalPlates) {
        setCurrentPlate(currentPlate + 1);
      } else {
        // Evaluate locally and navigate to results
        const responsesMap = new Map<number, string>(Object.entries(responses).map(([k, v]) => [parseInt(k), v]));
        responsesMap.set(currentPlate, response);
        const result = evaluateIshiharaTest(responsesMap, false);
        navigation.navigate('TestResults', { result, responses, mode: 'quick' });
      }
    }
  };  const handleSkip = () => {
    setResponses({ ...responses, [currentPlate]: '' });
    setResponse('');

    if (currentPlate < totalPlates) {
      setCurrentPlate(currentPlate + 1);
    } else {
      // Evaluate locally and navigate to results
      const responsesMap = new Map<number, string>(Object.entries(responses).map(([k, v]) => [parseInt(k), v]));
      responsesMap.set(currentPlate, '');
      const result = evaluateIshiharaTest(responsesMap, false);
      navigation.navigate('TestResults', { result, responses, mode: 'quick' });
    }
  };  const handleBack = () => {
    if (currentPlate > 1) {
      setCurrentPlate(currentPlate - 1);
      setResponse(responses[currentPlate - 1] || '');
    }
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4A90E2" />
          <Text style={styles.loadingText}>Loading test plates...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error || plates.length === 0) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>❌ {error || 'No plates available'}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={loadPlates}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const currentPlateData = plates[currentPlate - 1];

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.plateNumber}>
          Plate {currentPlate} of {totalPlates}
        </Text>
        <View style={styles.progressBar}>
          <View style={[styles.progress, { width: `${(currentPlate / totalPlates) * 100}%` }]} />
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.plateContainer}>
          <Image
            source={currentPlateData.imageSource}
            style={styles.plateImage}
            resizeMode="contain"
          />
        </View>

        <View style={styles.inputSection}>
          <Text style={styles.question}>What number or pattern do you see?</Text>
          <TextInput
            style={styles.input}
            value={response}
            onChangeText={setResponse}
            placeholder="Enter your answer (or leave blank if nothing visible)"
            keyboardType="number-pad"
            maxLength={2}
          />

          <View style={styles.buttonRow}>
            {currentPlate > 1 && (
              <TouchableOpacity style={styles.backButton} onPress={handleBack}>
                <Text style={styles.backButtonText}>← Back</Text>
              </TouchableOpacity>
            )}
            
            <TouchableOpacity style={styles.skipButton} onPress={handleSkip}>
              <Text style={styles.skipButtonText}>Skip</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.nextButton} onPress={handleNext}>
              <Text style={styles.nextButtonText}>
                {currentPlate === totalPlates ? 'Finish' : 'Next →'}
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.instructions}>
          <Text style={styles.instructionsTitle}>Instructions:</Text>
          <Text style={styles.instructionsText}>
            • View the plate for 3-5 seconds{'\n'}
            • State what you see immediately{'\n'}
            • If you don't see anything, leave it blank or skip{'\n'}
            • Don't overthink or guess
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#666',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    fontSize: 16,
    color: '#FF6B6B',
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: '#4A90E2',
    borderRadius: 10,
    padding: 15,
    paddingHorizontal: 30,
  },
  retryButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  header: {
    backgroundColor: '#FFF',
    padding: 15,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  plateNumber: {
    fontSize: 16,
    color: '#666',
    marginBottom: 10,
    textAlign: 'center',
  },
  progressBar: {
    height: 6,
    backgroundColor: '#E0E0E0',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progress: {
    height: '100%',
    backgroundColor: '#4A90E2',
  },
  content: {
    padding: 20,
  },
  plateContainer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  plateImage: {
    width: 300,
    height: 300,
    borderRadius: 150,
  },
  platePlaceholder: {
    width: 300,
    height: 300,
    backgroundColor: '#DDD',
    borderRadius: 150,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  plateText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#666',
  },
  placeholderNote: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
    marginTop: 10,
  },
  inputSection: {
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
  },
  question: {
    fontSize: 16,
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#DDD',
    borderRadius: 10,
    padding: 15,
    fontSize: 24,
    textAlign: 'center',
    marginBottom: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  backButton: {
    backgroundColor: '#E0E0E0',
    borderRadius: 10,
    padding: 15,
    flex: 1,
    marginRight: 10,
  },
  backButtonText: {
    textAlign: 'center',
    color: '#666',
    fontWeight: 'bold',
  },
  skipButton: {
    backgroundColor: '#FFE5B4',
    borderRadius: 10,
    padding: 15,
    flex: 1,
    marginRight: 10,
  },
  skipButtonText: {
    textAlign: 'center',
    color: '#CC8800',
    fontWeight: 'bold',
  },
  nextButton: {
    backgroundColor: '#4A90E2',
    borderRadius: 10,
    padding: 15,
    flex: 1,
  },
  nextButtonText: {
    textAlign: 'center',
    color: '#FFF',
    fontWeight: 'bold',
  },
  instructions: {
    backgroundColor: '#FFF9E6',
    borderRadius: 10,
    padding: 15,
    borderWidth: 1,
    borderColor: '#FFE5B4',
  },
  instructionsTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#CC8800',
    marginBottom: 8,
  },
  instructionsText: {
    fontSize: 12,
    color: '#665500',
    lineHeight: 18,
  },
});

export default QuickTestScreen;
