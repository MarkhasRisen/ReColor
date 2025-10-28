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

const ComprehensiveTestScreen = ({ navigation }: any) => {
  const [currentPlate, setCurrentPlate] = useState(1);
  const [response, setResponse] = useState('');
  const [responses, setResponses] = useState<{ [key: number]: string }>({});

  const totalPlates = 38;

  const handleNext = () => {
    if (response.trim()) {
      setResponses({ ...responses, [currentPlate]: response });
      setResponse('');
      
      if (currentPlate < totalPlates) {
        setCurrentPlate(currentPlate + 1);
      } else {
        navigation.navigate('TestResults', { responses, mode: 'comprehensive' });
      }
    }
  };

  const handleSkip = () => {
    setResponses({ ...responses, [currentPlate]: '' });
    setResponse('');
    
    if (currentPlate < totalPlates) {
      setCurrentPlate(currentPlate + 1);
    } else {
      navigation.navigate('TestResults', { responses, mode: 'comprehensive' });
    }
  };

  const handleBack = () => {
    if (currentPlate > 1) {
      setCurrentPlate(currentPlate - 1);
      setResponse(responses[currentPlate - 1] || '');
    }
  };

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
          <View style={styles.platePlaceholder}>
            <Text style={styles.plateText}>Plate {currentPlate}</Text>
            <Text style={styles.placeholderNote}>
              (Placeholder - replace with actual Ishihara plate images)
            </Text>
          </View>
        </View>

        <View style={styles.inputSection}>
          <Text style={styles.question}>What number or pattern do you see?</Text>
          <TextInput
            style={styles.input}
            value={response}
            onChangeText={setResponse}
            placeholder="Enter your answer"
            keyboardType="default"
            maxLength={3}
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
          <Text style={styles.instructionsTitle}>Comprehensive Test</Text>
          <Text style={styles.instructionsText}>
            This test includes all 38 plates for clinical-grade diagnosis. Take your time and respond accurately.
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
    backgroundColor: '#50C878',
  },
  content: {
    padding: 20,
  },
  plateContainer: {
    alignItems: 'center',
    marginBottom: 30,
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
    backgroundColor: '#50C878',
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
    backgroundColor: '#E8F8F0',
    borderRadius: 10,
    padding: 15,
    borderWidth: 1,
    borderColor: '#B4E5CC',
  },
  instructionsTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#228B52',
    marginBottom: 8,
  },
  instructionsText: {
    fontSize: 12,
    color: '#14573A',
    lineHeight: 18,
  },
});

export default ComprehensiveTestScreen;
