import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  TextInput,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { getIshiharaPlates } from '../services/api';

interface Plate {
  plate_number: number;
  image_url: string;
  is_control: boolean;
}

const ComprehensiveTestScreen = ({ navigation }: any) => {
  const [currentPlate, setCurrentPlate] = useState(1);
  const [response, setResponse] = useState('');
  const [responses, setResponses] = useState<{ [key: number]: string }>({});
  const [plates, setPlates] = useState<Plate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const totalPlates = plates.length || 38;

  // Fetch plates from backend on mount
  useEffect(() => {
    loadPlates();
  }, []);

  const loadPlates = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getIshiharaPlates('comprehensive');
      setPlates(data.plates || []);
    } catch (err: any) {
      console.error('Failed to load Ishihara plates:', err);
      setError(err.message || 'Failed to load test plates');
      Alert.alert(
        'Connection Error',
        'Could not load test plates. Please check your backend connection.',
        [
          { text: 'Retry', onPress: loadPlates },
          { text: 'Cancel', onPress: () => navigation.goBack() }
        ]
      );
    } finally {
      setLoading(false);
    }
  };

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

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#50C878" />
          <Text style={styles.loadingText}>Loading comprehensive test (38 plates)...</Text>
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
          {currentPlateData?.image_url ? (
            <Image
              source={{ uri: currentPlateData.image_url }}
              style={styles.plateImage}
              resizeMode="contain"
            />
          ) : (
            <View style={styles.platePlaceholder}>
              <Text style={styles.plateText}>Plate {currentPlate}</Text>
              <Text style={styles.placeholderNote}>
                (Image not available)
              </Text>
            </View>
          )}
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
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
    backgroundColor: '#50C878',
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
    backgroundColor: '#50C878',
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
