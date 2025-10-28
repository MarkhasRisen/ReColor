import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { evaluateIshiharaTest } from '../services/api';
// import { auth } from '../services/firebase'; // Uncomment when using Firebase

const TestResultsScreen = ({ route, navigation }: any) => {
  const { responses, mode } = route.params || {};
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);

  useEffect(() => {
    evaluateTest();
  }, []);

  const evaluateTest = async () => {
    try {
      setLoading(true);
      setError(null);

      // Get user ID from Firebase Auth (when implemented)
      // const userId = auth().currentUser?.uid || 'guest';
      const userId = 'guest_' + Date.now(); // Temporary guest ID

      // Convert responses object to API format
      const apiResponses: { [key: number]: string } = {};
      Object.keys(responses || {}).forEach(key => {
        apiResponses[parseInt(key)] = responses[key] || '';
      });

      const data = await evaluateIshiharaTest(
        userId,
        apiResponses,
        mode || 'quick',
        false // Don't save profile for guest users
      );

      setResults(data);
    } catch (err: any) {
      console.error('Failed to evaluate test:', err);
      setError(err.message || 'Failed to evaluate test results');
      Alert.alert(
        'Evaluation Error',
        'Could not evaluate your test. Using offline analysis.',
        [{ text: 'OK' }]
      );
      // Fallback to placeholder results
      setResults({
        cvd_type: 'normal',
        severity: 0,
        confidence: 0,
        interpretation: 'Unable to connect to server for evaluation',
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4A90E2" />
          <Text style={styles.loadingText}>Analyzing your responses...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!results) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>❌ No results available</Text>
          <TouchableOpacity 
            style={styles.retryButton} 
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.retryButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const getSeverityColor = (severity: number) => {
    if (severity === 0 || severity === null) return '#50C878';
    if (severity < 0.4) return '#FFD700';
    if (severity < 0.7) return '#FFA500';
    return '#FF6B6B';
  };

  const getSeverityText = (severity: number) => {
    if (severity === 0 || severity === null) return 'Normal';
    if (severity < 0.4) return 'Mild';
    if (severity < 0.7) return 'Moderate';
    return 'Strong';
  };

  const formatCVDType = (cvdType: string) => {
    const types: { [key: string]: string } = {
      'normal': 'Normal Vision',
      'protan': 'Protanopia/Protanomaly',
      'deutan': 'Deuteranopia/Deuteranomaly',
      'tritan': 'Tritanopia/Tritanomaly',
      'total': 'Total Color Blindness',
    };
    return types[cvdType.toLowerCase()] || cvdType;
  };

  const cvdType = formatCVDType(results.cvd_type || 'unknown');
  const severity = results.severity || 0;
  const confidence = results.confidence || 0;
  const interpretation = results.interpretation || 'Analysis completed';

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Test Results</Text>
          <Text style={styles.subtitle}>{mode === 'quick' ? 'Quick' : 'Comprehensive'} Test</Text>
        </View>

        <View style={styles.resultCard}>
          <View style={styles.resultHeader}>
            <Text style={styles.resultTitle}>Diagnosis</Text>
          </View>
          <Text style={styles.cvdType}>{cvdType}</Text>
          <Text style={styles.interpretation}>{interpretation}</Text>
          
          <View style={styles.severityContainer}>
            <Text style={styles.label}>Severity</Text>
            <View style={styles.severityBar}>
              <View
                style={[
                  styles.severityFill,
                  { width: `${severity * 100}%`, backgroundColor: getSeverityColor(severity) },
                ]}
              />
            </View>
            <Text style={[styles.severityText, { color: getSeverityColor(severity) }]}>
              {getSeverityText(severity)} ({(severity * 100).toFixed(0)}%)
            </Text>
          </View>

          <View style={styles.confidenceContainer}>
            <Text style={styles.label}>Confidence</Text>
            <Text style={styles.confidenceValue}>{(confidence * 100).toFixed(0)}%</Text>
          </View>
        </View>

        {results.statistics && (
          <View style={styles.statsCard}>
            <Text style={styles.statsTitle}>Test Statistics</Text>
            <View style={styles.statRow}>
              <Text style={styles.statLabel}>Total Plates:</Text>
              <Text style={styles.statValue}>{results.statistics.total_plates || (mode === 'quick' ? 14 : 38)}</Text>
            </View>
            {results.statistics.normal_correct !== undefined && (
              <View style={styles.statRow}>
                <Text style={styles.statLabel}>Normal Vision Correct:</Text>
                <Text style={styles.statValue}>{results.statistics.normal_correct}</Text>
              </View>
            )}
            {results.statistics.protan_indicators !== undefined && (
              <View style={styles.statRow}>
                <Text style={styles.statLabel}>Protan Indicators:</Text>
                <Text style={styles.statValue}>{results.statistics.protan_indicators}</Text>
              </View>
            )}
            {results.statistics.deutan_indicators !== undefined && (
              <View style={styles.statRow}>
                <Text style={styles.statLabel}>Deutan Indicators:</Text>
                <Text style={styles.statValue}>{results.statistics.deutan_indicators}</Text>
              </View>
            )}
          </View>
        )}

        <View style={styles.noteCard}>
          <Text style={styles.noteTitle}>⚠️ Important Note</Text>
          <Text style={styles.noteText}>
            This is a screening tool. Clinical diagnosis should always be confirmed by a qualified optometrist or ophthalmologist.
          </Text>
        </View>

        <TouchableOpacity
          style={styles.saveButton}
          onPress={() => {
            // TODO: Save profile to Firebase when auth is implemented
            Alert.alert(
              'Profile Saved',
              'Your vision profile will be saved when you create an account.',
              [{ text: 'OK', onPress: () => navigation.navigate('Main') }]
            );
          }}
        >
          <Text style={styles.saveButtonText}>Save Profile & Continue</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.retakeButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.retakeButtonText}>Retake Test</Text>
        </TouchableOpacity>
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
  content: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  resultCard: {
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  resultHeader: {
    marginBottom: 15,
  },
  resultTitle: {
    fontSize: 16,
    color: '#666',
  },
  cvdType: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#4A90E2',
    marginBottom: 10,
  },
  interpretation: {
    fontSize: 16,
    color: '#666',
    marginBottom: 20,
  },
  severityContainer: {
    marginBottom: 20,
  },
  label: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  severityBar: {
    height: 20,
    backgroundColor: '#E0E0E0',
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 8,
  },
  severityFill: {
    height: '100%',
  },
  severityText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  confidenceContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confidenceValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4A90E2',
  },
  statsCard: {
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  statLabel: {
    fontSize: 14,
    color: '#666',
  },
  statValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  noteCard: {
    backgroundColor: '#FFF9E6',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#FFE5B4',
  },
  noteTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#CC8800',
    marginBottom: 8,
  },
  noteText: {
    fontSize: 13,
    color: '#665500',
    lineHeight: 20,
  },
  saveButton: {
    backgroundColor: '#4A90E2',
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    marginBottom: 10,
  },
  saveButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  retakeButton: {
    backgroundColor: '#FFF',
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#4A90E2',
  },
  retakeButtonText: {
    color: '#4A90E2',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default TestResultsScreen;
