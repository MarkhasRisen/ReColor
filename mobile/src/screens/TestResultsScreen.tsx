import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';

const TestResultsScreen = ({ route, navigation }: any) => {
  const { responses, mode } = route.params || {};

  // Placeholder results - will be replaced with actual API call
  const results = {
    cvdType: 'Deutan',
    severity: 0.6,
    confidence: 0.85,
    interpretation: 'Moderate Deuteranomaly (green deficiency) detected',
    totalPlates: mode === 'quick' ? 14 : 38,
    correctNormal: 8,
    correctProtan: 2,
    correctDeutan: 1,
  };

  const getSeverityColor = (severity: number) => {
    if (severity === 0) return '#50C878';
    if (severity < 0.4) return '#FFD700';
    if (severity < 0.7) return '#FFA500';
    return '#FF6B6B';
  };

  const getSeverityText = (severity: number) => {
    if (severity === 0) return 'Normal';
    if (severity < 0.4) return 'Mild';
    if (severity < 0.7) return 'Moderate';
    return 'Strong';
  };

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
          <Text style={styles.cvdType}>{results.cvdType}</Text>
          <Text style={styles.interpretation}>{results.interpretation}</Text>
          
          <View style={styles.severityContainer}>
            <Text style={styles.label}>Severity</Text>
            <View style={styles.severityBar}>
              <View
                style={[
                  styles.severityFill,
                  { width: `${results.severity * 100}%`, backgroundColor: getSeverityColor(results.severity) },
                ]}
              />
            </View>
            <Text style={[styles.severityText, { color: getSeverityColor(results.severity) }]}>
              {getSeverityText(results.severity)} ({(results.severity * 100).toFixed(0)}%)
            </Text>
          </View>

          <View style={styles.confidenceContainer}>
            <Text style={styles.label}>Confidence</Text>
            <Text style={styles.confidenceValue}>{(results.confidence * 100).toFixed(0)}%</Text>
          </View>
        </View>

        <View style={styles.statsCard}>
          <Text style={styles.statsTitle}>Test Statistics</Text>
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Total Plates:</Text>
            <Text style={styles.statValue}>{results.totalPlates}</Text>
          </View>
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Normal Vision Correct:</Text>
            <Text style={styles.statValue}>{results.correctNormal}</Text>
          </View>
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Protan Indicators:</Text>
            <Text style={styles.statValue}>{results.correctProtan}</Text>
          </View>
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Deutan Indicators:</Text>
            <Text style={styles.statValue}>{results.correctDeutan}</Text>
          </View>
        </View>

        <View style={styles.noteCard}>
          <Text style={styles.noteTitle}>⚠️ Important Note</Text>
          <Text style={styles.noteText}>
            This is a screening tool. Clinical diagnosis should always be confirmed by a qualified optometrist or ophthalmologist.
          </Text>
        </View>

        <TouchableOpacity
          style={styles.saveButton}
          onPress={() => {
            // TODO: Save profile to Firebase
            navigation.navigate('Home');
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
