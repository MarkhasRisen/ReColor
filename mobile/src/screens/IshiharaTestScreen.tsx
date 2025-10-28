import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, SafeAreaView } from 'react-native';

const IshiharaTestScreen = ({ navigation }: any) => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Ishihara Color Vision Test</Text>
        <Text style={styles.description}>
          Choose your preferred test mode to assess your color vision deficiency.
        </Text>

        <TouchableOpacity
          style={[styles.optionCard, { borderLeftColor: '#4A90E2' }]}
          onPress={() => navigation.navigate('QuickTest')}
        >
          <View style={styles.optionHeader}>
            <Text style={styles.optionTitle}>Quick Test</Text>
            <Text style={styles.badge}>14 Plates</Text>
          </View>
          <Text style={styles.optionDescription}>
            Standard screening test (2-3 minutes)
          </Text>
          <Text style={styles.optionDetails}>
            ✓ High sensitivity for red-green CVD{'\n'}
            ✓ Suitable for routine screening
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.optionCard, { borderLeftColor: '#50C878' }]}
          onPress={() => navigation.navigate('ComprehensiveTest')}
        >
          <View style={styles.optionHeader}>
            <Text style={styles.optionTitle}>Comprehensive Test</Text>
            <Text style={styles.badge}>38 Plates</Text>
          </View>
          <Text style={styles.optionDescription}>
            Complete diagnostic assessment (5-7 minutes)
          </Text>
          <Text style={styles.optionDetails}>
            ✓ Clinical-grade diagnosis{'\n'}
            ✓ Includes protan/deutan classification{'\n'}
            ✓ Tracing plates included
          </Text>
        </TouchableOpacity>

        <View style={styles.infoBox}>
          <Text style={styles.infoTitle}>Testing Guidelines</Text>
          <Text style={styles.infoText}>
            • Ensure proper lighting (natural daylight preferred){'\n'}
            • Hold device at comfortable reading distance (~75cm){'\n'}
            • Respond within 3-5 seconds per plate{'\n'}
            • Don't guess - say what you see immediately
          </Text>
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  description: {
    fontSize: 16,
    color: '#666',
    marginBottom: 30,
  },
  optionCard: {
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    borderLeftWidth: 5,
  },
  optionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  optionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  badge: {
    backgroundColor: '#E8F4F8',
    color: '#4A90E2',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    fontSize: 12,
    fontWeight: 'bold',
  },
  optionDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
  },
  optionDetails: {
    fontSize: 12,
    color: '#888',
    lineHeight: 18,
  },
  infoBox: {
    backgroundColor: '#FFF9E6',
    borderRadius: 10,
    padding: 15,
    marginTop: 20,
    borderWidth: 1,
    borderColor: '#FFE5B4',
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#CC8800',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 13,
    color: '#665500',
    lineHeight: 20,
  },
});

export default IshiharaTestScreen;
