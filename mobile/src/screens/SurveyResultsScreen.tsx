import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';

const SurveyResultsScreen = ({ route, navigation }: any) => {
  const { answers } = route.params || {};

  // Calculate risk score based on answers
  const calculateRiskScore = () => {
    const scores = { Never: 0, Rarely: 1, Sometimes: 2, Often: 3, Always: 3, Frequently: 3 };
    let totalScore = 0;
    let maxScore = 0;

    Object.values(answers || {}).forEach((answer: any) => {
      totalScore += scores[answer as keyof typeof scores] || 0;
      maxScore += 3;
    });

    return (totalScore / maxScore) * 100;
  };

  const riskScore = calculateRiskScore();

  const getRiskLevel = (score: number) => {
    if (score < 25) return { level: 'Low', color: '#50C878', recommendation: 'You show minimal signs of color vision deficiency.' };
    if (score < 50) return { level: 'Moderate', color: '#FFD700', recommendation: 'You may have mild color vision deficiency. Consider taking the Ishihara test.' };
    if (score < 75) return { level: 'High', color: '#FFA500', recommendation: 'You show significant signs of color vision deficiency. We recommend taking the comprehensive Ishihara test.' };
    return { level: 'Very High', color: '#FF6B6B', recommendation: 'You show strong signs of color vision deficiency. Please take the Ishihara test and consult an eye specialist.' };
  };

  const risk = getRiskLevel(riskScore);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Survey Results</Text>
          <Text style={styles.subtitle}>Based on your responses</Text>
        </View>

        <View style={styles.scoreCard}>
          <Text style={styles.scoreLabel}>Risk Assessment</Text>
          <View style={[styles.scoreCircle, { borderColor: risk.color }]}>
            <Text style={[styles.scoreValue, { color: risk.color }]}>{riskScore.toFixed(0)}%</Text>
          </View>
          <Text style={[styles.riskLevel, { color: risk.color }]}>{risk.level} Risk</Text>
        </View>

        <View style={styles.recommendationCard}>
          <Text style={styles.recommendationTitle}>Recommendation</Text>
          <Text style={styles.recommendationText}>{risk.recommendation}</Text>
        </View>

        <TouchableOpacity
          style={styles.testButton}
          onPress={() => navigation.navigate('IshiharaTest')}
        >
          <Text style={styles.testButtonText}>Take Ishihara Test</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.educationButton}
          onPress={() => navigation.navigate('Education')}
        >
          <Text style={styles.educationButtonText}>Learn More About CVD</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.homeButton}
          onPress={() => navigation.navigate('Home')}
        >
          <Text style={styles.homeButtonText}>Back to Home</Text>
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
    marginBottom: 30,
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
  scoreCard: {
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 30,
    alignItems: 'center',
    marginBottom: 20,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  scoreLabel: {
    fontSize: 16,
    color: '#666',
    marginBottom: 20,
  },
  scoreCircle: {
    width: 150,
    height: 150,
    borderRadius: 75,
    borderWidth: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: 'bold',
  },
  riskLevel: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  recommendationCard: {
    backgroundColor: '#FFF9E6',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#FFE5B4',
  },
  recommendationTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#CC8800',
    marginBottom: 10,
  },
  recommendationText: {
    fontSize: 15,
    color: '#665500',
    lineHeight: 22,
  },
  testButton: {
    backgroundColor: '#4A90E2',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginBottom: 12,
  },
  testButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  educationButton: {
    backgroundColor: '#9B59B6',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    marginBottom: 12,
  },
  educationButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  homeButton: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 18,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#4A90E2',
  },
  homeButtonText: {
    color: '#4A90E2',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default SurveyResultsScreen;
