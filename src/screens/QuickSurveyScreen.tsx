import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';

interface Question {
  id: number;
  question: string;
  options: string[];
}

const questions: Question[] = [
  {
    id: 1,
    question: 'Do you have difficulty distinguishing between red and green colors?',
    options: ['Never', 'Sometimes', 'Often', 'Always'],
  },
  {
    id: 2,
    question: 'Do traffic lights appear confusing to you?',
    options: ['Never', 'Sometimes', 'Often', 'Always'],
  },
  {
    id: 3,
    question: 'Do you have trouble matching clothing colors?',
    options: ['Never', 'Sometimes', 'Often', 'Always'],
  },
  {
    id: 4,
    question: 'Have you ever been told you name colors incorrectly?',
    options: ['Never', 'Rarely', 'Sometimes', 'Frequently'],
  },
  {
    id: 5,
    question: 'Do you struggle to see colored text on colored backgrounds?',
    options: ['Never', 'Sometimes', 'Often', 'Always'],
  },
];

const QuickSurveyScreen = ({ navigation }: any) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<{ [key: number]: string }>({});

  const handleAnswer = (answer: string) => {
    const newAnswers = { ...answers, [questions[currentQuestion].id]: answer };
    setAnswers(newAnswers);

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      // Submit survey
      navigation.navigate('SurveyResults', { answers: newAnswers });
    }
  };

  const handleBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const progress = ((currentQuestion + 1) / questions.length) * 100;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>
          Question {currentQuestion + 1} of {questions.length}
        </Text>
        <View style={styles.progressBar}>
          <View style={[styles.progress, { width: `${progress}%` }]} />
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.questionCard}>
          <Text style={styles.questionNumber}>Q{questions[currentQuestion].id}</Text>
          <Text style={styles.questionText}>{questions[currentQuestion].question}</Text>
        </View>

        <View style={styles.optionsContainer}>
          {questions[currentQuestion].options.map((option, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.optionButton,
                answers[questions[currentQuestion].id] === option && styles.selectedOption,
              ]}
              onPress={() => handleAnswer(option)}
            >
              <Text
                style={[
                  styles.optionText,
                  answers[questions[currentQuestion].id] === option && styles.selectedOptionText,
                ]}
              >
                {option}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {currentQuestion > 0 && (
          <TouchableOpacity style={styles.backButton} onPress={handleBack}>
            <Text style={styles.backButtonText}>← Previous Question</Text>
          </TouchableOpacity>
        )}
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
  headerText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 10,
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
  questionCard: {
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 25,
    marginBottom: 30,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  questionNumber: {
    fontSize: 14,
    color: '#50C878',
    fontWeight: 'bold',
    marginBottom: 10,
  },
  questionText: {
    fontSize: 20,
    color: '#333',
    lineHeight: 28,
  },
  optionsContainer: {
    marginBottom: 20,
  },
  optionButton: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 18,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: '#E0E0E0',
  },
  selectedOption: {
    borderColor: '#50C878',
    backgroundColor: '#E8F8F0',
  },
  optionText: {
    fontSize: 16,
    color: '#333',
    textAlign: 'center',
  },
  selectedOptionText: {
    color: '#50C878',
    fontWeight: 'bold',
  },
  backButton: {
    padding: 15,
    alignItems: 'center',
  },
  backButtonText: {
    color: '#666',
    fontSize: 16,
  },
});

export default QuickSurveyScreen;
