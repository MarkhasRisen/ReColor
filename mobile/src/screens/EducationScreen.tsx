import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';

interface TopicCardProps {
  title: string;
  description: string;
  icon: string;
  onPress: () => void;
  color: string;
}

const TopicCard = ({ title, description, icon, onPress, color }: TopicCardProps) => (
  <TouchableOpacity style={[styles.card, { borderLeftColor: color }]} onPress={onPress}>
    <Text style={styles.icon}>{icon}</Text>
    <View style={styles.cardContent}>
      <Text style={styles.cardTitle}>{title}</Text>
      <Text style={styles.cardDescription}>{description}</Text>
    </View>
  </TouchableOpacity>
);

const EducationScreen = ({ navigation }: any) => {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Awareness & Education</Text>
          <Text style={styles.subtitle}>Learn about color vision deficiencies</Text>
        </View>

        <TopicCard
          title="What is Color Blindness?"
          description="Understanding color vision deficiency basics"
          icon="🎨"
          color="#4A90E2"
          onPress={() => navigation.navigate('LearnDetails', { topic: 'basics' })}
        />

        <TopicCard
          title="Types of CVD"
          description="Protan, Deutan, Tritan deficiencies explained"
          icon="🔬"
          color="#50C878"
          onPress={() => navigation.navigate('LearnDetails', { topic: 'types' })}
        />

        <TopicCard
          title="How Common is CVD?"
          description="Statistics and demographics worldwide"
          icon="📊"
          color="#FFD700"
          onPress={() => navigation.navigate('LearnDetails', { topic: 'statistics' })}
        />

        <TopicCard
          title="Living with CVD"
          description="Tips and strategies for daily life"
          icon="🏠"
          color="#FF6B6B"
          onPress={() => navigation.navigate('LearnDetails', { topic: 'living' })}
        />

        <TopicCard
          title="Genetics & Inheritance"
          description="How CVD is passed through families"
          icon="🧬"
          color="#9B59B6"
          onPress={() => navigation.navigate('LearnDetails', { topic: 'genetics' })}
        />

        <TopicCard
          title="Technology & Solutions"
          description="Modern tools and assistive technologies"
          icon="💡"
          color="#FFA500"
          onPress={() => navigation.navigate('LearnDetails', { topic: 'technology' })}
        />
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
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
  card: {
    backgroundColor: '#FFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    flexDirection: 'row',
    alignItems: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    borderLeftWidth: 5,
  },
  icon: {
    fontSize: 40,
    marginRight: 15,
  },
  cardContent: {
    flex: 1,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  cardDescription: {
    fontSize: 14,
    color: '#666',
  },
});

export default EducationScreen;
