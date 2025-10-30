import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';

const CameraScreen = ({ navigation }: any) => {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Real-time Camera Features</Text>
          <Text style={styles.subtitle}>Choose a feature to get started</Text>
        </View>

        <TouchableOpacity
          style={[styles.featureCard, { borderLeftColor: '#FF6B6B' }]}
          onPress={() => navigation.navigate('ColorEnhancement')}
        >
          <Text style={styles.featureIcon}>🎨</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureTitle}>Color Enhancement</Text>
            <Text style={styles.featureDescription}>
              Real-time color correction for better visibility
            </Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.featureCard, { borderLeftColor: '#4A90E2' }]}
          onPress={() => navigation.navigate('ColorIdentifier')}
        >
          <Text style={styles.featureIcon}>🎯</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureTitle}>Color Identifier</Text>
            <Text style={styles.featureDescription}>
              Point at any object to identify its color
            </Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.featureCard, { borderLeftColor: '#9B59B6' }]}
          onPress={() => navigation.navigate('CVDSimulation')}
        >
          <Text style={styles.featureIcon}>👁️</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureTitle}>CVD Simulation</Text>
            <Text style={styles.featureDescription}>
              See how others with color blindness see the world
            </Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.featureCard, { borderLeftColor: '#50C878' }]}
          onPress={() => navigation.navigate('Gallery')}
        >
          <Text style={styles.featureIcon}>🖼️</Text>
          <View style={styles.featureContent}>
            <Text style={styles.featureTitle}>Gallery</Text>
            <Text style={styles.featureDescription}>
              Apply color enhancement to saved photos
            </Text>
          </View>
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
  featureCard: {
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
  featureIcon: {
    fontSize: 40,
    marginRight: 15,
  },
  featureContent: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  featureDescription: {
    fontSize: 14,
    color: '#666',
  },
});

export default CameraScreen;
