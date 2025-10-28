import React from 'react';
import { View, Text, StyleSheet, SafeAreaView } from 'react-native';

const CVDSimulationScreen = () => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.placeholder}>
        <Text style={styles.icon}>👁️</Text>
        <Text style={styles.title}>CVD Simulation</Text>
        <Text style={styles.description}>
          Camera interface will be implemented here{'\n\n'}
          Features:{'\n'}
          • Simulate protan vision{'\n'}
          • Simulate deutan vision{'\n'}
          • Simulate tritan vision{'\n'}
          • Educational tool for awareness
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 30,
  },
  icon: {
    fontSize: 64,
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 15,
    textAlign: 'center',
  },
  description: {
    fontSize: 16,
    color: '#CCC',
    textAlign: 'center',
    lineHeight: 24,
  },
});

export default CVDSimulationScreen;
