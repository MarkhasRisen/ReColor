import React from 'react';
import { View, Text, StyleSheet, SafeAreaView } from 'react-native';

const ColorIdentifierScreen = () => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.placeholder}>
        <Text style={styles.icon}>🎯</Text>
        <Text style={styles.title}>Color Identifier</Text>
        <Text style={styles.description}>
          Camera interface will be implemented here{'\n\n'}
          Features:{'\n'}
          • Point-and-identify colors{'\n'}
          • RGB, HEX, and color name display{'\n'}
          • Voice announcement option{'\n'}
          • Color history log
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

export default ColorIdentifierScreen;
