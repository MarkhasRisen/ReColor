import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';

const SplashScreen = ({ navigation }: any) => {
  useEffect(() => {
    // Simulate loading, then navigate to Login
    setTimeout(() => {
      navigation.replace('Auth');
    }, 2000);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.logoPlaceholder}>
        <Text style={styles.logoText}>RC</Text>
      </View>
      <Text style={styles.title}>ReColor</Text>
      <Text style={styles.subtitle}>Color Vision Enhancement</Text>
      <ActivityIndicator size="large" color="#4A90E2" style={styles.loader} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoPlaceholder: {
    width: 120,
    height: 120,
    backgroundColor: '#4A90E2',
    borderRadius: 60,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  logoText: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#4A90E2',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 40,
  },
  loader: {
    marginTop: 20,
  },
});

export default SplashScreen;
