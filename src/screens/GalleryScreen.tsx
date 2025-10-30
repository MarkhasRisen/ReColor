import React from 'react';
import { View, Text, StyleSheet, SafeAreaView } from 'react-native';

const GalleryScreen = () => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.placeholder}>
        <Text style={styles.icon}>🖼️</Text>
        <Text style={styles.title}>Gallery</Text>
        <Text style={styles.description}>
          Gallery interface will be implemented here{'\n\n'}
          Features:{'\n'}
          • Browse device photos{'\n'}
          • Apply color enhancement{'\n'}
          • Before/after comparison{'\n'}
          • Save enhanced photos
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
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
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  description: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 24,
  },
});

export default GalleryScreen;
