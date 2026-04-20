import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { Text, View } from 'react-native';
import { styles } from '../theme/styles';

export default function DisclaimerBanner() {
  return (
    <View style={styles.disclaimerContainer}>
      <Ionicons name="warning-outline" size={14} color="#E65100" style={{ marginRight: 5 }} />
      <Text style={styles.disclaimerText}>Screening purpose only. Not a clinical diagnosis.</Text>
    </View>
  );
}
