import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { Text, TouchableOpacity, View } from 'react-native';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

export default function SurveySuccessScreen({ navigation }) {
  return (
    <View style={[styles.container, { justifyContent: 'center', alignItems: 'center', backgroundColor: '#F8F9FA' }]}>
      <View style={{ width: 80, height: 80, borderRadius: 40, backgroundColor: '#E8F5E9', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
        <Ionicons name="checkmark" size={40} color={COLORS.success} />
      </View>
      <Text style={{ fontSize: 22, fontWeight: 'bold', color: '#333' }}>Thank You!</Text>
      <Text style={{ color: '#777', marginTop: 10 }}>Your response has been recorded</Text>

      <TouchableOpacity
        style={[styles.btnOutline, { marginTop: 40, width: 200, backgroundColor: '#FFF' }]}
        onPress={() => navigation.navigate('MainTabs')}
      >
        <Text style={{ color: COLORS.primary, fontWeight: 'bold' }}>Back to Home</Text>
      </TouchableOpacity>
    </View>
  );
}
