import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { ScrollView, Text, View } from 'react-native';
import Card from '../components/Card';
import Header from '../components/Header';
import { styles } from '../theme/styles';

export default function IshiharaIntroScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Header title="Choose Test Type" back />
      <ScrollView contentContainerStyle={{ padding: 20 }}>
        <Text style={{ color: '#777', marginBottom: 20 }}>Select the test duration that works best for you</Text>

        <Card onPress={() => navigation.navigate('IshiharaOnboarding', { testType: 'comprehensive' })} style={{ marginBottom: 20 }}>
          <View style={{ flexDirection: 'row' }}>
            <View style={{ flex: 1 }}>
              <Text style={styles.cardTitle}>Comprehensive Test</Text>
              <Text style={styles.cardDesc}>Complete 38-plate assessment</Text>
              <View style={{ marginTop: 10, flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="time-outline" size={16} color="#666" />
                <Text style={{ fontSize: 12, marginLeft: 5, color: '#666' }}>15-20 minutes</Text>
              </View>
            </View>
            <View style={{ justifyContent: 'center', alignItems: 'center' }}>
              <View style={[styles.iconCircle, { backgroundColor: '#E3F2FD' }]}>
                <Ionicons name="shield-checkmark" size={24} color="#2196F3" />
              </View>
              <Text style={{ fontSize: 10, color: '#2196F3', marginTop: 5, textAlign: 'center' }}>Accurate</Text>
            </View>
          </View>
        </Card>

        <Card onPress={() => navigation.navigate('IshiharaOnboarding', { testType: 'quick' })} style={{ marginBottom: 20 }}>
          <View style={{ flexDirection: 'row' }}>
            <View style={{ flex: 1 }}>
              <Text style={styles.cardTitle}>Quick Test</Text>
              <Text style={styles.cardDesc}>14-plate screening</Text>
              <View style={{ marginTop: 10, flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="time-outline" size={16} color="#666" />
                <Text style={{ fontSize: 12, marginLeft: 5, color: '#666' }}>5-8 minutes</Text>
              </View>
            </View>
            <View style={{ justifyContent: 'center', alignItems: 'center' }}>
              <View style={[styles.iconCircle, { backgroundColor: '#F3E5F5' }]}>
                <Ionicons name="flash" size={24} color="#9C27B0" />
              </View>
              <Text style={{ fontSize: 10, color: '#9C27B0', marginTop: 5, textAlign: 'center' }}>Fast</Text>
            </View>
          </View>
        </Card>
      </ScrollView>
    </View>
  );
}
