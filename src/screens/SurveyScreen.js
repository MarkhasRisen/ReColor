import { Ionicons } from '@expo/vector-icons';
import { addDoc, collection, getDocs, query, serverTimestamp, where } from 'firebase/firestore';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, ScrollView, Text, TouchableOpacity, View } from 'react-native';
import { auth, db } from '../../firebaseConfig';
import Card from '../components/Card';
import Header from '../components/Header';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

const CAUSES = ['Medical Intake', 'Genetics', 'Ageing', 'Others'];

export default function SurveyScreen({ navigation }) {
  const [selectedCause, setSelectedCause] = useState(null);
  const [selectedSex, setSelectedSex] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [alreadySubmitted, setAlreadySubmitted] = useState(false);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const uid = auth.currentUser?.uid;
    if (!uid) { setChecking(false); return; }
    getDocs(query(collection(db, 'surveys'), where('userId', '==', uid)))
      .then((snap) => setAlreadySubmitted(!snap.empty))
      .catch(() => {})
      .finally(() => setChecking(false));
  }, []);

  const handleSubmit = async () => {
    if (alreadySubmitted) {
      Alert.alert('Already Submitted', 'You have already completed this survey. Thank you for your contribution!');
      return;
    }
    if (selectedCause === null || !selectedSex) {
      Alert.alert('Incomplete', 'Please answer all questions before submitting.');
      return;
    }
    setSubmitting(true);
    try {
      const surveyData = {
        cause: CAUSES[selectedCause],
        sex: selectedSex,
        userId: auth.currentUser?.uid || 'anonymous',
        timestamp: serverTimestamp(),
      };
      await addDoc(collection(db, 'surveys'), surveyData);
      if (auth.currentUser) {
        await addDoc(collection(db, 'users', auth.currentUser.uid, 'surveys'), surveyData);
      }
      navigation.replace('SurveySuccess');
    } catch (e) {
      console.warn('[Survey] save failed:', e);
      navigation.replace('SurveySuccess');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <View style={styles.container}>
      <Header title="Quick Survey" back />
      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        <Text style={{ textAlign: 'center', fontWeight: 'bold', fontSize: 18, marginBottom: 5 }}>Help Us Understand</Text>
        <Text style={{ textAlign: 'center', color: '#777', marginBottom: 25 }}>What do you think are the causes of your CVD?</Text>

        <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 20 }}>
          {CAUSES.map((item, idx) => (
            <TouchableOpacity
              key={idx}
              onPress={() => setSelectedCause(idx)}
              style={[styles.surveyOption, selectedCause === idx && { borderColor: COLORS.primary, borderWidth: 2, backgroundColor: '#F3E5F5' }]}
            >
              <Ionicons
                name={selectedCause === idx ? 'checkmark-circle' : 'radio-button-off'}
                size={24}
                color={selectedCause === idx ? COLORS.primary : '#CCC'}
              />
              <Text style={{ fontWeight: 'bold', marginTop: 10 }}>{item}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <Card>
          <Text style={{ fontWeight: 'bold', marginBottom: 15 }}>Sex</Text>
          <View style={{ flexDirection: 'row', justifyContent: 'space-around' }}>
            {['Male', 'Female'].map((sex) => (
              <TouchableOpacity
                key={sex}
                style={{ flexDirection: 'row', alignItems: 'center', padding: 10, borderWidth: 1, borderColor: selectedSex === sex ? COLORS.primary : '#EEE', borderRadius: 8, width: '45%', justifyContent: 'center' }}
                onPress={() => setSelectedSex(sex)}
              >
                <Ionicons name={selectedSex === sex ? 'radio-button-on' : 'radio-button-off'} size={20} color={selectedSex === sex ? COLORS.primary : '#999'} />
                <Text style={{ marginLeft: 10, fontWeight: selectedSex === sex ? 'bold' : 'normal' }}>{sex}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </Card>

        {checking ? (
          <ActivityIndicator style={{ marginTop: 30 }} color={COLORS.primary} />
        ) : alreadySubmitted ? (
          <View style={{ marginTop: 30, backgroundColor: '#E8F5E9', padding: 16, borderRadius: 12, alignItems: 'center', borderWidth: 1, borderColor: '#C8E6C9' }}>
            <Ionicons name="checkmark-circle" size={28} color={COLORS.success} />
            <Text style={{ color: '#2E7D32', fontWeight: '700', marginTop: 8 }}>Survey Already Submitted</Text>
            <Text style={{ color: '#555', fontSize: 12, marginTop: 4, textAlign: 'center' }}>Thank you for your contribution to the research.</Text>
          </View>
        ) : (
          <TouchableOpacity
            style={[styles.btnPrimary, { marginTop: 30, backgroundColor: '#111' }]}
            onPress={handleSubmit}
            disabled={submitting}
          >
            {submitting ? <ActivityIndicator color="#FFF" /> : <Text style={styles.btnText}>Submit Survey</Text>}
          </TouchableOpacity>
        )}
      </ScrollView>
    </View>
  );
}
