import { Ionicons } from '@expo/vector-icons';
import { collection, onSnapshot, orderBy, query } from 'firebase/firestore';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, Text, View } from 'react-native';
import { auth, db, onAuthStateChanged } from '../../firebaseConfig';
import BackgroundBubbles from '../components/BackgroundBubbles';
import Card from '../components/Card';
import Header from '../components/Header';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

export default function HistoryScreen() {
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let unsubFirestore = null;

    const unsubAuth = onAuthStateChanged(auth, (user) => {
      if (unsubFirestore) { unsubFirestore(); unsubFirestore = null; }
      if (!user) { setHistoryData([]); setLoading(false); return; }

      setLoading(true);
      const q = query(
        collection(db, 'users', user.uid, 'history'),
        orderBy('date', 'desc'),
      );
      unsubFirestore = onSnapshot(
        q,
        (snap) => {
          setHistoryData(snap.docs.map((doc) => {
            const d = doc.data();
            return {
              id: doc.id,
              type: d.diagnosis || 'Unknown',
              score: d.score !== undefined ? d.score : '?',
              total: d.total || 14,
              severity: d.severity || 'N/A',
              date: d.date?.toDate ? d.date.toDate().toLocaleDateString() : 'Just now',
            };
          }));
          setLoading(false);
        },
        (err) => { console.error('History fetch error:', err); setLoading(false); },
      );
    });

    return () => { unsubAuth(); if (unsubFirestore) unsubFirestore(); };
  }, []);

  return (
    <View style={styles.container}>
      <Header title="Your History" back />
      <BackgroundBubbles />

      {loading ? (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <ActivityIndicator size="large" color={COLORS.primary} />
        </View>
      ) : historyData.length === 0 ? (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', opacity: 0.6 }}>
          <Ionicons name="clipboard-outline" size={60} color="#333" />
          <Text style={{ marginTop: 10, fontSize: 16 }}>No tests taken yet.</Text>
        </View>
      ) : (
        <ScrollView contentContainerStyle={{ padding: 20 }}>
          {historyData.map((item, index) => (
            <Card key={index} style={{ marginBottom: 15, paddingVertical: 15 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                <View>
                  <Text style={{ fontSize: 16, fontWeight: 'bold', color: '#333' }}>{item.type}</Text>
                  <Text style={{ fontSize: 12, color: '#888' }}>{item.date}</Text>
                </View>
                <View style={{ alignItems: 'flex-end' }}>
                  <Text style={{ fontSize: 20, fontWeight: 'bold', color: COLORS.primary }}>{item.score}/{item.total}</Text>
                  <Text style={{ fontSize: 11, fontWeight: 'bold', color: item.severity === 'Severe' ? COLORS.danger : COLORS.warning }}>
                    {item.severity}
                  </Text>
                </View>
              </View>
            </Card>
          ))}
        </ScrollView>
      )}
    </View>
  );
}
