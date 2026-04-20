import { Ionicons } from '@expo/vector-icons';
import React, { useState } from 'react';
import { ActivityIndicator, Alert, ScrollView, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { auth, signInWithEmailAndPassword } from '../../firebaseConfig';
import BackgroundBubbles from '../components/BackgroundBubbles';
import Header from '../components/Header';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

export default function AdminLoginScreen({ navigation }) {
  const [adminEmail, setAdminEmail] = useState('');
  const [adminPass, setAdminPass] = useState('');
  const [adminLoading, setAdminLoading] = useState(false);

  const handleAdminLogin = async () => {
    if (!adminEmail || !adminPass) {
      Alert.alert('Error', 'Please enter email and password.');
      return;
    }
    setAdminLoading(true);
    try {
      await signInWithEmailAndPassword(auth, adminEmail, adminPass);
      navigation.navigate('AdminHub', { role: 'admin' });
    } catch (e) {
      Alert.alert('Login Failed', 'Incorrect email or password.');
    } finally {
      setAdminLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Header title="Admin Portal" subtitle="Secure Access" back />
      <BackgroundBubbles />

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        <View style={{ alignItems: 'center', marginVertical: 30 }}>
          <View style={styles.iconCircleGradient}>
            <Ionicons name="lock-closed-outline" size={40} color="#FFF" />
          </View>
          <Text style={[styles.headerTitle, { marginTop: 15 }]}>Admin Login</Text>
          <Text style={{ color: COLORS.textLight }}>Access ReColor Admin & Expert Portal</Text>
        </View>

        <View style={styles.infoBox}>
          <Ionicons name="shield-checkmark-outline" size={20} color="#333" />
          <Text style={{ marginLeft: 10, flex: 1, fontSize: 12, color: '#444' }}>
            This portal uses secure Firebase authentication with role-based access control.
          </Text>
        </View>

        <Text style={styles.label}>Email Address</Text>
        <View style={styles.inputContainer}>
          <Ionicons name="mail-outline" size={20} color="#999" />
          <TextInput
            style={styles.input}
            placeholder="admin@recolor.app"
            value={adminEmail}
            onChangeText={setAdminEmail}
            autoCapitalize="none"
          />
        </View>

        <Text style={styles.label}>Password</Text>
        <View style={styles.inputContainer}>
          <Ionicons name="key-outline" size={20} color="#999" />
          <TextInput style={styles.input} placeholder="••••••••" secureTextEntry value={adminPass} onChangeText={setAdminPass} />
        </View>

        <TouchableOpacity
          style={[styles.btnPrimary, { marginTop: 20, backgroundColor: '#8E24AA' }]}
          onPress={handleAdminLogin}
        >
          {adminLoading ? (
            <ActivityIndicator color="#FFF" />
          ) : (
            <>
              <Ionicons name="lock-open-outline" size={20} color="#FFF" style={{ marginRight: 10 }} />
              <Text style={styles.btnText}>Sign In Securely</Text>
            </>
          )}
        </TouchableOpacity>

        <View style={{ marginTop: 30, backgroundColor: '#E3F2FD', padding: 15, borderRadius: 12, borderWidth: 1, borderColor: '#BBDEFB' }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
            <Ionicons name="information-circle-outline" size={20} color="#1565C0" />
            <Text style={{ marginLeft: 5, color: '#1565C0', fontWeight: 'bold' }}>Demo Credentials (Testing Only):</Text>
          </View>

          <TouchableOpacity
            style={{ backgroundColor: '#FFF', padding: 12, borderRadius: 8, marginBottom: 10, borderLeftWidth: 4, borderLeftColor: '#6C63FF' }}
            onPress={() => navigation.navigate('AdminHub', { role: 'researcher' })}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="flask-outline" size={20} color="#6C63FF" />
              <Text style={{ marginLeft: 10, fontWeight: 'bold', color: '#333' }}>PERI Researcher</Text>
            </View>
            <Text style={{ color: '#777', fontSize: 12, marginTop: 2 }}>researcher@peri.edu / peri123</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={{ backgroundColor: '#FFF', padding: 12, borderRadius: 8, borderLeftWidth: 4, borderLeftColor: '#9C27B0' }}
            onPress={() => navigation.navigate('AdminHub', { role: 'admin' })}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="settings-outline" size={20} color="#9C27B0" />
              <Text style={{ marginLeft: 10, fontWeight: 'bold', color: '#333' }}>System Administrator</Text>
            </View>
            <Text style={{ color: '#777', fontSize: 12, marginTop: 2 }}>admin@recolor.app / admin123</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}
