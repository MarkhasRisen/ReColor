import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { Image, ImageBackground, ScrollView, Text, TouchableOpacity, View } from 'react-native';
import { auth, signOut } from '../../firebaseConfig';
import BackgroundBubbles from '../components/BackgroundBubbles';
import Card from '../components/Card';
import Header from '../components/Header';
import { COLORS } from '../theme/colors';

export default function AdminHubScreen({ route, navigation }) {
  const { role } = route.params || { role: 'researcher' };
  const isResearcher = role === 'researcher';

  return (
    <ScrollView style={{ flex: 1, backgroundColor: COLORS.background }} showsVerticalScrollIndicator={false}>
      <Header title="Admin Hub" subtitle={isResearcher ? 'Dr. PERI Researcher' : 'System Administrator'} back />
      <BackgroundBubbles />

      <View style={{ padding: 20 }}>
        <ImageBackground
          style={{ width: '100%', padding: 25, borderRadius: 20, overflow: 'hidden', marginBottom: 20, backgroundColor: isResearcher ? '#6200EA' : '#651FFF' }}
          source={{ uri: 'https://placehold.co/400x150/6200EA/B388FF?text= ' }}
        >
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <View style={{ width: 60, height: 60, borderRadius: 30, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' }}>
              <Ionicons name="person" size={30} color="#FFF" />
            </View>
            <View style={{ marginLeft: 15 }}>
              <Text style={{ color: '#FFF', fontSize: 18, fontWeight: 'bold' }}>
                {isResearcher ? 'Dr. PERI Researcher' : 'System Administrator'}
              </Text>
              <Text style={{ color: 'rgba(255,255,255,0.9)', marginBottom: 5 }}>
                {isResearcher ? 'researcher@peri.edu' : 'admin@recolor.app'}
              </Text>
              <View style={{ backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12, alignSelf: 'flex-start' }}>
                <Text style={{ color: '#FFF', fontSize: 10, fontWeight: 'bold' }}>
                  {isResearcher ? 'PERI Researcher' : 'System Administrator'}
                </Text>
              </View>
            </View>
          </View>
        </ImageBackground>

        <View style={{ backgroundColor: '#E8F5E9', padding: 15, borderRadius: 12, flexDirection: 'row', alignItems: 'center', marginBottom: 20, borderWidth: 1, borderColor: '#C8E6C9' }}>
          <View style={{ width: 12, height: 12, borderRadius: 6, backgroundColor: COLORS.success, marginRight: 10 }} />
          <View>
            <Text style={{ fontWeight: 'bold', color: '#2E7D32' }}>System Status: All Systems Operational</Text>
            <Text style={{ fontSize: 12, color: '#666' }}>Last checked: Just now</Text>
          </View>
        </View>

        <Card style={{ marginBottom: 20 }}>
          <Text style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 5 }}>System Architecture</Text>
          <Text style={{ fontSize: 12, color: '#666', marginBottom: 15 }}>ReColor Admin/Expert Flow Diagram</Text>
          <Image source={{ uri: 'https://placehold.co/600x300/FFF/000?text=Flow+Diagram' }} style={{ width: '100%', height: 150, resizeMode: 'contain', marginBottom: 15 }} />
          <View style={{ backgroundColor: '#E3F2FD', padding: 10, borderRadius: 8, flexDirection: 'row', alignItems: 'center' }}>
            <Ionicons name="information-circle-outline" size={16} color="#1565C0" style={{ marginRight: 8 }} />
            <Text style={{ fontSize: 11, color: '#1565C0', flex: 1 }}>
              Following secure Firebase authentication with role-based access control
            </Text>
          </View>
        </Card>

        <Text style={{ color: '#999', fontSize: 12, marginBottom: 10, letterSpacing: 1 }}>YOUR ACCESS PORTALS</Text>

        <Card onPress={() => navigation.navigate('ResearchDashboard')} style={{ flexDirection: 'row', alignItems: 'center' }}>
          <View style={{ width: 50, height: 50, borderRadius: 25, backgroundColor: isResearcher ? '#E3F2FD' : '#F3E5F5', alignItems: 'center', justifyContent: 'center' }}>
            <Ionicons name={isResearcher ? 'bar-chart' : 'settings'} size={24} color={isResearcher ? '#2196F3' : '#9C27B0'} />
          </View>
          <View style={{ marginLeft: 15, flex: 1 }}>
            <Text style={{ fontSize: 16, fontWeight: 'bold' }}>{isResearcher ? 'Research Dashboard' : 'System Management'}</Text>
            <Text style={{ fontSize: 12, color: '#666' }}>{isResearcher ? 'View anonymized data • Statistics' : 'User management • Content CMS'}</Text>
          </View>
          {isResearcher && (
            <View style={{ backgroundColor: '#E3F2FD', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 }}>
              <Text style={{ color: '#2196F3', fontSize: 10, fontWeight: 'bold' }}>Primary</Text>
            </View>
          )}
        </Card>

        <TouchableOpacity
          style={{ marginTop: 30, backgroundColor: '#D32F2F', padding: 15, borderRadius: 12, alignItems: 'center', flexDirection: 'row', justifyContent: 'center' }}
          onPress={() => { signOut(auth).catch(() => {}); navigation.navigate('Login'); }}
        >
          <Ionicons name="log-out-outline" size={20} color="#FFF" style={{ marginRight: 10 }} />
          <Text style={{ fontWeight: 'bold', color: '#FFF' }}>Logout from Admin Portal</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}
