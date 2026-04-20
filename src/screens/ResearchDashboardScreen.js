import { Ionicons } from '@expo/vector-icons';
import React, { useState } from 'react';
import { ScrollView, Text, TextInput, TouchableOpacity, View } from 'react-native';
import BackgroundBubbles from '../components/BackgroundBubbles';
import Card from '../components/Card';
import Header from '../components/Header';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

function StatBar({ label, count, percent, color, badge }) {
  return (
    <View style={{ marginBottom: 15 }}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <Text style={{ fontSize: 13, color: '#333' }}>{label}</Text>
          {badge && (
            <View style={{ marginLeft: 8, borderWidth: 1, borderColor: '#EEE', paddingHorizontal: 6, borderRadius: 4 }}>
              <Text style={{ fontSize: 10, color: '#555' }}>{badge}</Text>
            </View>
          )}
        </View>
        <Text style={{ fontWeight: 'bold', fontSize: 13 }}>{count} <Text style={{ color: '#999', fontWeight: 'normal' }}>({percent})</Text></Text>
      </View>
      <View style={{ height: 8, backgroundColor: '#F0F0F0', borderRadius: 4 }}>
        <View style={{ width: percent, height: '100%', backgroundColor: color, borderRadius: 4 }} />
      </View>
    </View>
  );
}

export default function ResearchDashboardScreen({ navigation }) {
  const [activeTab, setActiveTab] = useState('View Data');

  return (
    <View style={styles.container}>
      <Header title="Research Dashboard" subtitle="PERI Researcher Portal" back />
      <BackgroundBubbles />

      <View style={{ flexDirection: 'row', padding: 20, paddingBottom: 0 }}>
        {['View Data', 'Guidelines'].map((tab, i) => (
          <TouchableOpacity
            key={tab}
            style={{ flex: 1, paddingVertical: 10, backgroundColor: activeTab === tab ? '#FFF' : '#F5F5F5', alignItems: 'center', borderTopLeftRadius: i === 0 ? 20 : 0, borderBottomLeftRadius: i === 0 ? 20 : 0, borderTopRightRadius: i === 1 ? 20 : 0, borderBottomRightRadius: i === 1 ? 20 : 0, borderWidth: 1, borderColor: '#E0E0E0', borderRightWidth: i === 0 ? 0 : 1 }}
            onPress={() => setActiveTab(tab)}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name={i === 0 ? 'stats-chart' : 'document-text'} size={18} color={activeTab === tab ? '#333' : '#999'} />
              <Text style={{ marginLeft: 8, fontWeight: 'bold', color: activeTab === tab ? '#333' : '#999' }}>{tab}</Text>
            </View>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        {activeTab === 'View Data' && (
          <>
            <View style={{ backgroundColor: '#E1F5FE', padding: 15, borderRadius: 8, marginBottom: 20, borderWidth: 1, borderColor: '#B3E5FC' }}>
              <Text style={{ color: '#0277BD', fontSize: 12, lineHeight: 18 }}>
                <Text style={{ fontWeight: 'bold' }}>Privacy Protected: </Text>All data is anonymized. No personally identifiable information (PII) is visible.
              </Text>
            </View>

            <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 20 }}>
              {[
                { bg: '#E3F2FD', icon: 'trending-up', iconColor: '#1E88E5', val: '3891', label: 'Total Tests', valColor: '#1565C0' },
                { bg: '#F3E5F5', icon: 'people', iconColor: '#8E24AA', val: '1247', label: 'Participants', valColor: '#6A1B9A' },
                { bg: '#E8F5E9', icon: 'time-outline', iconColor: '#43A047', val: '8.5 min', label: 'Avg Time', valColor: '#2E7D32' },
                { bg: '#FCE4EC', icon: 'checkbox-outline', iconColor: '#E91E63', val: '94%', label: 'Data Quality', valColor: '#C2185B' },
              ].map((s) => (
                <View key={s.label} style={{ width: '48%', backgroundColor: s.bg, padding: 15, borderRadius: 12, marginBottom: 10 }}>
                  <Ionicons name={s.icon} size={20} color={s.iconColor} />
                  <Text style={{ fontSize: 24, fontWeight: 'bold', color: s.valColor, marginTop: 10 }}>{s.val}</Text>
                  <Text style={{ fontSize: 11, color: s.iconColor }}>{s.label}</Text>
                </View>
              ))}
            </View>

            <TouchableOpacity style={{ backgroundColor: '#FFF', borderWidth: 1, borderColor: '#DDD', padding: 12, borderRadius: 8, alignItems: 'center', marginBottom: 25 }}>
              <Text style={{ fontWeight: 'bold', color: '#333' }}>
                <Ionicons name="download-outline" size={16} /> Export Full Report (CSV)
              </Text>
            </TouchableOpacity>

            <Card style={{ marginBottom: 20 }}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 20 }}>
                <Ionicons name="bar-chart-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Analyze Screening Statistics</Text>
              </View>
              <StatBar label="Normal Vision" badge="N/A" count={478} percent="38%" color={COLORS.success} />
              <StatBar label="Protanomaly" badge="Mild-Moderate" count={412} percent="33%" color="#2979FF" />
              <StatBar label="Deuteranomaly" badge="Mild-Moderate" count={289} percent="23%" color="#E040FB" />
              <StatBar label="Tritanomaly" badge="Mild" count={68} percent="6%" color="#FF5722" />
            </Card>

            <Card>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 20 }}>
                <Ionicons name="people-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Demographics Analysis</Text>
              </View>
              <Text style={{ fontSize: 12, color: '#666', marginBottom: 15 }}>Age distribution (anonymized data only)</Text>
              <StatBar label="Age 18-30" count={534} percent="43%" color="#03A9F4" />
              <StatBar label="Age 31-50" count={412} percent="33%" color="#E040FB" />
              <StatBar label="Age 51+" count={301} percent="24%" color="#FF9800" />
            </Card>
          </>
        )}

        {activeTab === 'Guidelines' && (
          <>
            <Card style={{ marginBottom: 20 }}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
                <Ionicons name="document-text-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Update Clinical Protocols</Text>
              </View>
              <Text style={{ color: '#666', fontSize: 12, marginBottom: 15 }}>Modify testing guidelines and recommendations</Text>
              <View style={{ backgroundColor: '#F5F5F5', borderRadius: 8, padding: 15, marginBottom: 15 }}>
                <Text style={{ color: '#333', fontSize: 12, fontWeight: 'bold', marginBottom: 5 }}>Protocol Updates</Text>
                <TextInput placeholder="Enter updated clinical protocols, testing procedures..." multiline style={{ height: 60, textAlignVertical: 'top', fontSize: 12 }} />
              </View>
              <TouchableOpacity style={[styles.btnPrimary, { backgroundColor: '#7B1FA2' }]}>
                <Ionicons name="save-outline" size={18} color="#FFF" style={{ marginRight: 8 }} />
                <Text style={styles.btnText}>Update Protocol</Text>
              </TouchableOpacity>
              <View style={{ marginTop: 20, backgroundColor: '#E3F2FD', padding: 15, borderRadius: 8 }}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
                  <Ionicons name="information-circle" size={16} color="#1565C0" />
                  <Text style={{ marginLeft: 5, color: '#1565C0', fontWeight: 'bold', fontSize: 12 }}>Current Active Protocols:</Text>
                </View>
                <Text style={{ fontSize: 11, color: '#555', lineHeight: 18 }}>• Ishihara Test - 38 plates (comprehensive){'\n'}• Quick Test - 14 plates for rapid screening{'\n'}• Color Enhancement filters - Calibrated{'\n'}• AI Color Identifier - 95% accuracy baseline</Text>
              </View>
            </Card>

            <Card>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
                <Ionicons name="notifications-outline" size={20} color="#333" />
                <Text style={{ fontWeight: 'bold', marginLeft: 10, fontSize: 16 }}>Push Notifications to Users</Text>
              </View>
              <Text style={{ color: '#666', fontSize: 12, marginBottom: 15 }}>Send announcements to all app users</Text>
              <View style={{ backgroundColor: '#F5F5F5', borderRadius: 8, padding: 15, marginBottom: 15 }}>
                <Text style={{ color: '#333', fontSize: 12, fontWeight: 'bold', marginBottom: 5 }}>Notification Message</Text>
                <TextInput placeholder="Enter notification message..." style={{ fontSize: 12 }} />
              </View>
              <TouchableOpacity style={[styles.btnPrimary, { backgroundColor: '#C2185B' }]}>
                <Ionicons name="notifications" size={18} color="#FFF" style={{ marginRight: 8 }} />
                <Text style={styles.btnText}>Send Notification to All Users</Text>
              </TouchableOpacity>
            </Card>
          </>
        )}
      </ScrollView>
    </View>
  );
}
