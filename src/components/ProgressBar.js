import React from 'react';
import { Text, View } from 'react-native';
import { COLORS } from '../theme/colors';

export default function ProgressBar({ label, value, color, count, percentage }) {
  return (
    <View style={{ marginBottom: 15 }}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 }}>
        <Text style={{ fontSize: 14, fontWeight: '600', color: '#555' }}>{label}</Text>
        <Text style={{ fontSize: 14, fontWeight: 'bold' }}>
          {count} <Text style={{ color: COLORS.textLight }}>({percentage})</Text>
        </Text>
      </View>
      <View style={{ height: 10, backgroundColor: '#E0E0E0', borderRadius: 5, overflow: 'hidden' }}>
        <View style={{ width: value, height: '100%', backgroundColor: color }} />
      </View>
    </View>
  );
}
