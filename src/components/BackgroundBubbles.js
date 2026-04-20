import React from 'react';
import { StyleSheet, View } from 'react-native';
import { styles } from '../theme/styles';

export default function BackgroundBubbles() {
  return (
    <View style={[StyleSheet.absoluteFill, { zIndex: -1, overflow: 'hidden' }]} pointerEvents="none">
      <View style={[styles.bubble, { top: -50, left: -50, width: 200, height: 200, backgroundColor: '#FFCDD2' }]} />
      <View style={[styles.bubble, { top: 100, left: -80, width: 120, height: 120, backgroundColor: '#E1BEE7' }]} />
      <View style={[styles.bubble, { bottom: -50, right: -50, width: 250, height: 250, backgroundColor: '#BBDEFB' }]} />
      <View style={[styles.bubble, { bottom: 200, right: -60, width: 150, height: 150, backgroundColor: '#C8E6C9' }]} />
      <View style={[styles.bubble, { top: '40%', left: '10%', width: 50, height: 50, backgroundColor: '#FFF9C4' }]} />
    </View>
  );
}
