import React, { useRef, useState } from 'react';
import { Animated, Pressable } from 'react-native';
import { COLORS } from '../theme/colors';
import { styles } from '../theme/styles';

export default function Card({ children, style, onPress }) {
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const [isPressed, setIsPressed] = useState(false);

  const handlePressIn = () => {
    setIsPressed(true);
    Animated.spring(scaleAnim, { toValue: 0.96, useNativeDriver: true, speed: 20, bounciness: 10 }).start();
  };

  const handlePressOut = () => {
    setIsPressed(false);
    Animated.spring(scaleAnim, { toValue: 1, useNativeDriver: true, speed: 20, bounciness: 10 }).start();
  };

  return (
    <Pressable
      onPress={onPress}
      onPressIn={onPress ? handlePressIn : null}
      onPressOut={onPress ? handlePressOut : null}
      style={{ marginBottom: 15 }}
    >
      <Animated.View
        style={[
          styles.card,
          style,
          {
            transform: [{ scale: scaleAnim }],
            borderColor: isPressed ? COLORS.primary : 'transparent',
            borderWidth: isPressed ? 1 : 0,
            opacity: isPressed ? 0.9 : 1,
            marginBottom: 0,
          },
        ]}
      >
        {children}
      </Animated.View>
    </Pressable>
  );
}
