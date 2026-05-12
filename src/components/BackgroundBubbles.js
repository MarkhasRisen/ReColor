import { useEffect } from "react";
import { Dimensions, StyleSheet, View } from "react-native";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withRepeat,
  withSequence,
  withTiming,
} from "react-native-reanimated";

const { width, height } = Dimensions.get("window");

const FloatingBubble = ({
  size,
  color,
  top,
  left,
  duration,
  delay = 0,
  opacity = 0.3,
}) => {
  const translateY = useSharedValue(0);
  const translateX = useSharedValue(0);

  useEffect(() => {
    translateY.value = withDelay(
      delay,
      withRepeat(
        withSequence(
          withTiming(-30, { duration }),
          withTiming(30, { duration }),
        ),
        -1,
        true,
      ),
    );

    translateX.value = withDelay(
      delay,
      withRepeat(
        withSequence(
          withTiming(20, { duration: duration * 1.5 }),
          withTiming(-20, { duration: duration * 1.5 }),
        ),
        -1,
        true,
      ),
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateY: translateY.value },
      { translateX: translateX.value },
    ],
  }));

  return (
    <Animated.View
      style={[
        animatedStyle,
        {
          position: "absolute",
          top,
          left,
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: color,
          opacity,
        },
      ]}
    />
  );
};

export default function BackgroundBubbles() {
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Top Cluster */}
      <FloatingBubble
        size={200}
        color="#FFCDD2"
        top={-40}
        left={-60}
        duration={4000}
      />
      <FloatingBubble
        size={80}
        color="#E1BEE7"
        top={120}
        left={20}
        duration={3000}
        delay={500}
      />

      {/* Middle Floating accents */}
      <FloatingBubble
        size={40}
        color="#FFF9C4"
        top={height * 0.3}
        left={width * 0.1}
        duration={2500}
        opacity={0.5}
      />
      <FloatingBubble
        size={120}
        color="#C8E6C9"
        top={height * 0.4}
        left={width * 0.8}
        duration={5000}
      />
      <FloatingBubble
        size={60}
        color="#BBDEFB"
        top={height * 0.55}
        left={width * 0.05}
        duration={3500}
        delay={200}
      />

      {/* Bottom Cluster */}
      <FloatingBubble
        size={280}
        color="#BBDEFB"
        top={height * 0.75}
        left={width * 0.5}
        duration={6000}
      />
      <FloatingBubble
        size={100}
        color="#FFCDD2"
        top={height * 0.85}
        left={width * 0.1}
        duration={4500}
      />
      <FloatingBubble
        size={150}
        color="#E1BEE7"
        top={height * 0.6}
        left={width * 0.7}
        duration={5500}
        delay={1000}
      />

      {/* Extra "Bubbly" Small Orbs */}
      <FloatingBubble
        size={25}
        color="#C8E6C9"
        top={height * 0.1}
        left={width * 0.7}
        duration={2000}
      />
      <FloatingBubble
        size={30}
        color="#FFF9C4"
        top={height * 0.9}
        left={width * 0.8}
        duration={3000}
      />
      <FloatingBubble
        size={20}
        color="#FFCDD2"
        top={height * 0.25}
        left={width * 0.85}
        duration={2200}
      />
      <FloatingBubble
        size={35}
        color="#BBDEFB"
        top={height * 0.05}
        left={width * 0.3}
        duration={2800}
      />
    </View>
  );
}
