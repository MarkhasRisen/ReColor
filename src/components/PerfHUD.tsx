import React from 'react';
import { View, Text, ViewStyle } from 'react-native';

type PerfHUDProps = { style?: ViewStyle };
export const PerfHUD: React.FC<PerfHUDProps> = ({ style }: PerfHUDProps) => {
  const [fps, setFps] = React.useState(0);

  const rafRef = React.useRef<number | null>(null);
  const lastTimeRef = React.useRef<number>(performance.now());
  const framesRef = React.useRef(0);
  const lastUpdateRef = React.useRef<number>(performance.now());

  const loop = React.useCallback(() => {
    const now = performance.now();
    framesRef.current += 1;

    // Update FPS roughly twice a second to reduce re-render cost
    if (now - lastUpdateRef.current >= 500) {
      const delta = now - lastTimeRef.current;
      const fpsNow = (framesRef.current / delta) * 1000;
      setFps(Math.round(fpsNow));
      framesRef.current = 0;
      lastTimeRef.current = now;
      lastUpdateRef.current = now;
    }

    rafRef.current = requestAnimationFrame(loop);
  }, []);

  React.useEffect(() => {
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [loop]);

  return (
    <View
      pointerEvents="none"
      style={[
        {
          position: 'absolute',
          top: 12,
          right: 12,
          backgroundColor: 'rgba(0,0,0,0.6)',
          paddingHorizontal: 8,
          paddingVertical: 4,
          borderRadius: 6,
          borderWidth: 1,
          borderColor: 'rgba(255,255,255,0.2)',
        },
        style,
      ]}
    >
      <Text style={{ color: 'white', fontVariant: ['tabular-nums'] }}>FPS: {fps}</Text>
    </View>
  );
};

export default PerfHUD;
