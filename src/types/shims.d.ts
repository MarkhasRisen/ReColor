// Temporary shims for missing type packages in dev
// Prefer installing: npm i -D @types/react @types/react-native

declare module 'react/jsx-runtime' {
  const ReactJSXRuntime: any;
  export = ReactJSXRuntime;
}

declare module 'react-native-screen-brightness' {
  const mod: {
    getBrightness: () => Promise<number>;
    setBrightness: (value: number) => Promise<void>;
  };
  export default mod;
}

declare module 'react-native-keep-awake' {
  export function activateKeepAwake(tag?: string): void;
  export function deactivateKeepAwake(tag?: string): void;
}

declare module '@react-native-async-storage/async-storage' {
  const AsyncStorage: {
    getItem: (key: string) => Promise<string | null>;
    setItem: (key: string, value: string) => Promise<void>;
    removeItem: (key: string) => Promise<void>;
  };
  export default AsyncStorage;
}

// Minimal Skia typings to unblock TS on specific hooks/components we use
declare module '@shopify/react-native-skia' {
  import * as React from 'react';
  export type SkImage = any;
  export type SkiaValue<T> = { current: T };
  export function useValue<T>(initial: T): SkiaValue<T>;
  export const useSharedValueEffect: (cb: () => void, sharedValue: { value: any }) => void;

  export const Canvas: React.ComponentType<any>;
  export const Fill: React.ComponentType<any>;
  export const ImageShader: React.ComponentType<any>;
  export const RuntimeShader: React.ComponentType<any>;
  export const Image: React.ComponentType<any>;
  export const Rect: React.ComponentType<any>;
  export const Mask: React.ComponentType<any>;
  export const Skia: any;
}
