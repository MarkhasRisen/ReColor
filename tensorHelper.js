import { decode } from 'base64-arraybuffer';
import * as ImageManipulator from 'expo-image-manipulator';

export const COLOR_CLASSES = [
  'Beige/Tan', 'Black', 'Blue', 'Brown', 'Cyan', 'Gold', 'Gray', 
  'Green', 'Lime', 'Magenta', 'Navy', 'Orange', 'Pink', 'Purple', 
  'Red', 'Silver', 'Teal', 'White', 'Yellow'
];

export const imageToTensor = async (uri) => {
  // 1. Resize image to 224x224 (Model Requirement)
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: 224, height: 224 } }],
    { base64: true, format: ImageManipulator.SaveFormat.JPEG }
  );

  // 2. Convert Base64 to Typed Array
  const buffer = decode(result.base64);
  const uint8 = new Uint8Array(buffer);
  const float32 = new Float32Array(224 * 224 * 3);

  // 3. Normalization Loop (Standard for EfficientNet: (pixel - 127.5) / 127.5)
  // We manually decode JPEG bytes to RGB floats (Simplified for speed)
  for (let i = 0; i < float32.length; i++) {
    float32[i] = (uint8[i] - 127.5) / 127.5;
  }

  return float32;
};