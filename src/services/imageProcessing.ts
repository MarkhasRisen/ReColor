/**
 * On-device image processing utilities
 * Handles image manipulation, canvas operations, and data conversions
 */

import { CVDType, daltonizeImage, simulateCVD } from './daltonization';

/**
 * User vision profile stored locally
 */
export interface VisionProfile {
  cvdType: CVDType;
  severity: number;
  userId: string;
  timestamp: number;
}

/**
 * Process image with daltonization correction
 * @param imageUri - Local URI to the image
 * @param profile - User's vision profile
 * @returns Base64 encoded processed image
 */
export async function processImageWithDaltonization(
  imageUri: string,
  profile: VisionProfile
): Promise<string> {
  try {
    // Note: Actual implementation would use react-native-image-manipulator
    // or similar library for real image processing
    
    // For now, return a placeholder
    // In production, you would:
    // 1. Load image from URI
    // 2. Get image data as Uint8ClampedArray
    // 3. Apply daltonizeImage()
    // 4. Convert back to base64

    return imageUri; // Placeholder
  } catch (error) {
    console.error('Error processing image:', error);
    throw new Error('Failed to process image');
  }
}

/**
 * Process camera frame in real-time
 * This is called for each frame in the color enhancement live view
 */
export function processFrame(
  frameData: Uint8ClampedArray,
  width: number,
  height: number,
  profile: VisionProfile
): Uint8ClampedArray {
  if (profile.cvdType === 'normal' || profile.severity === 0) {
    return frameData;
  }

  return daltonizeImage(frameData, profile.cvdType, profile.severity);
}

/**
 * Simulate CVD on an image (for education/preview)
 */
export function simulateCVDOnImage(
  imageData: Uint8ClampedArray,
  cvdType: CVDType
): Uint8ClampedArray {
  return simulateCVD(imageData, cvdType);
}

/**
 * Extract pixel data from a specific point (for color identifier)
 */
export function getPixelColor(
  imageData: Uint8ClampedArray,
  x: number,
  y: number,
  width: number
): number[] {
  const index = (y * width + x) * 4;
  return [
    imageData[index],     // R
    imageData[index + 1], // G
    imageData[index + 2], // B
  ];
}

/**
 * Get average color in a region (for better color identification)
 */
export function getRegionColor(
  imageData: Uint8ClampedArray,
  centerX: number,
  centerY: number,
  radius: number,
  width: number,
  height: number
): number[] {
  let totalR = 0;
  let totalG = 0;
  let totalB = 0;
  let count = 0;

  const radiusSq = radius * radius;

  for (let y = Math.max(0, centerY - radius); y < Math.min(height, centerY + radius); y++) {
    for (let x = Math.max(0, centerX - radius); x < Math.min(width, centerX + radius); x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      
      if (dx * dx + dy * dy <= radiusSq) {
        const index = (y * width + x) * 4;
        totalR += imageData[index];
        totalG += imageData[index + 1];
        totalB += imageData[index + 2];
        count++;
      }
    }
  }

  return [
    Math.round(totalR / count),
    Math.round(totalG / count),
    Math.round(totalB / count),
  ];
}

/**
 * Convert base64 image to ImageData
 * Used when loading images from storage
 */
export async function base64ToImageData(
  base64: string
): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  // Placeholder - actual implementation would use Canvas or Image libraries
  throw new Error('Not implemented - use react-native-image-manipulator');
}

/**
 * Convert ImageData to base64
 * Used when saving processed images
 */
export async function imageDataToBase64(
  imageData: Uint8ClampedArray,
  width: number,
  height: number
): Promise<string> {
  // Placeholder - actual implementation would use Canvas or Image libraries
  throw new Error('Not implemented - use react-native-image-manipulator');
}

/**
 * Performance optimization: downsample image for real-time processing
 */
export function downsampleImage(
  imageData: Uint8ClampedArray,
  width: number,
  height: number,
  scale: number
): { data: Uint8ClampedArray; width: number; height: number } {
  const newWidth = Math.floor(width * scale);
  const newHeight = Math.floor(height * scale);
  const newData = new Uint8ClampedArray(newWidth * newHeight * 4);

  const xRatio = width / newWidth;
  const yRatio = height / newHeight;

  for (let y = 0; y < newHeight; y++) {
    for (let x = 0; x < newWidth; x++) {
      const srcX = Math.floor(x * xRatio);
      const srcY = Math.floor(y * yRatio);
      
      const srcIndex = (srcY * width + srcX) * 4;
      const dstIndex = (y * newWidth + x) * 4;

      newData[dstIndex] = imageData[srcIndex];
      newData[dstIndex + 1] = imageData[srcIndex + 1];
      newData[dstIndex + 2] = imageData[srcIndex + 2];
      newData[dstIndex + 3] = imageData[srcIndex + 3];
    }
  }

  return { data: newData, width: newWidth, height: newHeight };
}
