"""
Dataset Manager for Color Detection Training.
Handles generation of synthetic color datasets for training K-means and neural network models.
Based on Dataset.ipynb notebook implementation.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import colorsys
import random
import logging
from typing import Dict, Tuple, List, Optional


class ColorDatasetGenerator:
    """Generate synthetic color datasets for training."""
    
    def __init__(self, base_dir: str = "color_datasets"):
        """
        Initialize dataset generator.
        
        Args:
            base_dir: Base directory for datasets
        """
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Define color families with hue values (in degrees)
        self.color_families = {
            "red": 0,
            "orange": 30,
            "yellow": 60,
            "green": 120,
            "cyan": 180,
            "blue": 240,
            "purple": 300,
            "pink": 330,
            "brown": 20  # Brown is dark orange
        }
    
    def generate_rgb_hsv_dataset(self, 
                                 output_dir: str = "color_dataset_rgb_hsv",
                                 samples_per_family: int = 1000,
                                 img_size: Tuple[int, int] = (64, 64)) -> bool:
        """
        Generate HSV-based RGB color dataset with controlled variation.
        
        Args:
            output_dir: Output directory name
            samples_per_family: Number of samples per color family
            img_size: Image dimensions (width, height)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            dataset_dir = os.path.join(self.base_dir, output_dir)
            os.makedirs(dataset_dir, exist_ok=True)
            
            self.logger.info(f"Generating RGB/HSV dataset in {dataset_dir}")
            
            for family, base_h in self.color_families.items():
                folder = os.path.join(dataset_dir, family)
                os.makedirs(folder, exist_ok=True)
                
                self.logger.info(f"Generating {samples_per_family} samples for {family}...")
                
                for i in range(samples_per_family):
                    # Small hue jitter to stay within color family
                    h_jitter = random.uniform(-8, 8)
                    
                    # Saturation and value variations
                    s_values = np.linspace(0.4, 1.0, num=5)  # Skip very low saturation
                    v_values = np.linspace(0.3, 1.0, num=5)  # Skip too dark
                    
                    s = random.choice(s_values)
                    v = random.choice(v_values)
                    
                    # Convert HSV to RGB
                    rgb = self._hsv_to_rgb_uint8(base_h + h_jitter, s, v)
                    
                    # Create image with color and subtle noise
                    base = np.ones((img_size[1], img_size[0], 3), dtype=np.int16) * np.array(rgb, dtype=np.int16)
                    noise = np.random.randint(-8, 9, (img_size[1], img_size[0], 3), dtype=np.int16)
                    img_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
                    
                    # Save image
                    Image.fromarray(img_arr).save(os.path.join(folder, f"{family}_{i}.png"))
                
                self.logger.info(f"Completed {family} dataset")
            
            self.logger.info(f"RGB/HSV dataset created successfully at {dataset_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating RGB/HSV dataset: {e}")
            return False
    
    def generate_varied_dataset(self,
                               output_dir: str = "color_dataset_varied",
                               samples_per_family: int = 1000,
                               img_size: Tuple[int, int] = (64, 64)) -> bool:
        """
        Generate varied color dataset with textures, gradients, and patterns.
        
        Args:
            output_dir: Output directory name
            samples_per_family: Number of samples per color family
            img_size: Image dimensions (width, height)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            dataset_dir = os.path.join(self.base_dir, output_dir)
            os.makedirs(dataset_dir, exist_ok=True)
            
            self.logger.info(f"Generating varied dataset in {dataset_dir}")
            
            # Pattern generators
            generators = [
                self._generate_gradient,
                self._generate_stripes,
                self._generate_blobs,
                self._generate_texture
            ]
            
            for family, base_h in self.color_families.items():
                folder = os.path.join(dataset_dir, family)
                os.makedirs(folder, exist_ok=True)
                
                self.logger.info(f"Generating {samples_per_family} varied samples for {family}...")
                
                for i in range(samples_per_family):
                    # Hue variation
                    hue_jitter = random.uniform(-8, 8)
                    s = random.uniform(0.5, 1.0)
                    v = random.uniform(0.4, 1.0)
                    
                    # Randomly select pattern generator
                    gen = random.choice(generators)
                    img = gen(base_h + hue_jitter, s, v, img_size)
                    
                    # Save image
                    Image.fromarray(img.astype(np.uint8)).save(
                        os.path.join(folder, f"{family}_{i}.png")
                    )
                
                self.logger.info(f"Completed {family} varied dataset")
            
            self.logger.info(f"Varied dataset created successfully at {dataset_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating varied dataset: {e}")
            return False
    
    def _hsv_to_rgb_uint8(self, h_deg: float, s: float, v: float) -> Tuple[int, int, int]:
        """
        Convert HSV to RGB (uint8).
        
        Args:
            h_deg: Hue in degrees (0-360)
            s: Saturation (0-1)
            v: Value (0-1)
            
        Returns:
            RGB tuple (0-255)
        """
        h = (h_deg % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
    
    def _generate_gradient(self, hue_deg: float, s: float, v: float, 
                          img_size: Tuple[int, int]) -> np.ndarray:
        """Generate linear gradient image."""
        base = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        
        for x in range(img_size[0]):
            t = x / (img_size[0] - 1)
            # Vary saturation and value across gradient
            s_x = np.clip(s * (1 - 0.2 * (0.5 - t)), 0, 1)
            v_x = np.clip(v * (0.6 + 0.4 * t), 0, 1)
            rgb = self._hsv_to_rgb_uint8(hue_deg + random.uniform(-4, 4), s_x, v_x)
            base[:, x, :] = rgb
        
        # Add subtle noise
        noise = np.random.randint(-10, 11, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def _generate_stripes(self, hue_deg: float, s: float, v: float,
                         img_size: Tuple[int, int]) -> np.ndarray:
        """Generate striped pattern image."""
        base = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        n_stripes = random.randint(3, 7)
        stripe_width = img_size[0] // n_stripes
        
        for i in range(n_stripes):
            rgb = self._hsv_to_rgb_uint8(
                hue_deg + random.uniform(-6, 6),
                np.clip(s * random.uniform(0.7, 1.0), 0, 1),
                np.clip(v * random.uniform(0.6, 1.0), 0, 1)
            )
            x0 = i * stripe_width
            x1 = img_size[0] if i == n_stripes - 1 else x0 + stripe_width
            base[:, x0:x1, :] = rgb
        
        # Add noise
        noise = np.random.randint(-12, 13, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def _generate_blobs(self, hue_deg: float, s: float, v: float,
                       img_size: Tuple[int, int]) -> np.ndarray:
        """Generate blob pattern image using PIL."""
        # Base color
        base_rgb = self._hsv_to_rgb_uint8(hue_deg + random.uniform(-3, 3), s, v)
        img = Image.new('RGB', img_size, base_rgb)
        draw = ImageDraw.Draw(img)
        
        # Draw random blobs
        num_blobs = random.randint(5, 12)
        for _ in range(num_blobs):
            rx = random.randint(5, img_size[0] // 2)
            ry = random.randint(5, img_size[1] // 2)
            x = random.randint(0, img_size[0] - rx)
            y = random.randint(0, img_size[1] - ry)
            
            hue_j = hue_deg + random.uniform(-10, 10)
            rgb = self._hsv_to_rgb_uint8(
                hue_j,
                np.clip(s * random.uniform(0.6, 1.0), 0, 1),
                np.clip(v * random.uniform(0.5, 1.0), 0, 1)
            )
            draw.ellipse([x, y, x + rx, y + ry], fill=rgb)
        
        # Apply blur for smooth transitions
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        
        # Convert to array and add noise
        arr = np.array(img)
        noise = np.random.randint(-8, 9, arr.shape, dtype=np.int16)
        img_arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img_arr
    
    def _generate_texture(self, hue_deg: float, s: float, v: float,
                         img_size: Tuple[int, int]) -> np.ndarray:
        """Generate textured image with layered noise."""
        base = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        
        # Layer multiple noise patterns
        for _ in range(4):
            rgb = self._hsv_to_rgb_uint8(
                hue_deg + random.uniform(-5, 5),
                np.clip(s * random.uniform(0.6, 1.0), 0, 1),
                np.clip(v * random.uniform(0.5, 1.0), 0, 1)
            )
            
            # Create noise layer
            layer = np.random.randint(0, 255, (img_size[1], img_size[0], 1), dtype=np.uint8)
            layer = np.repeat(layer, 3, axis=2)
            
            # Blend with base
            blend = (layer.astype(np.int16) * np.array(rgb) // 255)
            base = np.clip(base.astype(np.int16) + blend, 0, 255).astype(np.uint8)
        
        # Final noise
        noise = np.random.randint(-15, 16, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def load_dataset(self, dataset_dir: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load dataset from directory.
        
        Args:
            dataset_dir: Directory containing color family folders
            
        Returns:
            Tuple of (pixel_array, labels)
        """
        try:
            dataset_pixels = []
            labels = []
            
            full_path = os.path.join(self.base_dir, dataset_dir)
            
            if not os.path.exists(full_path):
                self.logger.error(f"Dataset directory not found: {full_path}")
                return np.array([]), []
            
            # Get color families
            color_folders = [f for f in os.listdir(full_path) 
                           if os.path.isdir(os.path.join(full_path, f))]
            
            for color_name in color_folders:
                folder_path = os.path.join(full_path, color_name)
                
                self.logger.info(f"Loading {color_name} images...")
                
                for file in os.listdir(folder_path):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, file)
                        try:
                            img = Image.open(img_path).convert("RGB").resize((32, 32))
                            pixels = np.array(img).reshape(-1, 3)
                            dataset_pixels.append(pixels)
                            labels.extend([color_name] * len(pixels))
                        except Exception as e:
                            self.logger.warning(f"Error loading image {img_path}: {e}")
            
            if dataset_pixels:
                dataset_pixels = np.vstack(dataset_pixels)
                self.logger.info(f"Dataset loaded: {dataset_pixels.shape}")
                return dataset_pixels, labels
            else:
                self.logger.warning("No dataset loaded")
                return np.array([]), []
                
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return np.array([]), []
    
    def get_dataset_stats(self, dataset_dir: str) -> Dict[str, any]:
        """
        Get statistics about a dataset.
        
        Args:
            dataset_dir: Directory containing dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            full_path = os.path.join(self.base_dir, dataset_dir)
            
            if not os.path.exists(full_path):
                return {'error': 'Dataset not found'}
            
            stats = {
                'total_images': 0,
                'color_families': {},
                'total_pixels': 0
            }
            
            color_folders = [f for f in os.listdir(full_path) 
                           if os.path.isdir(os.path.join(full_path, f))]
            
            for color_name in color_folders:
                folder_path = os.path.join(full_path, color_name)
                num_images = len([f for f in os.listdir(folder_path) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                stats['color_families'][color_name] = num_images
                stats['total_images'] += num_images
            
            # Estimate total pixels (assuming 64x64 images)
            stats['total_pixels'] = stats['total_images'] * 64 * 64
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting dataset stats: {e}")
            return {'error': str(e)}


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Dataset Generator...")
    
    generator = ColorDatasetGenerator(base_dir="test_datasets")
    
    # Generate small test datasets
    print("\n1. Generating RGB/HSV dataset (10 samples per family)...")
    success = generator.generate_rgb_hsv_dataset(
        output_dir="test_rgb_hsv",
        samples_per_family=10,
        img_size=(32, 32)
    )
    print(f"RGB/HSV dataset generation: {'Success' if success else 'Failed'}")
    
    print("\n2. Generating varied dataset (10 samples per family)...")
    success = generator.generate_varied_dataset(
        output_dir="test_varied",
        samples_per_family=10,
        img_size=(32, 32)
    )
    print(f"Varied dataset generation: {'Success' if success else 'Failed'}")
    
    print("\n3. Getting dataset statistics...")
    stats = generator.get_dataset_stats("test_rgb_hsv")
    print(f"Dataset stats: {stats}")
    
    print("\nDataset Generator test completed!")
