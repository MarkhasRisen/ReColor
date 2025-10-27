"""Generate varied color datasets for K-Means training and CNN preprocessing.

This script creates synthetic images with controlled color variations to train
the adaptive color correction pipeline. Uses HSV for generation but stores RGB.
"""
import os
import random
import colorsys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


# Color family definitions (hue in degrees)
COLOR_FAMILIES = {
    "red": 0,
    "orange": 30,
    "yellow": 60,
    "green": 120,
    "cyan": 180,
    "blue": 240,
    "indigo": 275,
    "violet": 300,
}

# Dataset generation parameters
IMG_SIZE = (64, 64)
SAMPLES_PER_FAMILY = 1000
HUE_JITTER = 6.0  # degrees
SATURATION_RANGE = (0.6, 1.0)
VALUE_RANGE = (0.5, 1.0)


def hsv_to_rgb_uint8(h_deg: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV to RGB uint8 values."""
    h = (h_deg % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def generate_gradient(hue_deg: float, s: float, v: float) -> np.ndarray:
    """Create linear gradient with subtle hue & saturation variation."""
    base = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    for x in range(IMG_SIZE[0]):
        t = x / (IMG_SIZE[0] - 1)
        s_x = np.clip(s * (1 - 0.2 * (0.5 - t)), 0, 1)
        v_x = np.clip(v * (0.6 + 0.4 * t), 0, 1)
        rgb = hsv_to_rgb_uint8(hue_deg + random.uniform(-4, 4), s_x, v_x)
        base[:, x, :] = rgb
    
    # Add subtle per-pixel noise
    noise = np.random.randint(-8, 9, base.shape, dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_stripes(hue_deg: float, s: float, v: float) -> np.ndarray:
    """Create vertical stripes with color variation."""
    base = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    n_stripes = random.randint(2, 6)
    stripe_width = IMG_SIZE[0] // n_stripes
    
    for i in range(n_stripes):
        rgb = hsv_to_rgb_uint8(
            hue_deg + random.uniform(-6, 6),
            np.clip(s * random.uniform(0.7, 1.0), 0, 1),
            np.clip(v * random.uniform(0.6, 1.0), 0, 1)
        )
        x0 = i * stripe_width
        x1 = IMG_SIZE[0] if i == n_stripes - 1 else x0 + stripe_width
        base[:, x0:x1, :] = rgb
    
    noise = np.random.randint(-10, 11, base.shape, dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_blobs(hue_deg: float, s: float, v: float) -> np.ndarray:
    """Create random elliptical blobs with soft transitions."""
    img = Image.new('RGB', IMG_SIZE, hsv_to_rgb_uint8(
        hue_deg + random.uniform(-3, 3), s, v
    ))
    draw = ImageDraw.Draw(img)
    num_blobs = random.randint(3, 10)
    
    for _ in range(num_blobs):
        rx = random.randint(5, IMG_SIZE[0] // 2)
        ry = random.randint(5, IMG_SIZE[1] // 2)
        x = random.randint(0, IMG_SIZE[0] - rx)
        y = random.randint(0, IMG_SIZE[1] - ry)
        hue_j = hue_deg + random.uniform(-8, 8)
        rgb = hsv_to_rgb_uint8(
            hue_j,
            np.clip(s * random.uniform(0.6, 1.0), 0, 1),
            np.clip(v * random.uniform(0.5, 1.0), 0, 1)
        )
        draw.ellipse([x, y, x + rx, y + ry], fill=rgb)
    
    # Apply Gaussian blur for soft transitions
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.8)))
    arr = np.array(img)
    noise = np.random.randint(-7, 8, arr.shape, dtype=np.int16)
    img_arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img_arr


def generate_texture(hue_deg: float, s: float, v: float) -> np.ndarray:
    """Create layered noise texture with dominant hue tint."""
    base = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    
    for _ in range(4):
        rgb = hsv_to_rgb_uint8(
            hue_deg + random.uniform(-5, 5),
            np.clip(s * random.uniform(0.6, 1.0), 0, 1),
            np.clip(v * random.uniform(0.5, 1.0), 0, 1)
        )
        layer = np.random.randint(0, 255, (IMG_SIZE[1], IMG_SIZE[0], 1), dtype=np.uint8)
        layer = np.repeat(layer, 3, axis=2)
        base = np.clip(
            (base.astype(np.int16) + (layer.astype(np.int16) * np.array(rgb) // 255)),
            0, 255
        ).astype(np.uint8)
    
    noise = np.random.randint(-20, 21, base.shape, dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# All generators
GENERATORS = [generate_gradient, generate_stripes, generate_blobs, generate_texture]


def generate_dataset(output_dir: Path, samples_per_family: int = SAMPLES_PER_FAMILY):
    """Generate varied color dataset for training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 Generating dataset with {samples_per_family} samples per family...")
    print(f"📁 Output directory: {output_dir}")
    
    for family, base_h in COLOR_FAMILIES.items():
        folder = output_dir / family
        folder.mkdir(exist_ok=True)
        
        print(f"  Generating {family} family (hue={base_h}°)...")
        for i in tqdm(range(samples_per_family), desc=f"  {family}", ncols=80):
            hue_jitter = random.uniform(-HUE_JITTER, HUE_JITTER)
            s = random.uniform(*SATURATION_RANGE)
            v = random.uniform(*VALUE_RANGE)
            
            gen = random.choice(GENERATORS)
            img = gen(base_h + hue_jitter, s, v)
            
            Image.fromarray(img.astype(np.uint8)).save(folder / f"{family}_{i:04d}.png")
    
    print(f"✅ Dataset generation complete!")
    print(f"   Total images: {len(COLOR_FAMILIES) * samples_per_family}")


if __name__ == "__main__":
    import sys
    
    output_path = Path(__file__).parent.parent / "datasets" / "color_varied"
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    
    generate_dataset(output_path)
