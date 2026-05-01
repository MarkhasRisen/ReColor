#!/usr/bin/env python3
"""
Analyze color gaps in IDENTIFIER_DB using Delta-E distance.
Finds regions in LAB space where nearest color is > ΔE 8 away.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# 1. RGB to LAB conversion (matching tensorHelper.js)
# ─────────────────────────────────────────────────────────────

def srgb_to_linear(c):
    """IEC 61966-2-1 sRGB gamma decoding"""
    return c / 12.92 if c <= 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)

def rgb_to_lab(r, g, b):
    """Convert sRGB [0-255] to CIELAB using D65 illuminant"""
    # sRGB → linear RGB
    rl = srgb_to_linear(r / 255.0)
    gl = srgb_to_linear(g / 255.0)
    bl = srgb_to_linear(b / 255.0)

    # Linear RGB → XYZ (sRGB D65 matrix)
    x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047
    y = (0.2126729 * rl + 0.7151522 * gl + 0.072175 * bl) / 1.0
    z = (0.0193339 * rl + 0.0961964 * gl + 0.9503041 * bl) / 1.08883

    # XYZ → LAB
    def f(t):
        return math.cbrt(t) if t > 0.008856 else 7.787 * t + 16 / 116

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_comp = 200 * (fy - fz)

    return (L, a, b_comp)

def hex_to_rgb(hex_str):
    """Convert hex string to RGB tuple"""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

# ─────────────────────────────────────────────────────────────
# 2. Delta-E calculation (matching tensorHelper.js)
# ─────────────────────────────────────────────────────────────

L_WEIGHT = 0.5  # Lightness down-weighted for color naming

def delta_e(lab1: Tuple[float, float, float], 
            lab2: Tuple[float, float, float]) -> float:
    """Weighted Delta-E distance"""
    return math.sqrt(
        L_WEIGHT * (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    )

# ─────────────────────────────────────────────────────────────
# 3. Color database (77 colors from tensorHelper.js lines 192-295)
# ─────────────────────────────────────────────────────────────

IDENTIFIER_DB = [
    # Neutral
    ("black", "#000000", 0, 0, 0),
    ("dark gray", "#404040", 64, 64, 64),
    ("gray", "#808080", 128, 128, 128),
    ("light gray", "#C0C0C0", 192, 192, 192),
    ("white", "#FFFFFF", 255, 255, 255),

    # Red
    ("Red", "#FF0000", 255, 0, 0),
    ("Red", "#CC0000", 204, 0, 0),
    ("Red", "#8B0000", 139, 0, 0),
    ("Crimson", "#DC143C", 220, 20, 60),
    ("Firebrick", "#B22222", 178, 34, 34),
    ("Red", "#FF3333", 255, 51, 51),
    ("Indian Red", "#CD5C5C", 205, 92, 92),
    ("Dark Muted Red", "#8B3A3A", 139, 58, 58),
    ("Soft Red", "#E06060", 224, 96, 96),

    # Orange
    ("Dark Orange", "#FF8C00", 255, 140, 0),
    ("Orange", "#FFA500", 255, 165, 0),
    ("Coral", "#FF7F50", 255, 127, 80),
    ("Orange", "#E8751A", 232, 117, 26),
    ("Orange", "#CC7000", 204, 112, 0),
    ("Muted Orange", "#C48040", 196, 128, 64),
    ("Peach", "#E0976E", 224, 151, 110),
    ("Dusty Orange", "#B8743A", 184, 116, 58),

    # Yellow
    ("Yellow", "#FFFF00", 255, 255, 0),
    ("Gold", "#FFD700", 255, 215, 0),
    ("Light Goldenrod", "#FFEC8B", 255, 236, 139),
    ("Goldenrod", "#DAA520", 218, 165, 32),
    ("Khaki", "#F0E68C", 240, 230, 140),
    ("Dark Khaki", "#BDB76B", 189, 183, 107),
    ("Muted Yellow", "#D4CC6A", 212, 204, 106),

    # Green
    ("Green", "#008000", 0, 128, 0),
    ("Lime", "#00FF00", 0, 255, 0),
    ("Forest Green", "#228B22", 34, 139, 34),
    ("Dark Green", "#006400", 0, 100, 0),
    ("Lime Green", "#32CD32", 50, 205, 50),
    ("Light Green", "#90EE90", 144, 238, 144),
    ("Olive Drab", "#6B8E23", 107, 142, 35),
    ("Dark Olive Green", "#556B2F", 85, 107, 47),
    ("Dark Sea Green", "#8FBC8F", 143, 188, 143),
    ("Muted Green", "#4A7A4A", 74, 122, 74),

    # Cyan
    ("Cyan", "#00FFFF", 0, 255, 255),
    ("Dark Cyan", "#008B8B", 0, 139, 139),
    ("Light Sea Green", "#20B2AA", 32, 178, 170),
    ("Dark Turquoise", "#00CED1", 0, 206, 209),
    ("Turquoise", "#40E0D0", 64, 224, 208),
    ("Cadet Blue", "#5F9EA0", 95, 158, 160),
    ("Muted Teal", "#6B9B9B", 107, 155, 155),

    # Blue
    ("Blue", "#0000FF", 0, 0, 255),
    ("Navy", "#000080", 0, 0, 128),
    ("Dodger Blue", "#1E90FF", 30, 144, 255),
    ("Royal Blue", "#4169E1", 65, 105, 225),
    ("Sky Blue", "#87CEEB", 135, 206, 235),
    ("Steel Blue", "#4682B4", 70, 130, 180),
    ("Slate Blue", "#6A7B8D", 106, 123, 141),
    ("Denim", "#4A6A8A", 74, 106, 138),
    ("Light Steel Blue", "#B0C4DE", 176, 196, 222),

    # Violet
    ("Violet", "#8B00FF", 139, 0, 255),
    ("Purple", "#800080", 128, 0, 128),
    ("Dark Violet", "#9400D3", 148, 0, 211),
    ("Medium Orchid", "#BA55D3", 186, 85, 211),
    ("Indigo", "#4B0082", 75, 0, 130),
    ("Rebecca Purple", "#663399", 102, 51, 153),
    ("Medium Purple", "#9370DB", 147, 112, 219),
    ("Muted Lavender", "#7B68A5", 123, 104, 165),
    ("Dusty Purple", "#5D4E7A", 93, 78, 122),

    # Pink
    ("Pink", "#FFC0CB", 255, 192, 203),
    ("Hot Pink", "#FF69B4", 255, 105, 180),
    ("Deep Pink", "#FF1493", 255, 20, 147),
    ("Pale Violet Red", "#DB7093", 219, 112, 147),
    ("Light Pink", "#FFB6C1", 255, 182, 193),
    ("Magenta", "#FF00FF", 255, 0, 255),
    ("Dusty Rose", "#C48A9A", 196, 138, 154),
    ("Muted Pink", "#D4A0A0", 212, 160, 160),
    ("Mauve", "#B07080", 176, 112, 128),

    # Brown
    ("Saddle Brown", "#8B4513", 139, 69, 19),
    ("Sienna", "#A0522D", 160, 82, 45),
    ("Chocolate", "#D2691E", 210, 105, 30),
    ("Dark Brown", "#654321", 101, 67, 33),
    ("Brown", "#A52A2A", 165, 42, 42),
    ("Burlywood", "#DEB887", 222, 184, 135),
    ("Muted Tan", "#8B7355", 139, 115, 85),
    ("Muted Brown", "#6B4F3A", 107, 79, 58),
    ("Sand/Beige-Brown", "#C4A882", 196, 168, 130),
    ("Medium Brown", "#806040", 128, 96, 64),
]

# ─────────────────────────────────────────────────────────────
# 4. Convert all database colors to LAB
# ─────────────────────────────────────────────────────────────

colors_lab = []
for name, hex_color, r, g, b in IDENTIFIER_DB:
    lab = rgb_to_lab(r, g, b)
    colors_lab.append({
        'name': name,
        'hex': hex_color,
        'rgb': (r, g, b),
        'lab': lab
    })

print(f"✓ Loaded {len(colors_lab)} colors from IDENTIFIER_DB")

# ─────────────────────────────────────────────────────────────
# 5. Sample LAB space and find gaps
# ─────────────────────────────────────────────────────────────

gaps = []
total_samples = 0

# L*: 20-100 step 5
# a*: -80 to +80 step 10
# b*: -80 to +80 step 10
for L in range(20, 101, 5):
    for a in range(-80, 81, 10):
        for b in range(-80, 81, 10):
            total_samples += 1
            sample_lab = (L, a, b)

            # Find nearest color in database
            min_distance = float('inf')
            nearest_color = None

            for color in colors_lab:
                dist = delta_e(sample_lab, color['lab'])
                if dist < min_distance:
                    min_distance = dist
                    nearest_color = color

            # Record if gap exists (ΔE > 8)
            if min_distance > 8:
                gaps.append({
                    'sample_lab': sample_lab,
                    'delta_e': min_distance,
                    'nearest_color': nearest_color,
                    'distance': min_distance
                })

# Sort by worst coverage (highest ΔE)
gaps.sort(key=lambda x: x['delta_e'], reverse=True)

# ─────────────────────────────────────────────────────────────
# 6. Convert LAB back to RGB for visualization
# ─────────────────────────────────────────────────────────────

def lab_to_rgb(L, a, b):
    """Convert LAB back to sRGB [0-255] for visualization"""
    # LAB → XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def f_inv(t):
        t3 = t ** 3
        return t3 if t3 > 0.008856 else (116 * t - 16) / 7.787

    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.0
    z = f_inv(fz) * 1.08883

    # XYZ → linear RGB
    rl = x * 3.2404542 + y * (-1.5371385) + z * (-0.4985314)
    gl = x * (-0.9692660) + y * 1.8760108 + z * 0.0415560
    bl = x * 0.0556434 + y * (-0.2040259) + z * 1.0572252

    # Linear RGB → sRGB (gamma encoding)
    def linear_to_srgb(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055

    r = max(0, min(255, round(linear_to_srgb(rl) * 255)))
    g = max(0, min(255, round(linear_to_srgb(gl) * 255)))
    b_out = max(0, min(255, round(linear_to_srgb(bl) * 255)))

    return (r, g, b_out)

def rgb_to_hex(r, g, b):
    """Convert RGB to hex string"""
    return f"#{r:02X}{g:02X}{b:02X}"

# ─────────────────────────────────────────────────────────────
# 7. Output results
# ─────────────────────────────────────────────────────────────

print(f"\n{'='*100}")
print(f"COLOR GAP ANALYSIS - IDENTIFIER_DB Coverage")
print(f"{'='*100}\n")

print(f"Sampling parameters:")
print(f"  L* (lightness): 20-100 (step 5)    → {len(range(20, 101, 5))} values")
print(f"  a* (red-green): -80 to +80 (step 10) → {len(range(-80, 81, 10))} values")
print(f"  b* (yellow-blue): -80 to +80 (step 10) → {len(range(-80, 81, 10))} values")
print(f"  Total sample points: {total_samples}")
print(f"  Weighted Delta-E threshold: ΔE > 8 (L_WEIGHT=0.5)")
print(f"\nGAP STATISTICS:")
print(f"  Gaps found (ΔE > 8): {len(gaps)} / {total_samples} ({100*len(gaps)/total_samples:.2f}%)")

if gaps:
    de_values = [g['delta_e'] for g in gaps]
    print(f"  ΔE range: {min(de_values):.2f} - {max(de_values):.2f}")
    print(f"  Mean ΔE: {sum(de_values)/len(de_values):.2f}")
    print(f"  Median ΔE: {sorted(de_values)[len(de_values)//2]:.2f}")

    print(f"\n{'='*100}")
    print(f"TOP 30 WORST-COVERED REGIONS (sorted by ΔE)")
    print(f"{'='*100}\n")

    for i, gap in enumerate(gaps[:30], 1):
        L, a, b = gap['sample_lab']
        rgb = lab_to_rgb(L, a, b)
        hex_color = rgb_to_hex(*rgb)
        nearest = gap['nearest_color']
        de = gap['delta_e']

        print(f"{i:2d}. ΔE = {de:6.2f}")
        print(f"    Sample LAB: L*={L:5.0f} a*={a:+6.0f} b*={b:+6.0f}")
        print(f"    Sample RGB: R={rgb[0]:3d} G={rgb[1]:3d} B={rgb[2]:3d} → {hex_color}")
        print(f"    Nearest: '{nearest['name']}' @ {nearest['hex']}")
        print(f"    Nearest LAB: L*={nearest['lab'][0]:5.1f} a*={nearest['lab'][1]:+6.1f} b*={nearest['lab'][2]:+6.1f}")
        print()
else:
    print("  ✓ No gaps found! Database provides excellent coverage.")

print(f"{'='*100}\n")
