"""
Generate placeholder Ishihara-style test plates for development/testing.

⚠️ IMPORTANT: These are simplified placeholders, NOT medical-grade test plates.
For production, use official Ishihara plates or properly licensed alternatives.
"""
import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("❌ Pillow not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pillow'])
    from PIL import Image, ImageDraw, ImageFont


def generate_placeholder_plate(plate_number: int, text: str, output_dir: Path):
    """Generate a simple placeholder test plate with colored circles."""
    
    # Create blank image
    size = 600
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Color schemes for different plates
    color_schemes = [
        # Background colors (R, G, B ranges)
        [(150, 200), (180, 220), (150, 190)],  # Greenish
        [(180, 220), (150, 200), (150, 190)],  # Reddish
        [(180, 220), (180, 220), (150, 190)],  # Yellowish
        [(150, 200), (150, 200), (180, 220)],  # Bluish
    ]
    
    scheme_idx = (plate_number - 1) % len(color_schemes)
    bg_r_range, bg_g_range, bg_b_range = color_schemes[scheme_idx]
    
    # Draw background circles
    random.seed(plate_number * 42)  # Consistent randomness per plate
    for _ in range(180):
        x = random.randint(50, size - 50)
        y = random.randint(50, size - 50)
        r = random.randint(12, 22)
        
        color = (
            random.randint(*bg_r_range),
            random.randint(*bg_g_range),
            random.randint(*bg_b_range)
        )
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    # Draw number circles (different color for visibility)
    # Simplified: just darker/lighter circles in number shape
    center_x, center_y = size // 2, size // 2
    
    # Draw text overlay for testing
    try:
        # Try to use a larger font
        font = ImageFont.truetype("arial.ttf", 120)
    except:
        font = ImageFont.load_default()
    
    # Draw the number in contrasting color
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2 - 20
    
    # Draw text with slight transparency effect (circles overlay)
    contrast_r = bg_r_range[0] - 40 if bg_r_range[0] > 80 else bg_r_range[1] + 40
    contrast_g = bg_g_range[0] - 40 if bg_g_range[0] > 80 else bg_g_range[1] + 40
    contrast_b = bg_b_range[0] - 40 if bg_b_range[0] > 80 else bg_b_range[1] + 40
    
    draw.text((text_x, text_y), text, fill=(contrast_r, contrast_g, contrast_b), font=font)
    
    # Add label at bottom
    label = f"PLATE {plate_number} - Placeholder"
    try:
        label_font = ImageFont.truetype("arial.ttf", 16)
    except:
        label_font = ImageFont.load_default()
    
    label_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]
    draw.text(((size - label_width) // 2, size - 30), label, fill='gray', font=label_font)
    
    # Save image
    filename = f"plate_{plate_number:02d}.png"
    filepath = output_dir / filename
    img.save(filepath)
    print(f"✅ Created: {filename}")
    
    return filepath


def main():
    """Generate 8 placeholder test plates."""
    
    # Determine output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "backend" / "static" / "ishihara"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating Placeholder Ishihara Test Plates")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Standard Ishihara plate numbers (visible to normal vision)
    plate_data = [
        (1, "12"),   # Control plate
        (2, "8"),    # Red-green test
        (3, "6"),    # Red-green test
        (4, "29"),   # Red-green test
        (5, "57"),   # Red-green test
        (6, "5"),    # Red-green test
        (7, "3"),    # Red-green test
        (8, "15"),   # Red-green test
    ]
    
    for plate_num, text in plate_data:
        generate_placeholder_plate(plate_num, text, output_dir)
    
    print()
    print("=" * 60)
    print("✅ All placeholder plates generated!")
    print()
    print("⚠️  IMPORTANT NOTES:")
    print("   • These are simplified placeholders for development only")
    print("   • NOT suitable for medical diagnosis")
    print("   • For production, use official Ishihara plates")
    print()
    print("🔗 Test images at:")
    print("   http://192.168.1.9:8000/static/ishihara/plate_01.png")
    print("   http://192.168.1.9:8000/static/ishihara/plate_02.png")
    print("   ... etc")


if __name__ == "__main__":
    main()
