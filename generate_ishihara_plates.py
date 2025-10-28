"""
Generate placeholder Ishihara test plates (plates 9-38).
These are simple placeholder images with circles and numbers.
Replace with actual licensed Ishihara plates for production use.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Plate configurations from backend/app/ishihara/test.py
PLATES_CONFIG = [
    {"num": 9, "normal": "74", "protan": "21", "deutan": "21"},
    {"num": 10, "normal": "6", "protan": "None", "deutan": "None"},
    {"num": 11, "normal": "5", "protan": "None", "deutan": "None"},
    {"num": 12, "normal": "3", "protan": "None", "deutan": "None"},
    {"num": 13, "normal": "15", "protan": "17", "deutan": "17"},
    {"num": 14, "normal": "74", "protan": "21", "deutan": "21"},
    {"num": 15, "normal": "2", "protan": "None", "deutan": "None"},
    {"num": 16, "normal": "6", "protan": "None", "deutan": "None"},
    {"num": 17, "normal": "16", "protan": "None", "deutan": "None"},
    # Transformation plates
    {"num": 18, "normal": "26", "protan": "6", "deutan": "2"},
    {"num": 19, "normal": "42", "protan": "2", "deutan": "4"},
    {"num": 20, "normal": "35", "protan": "5", "deutan": "3"},
    {"num": 21, "normal": "96", "protan": "6", "deutan": "9"},
    # Hidden digit plates
    {"num": 22, "normal": "None", "protan": "5", "deutan": "None"},
    {"num": 23, "normal": "None", "protan": "45", "deutan": "None"},
    {"num": 24, "normal": "None", "protan": "None", "deutan": "5"},
    {"num": 25, "normal": "None", "protan": "None", "deutan": "45"},
    # Classification plates
    {"num": 26, "normal": "Purple/Red line", "protan": "Purple line", "deutan": "Red line"},
    {"num": 27, "normal": "Purple/Red line", "protan": "Purple line", "deutan": "Red line"},
    {"num": 28, "normal": "Purple/Red line", "protan": "Purple line", "deutan": "Red line"},
    {"num": 29, "normal": "Purple/Red line", "protan": "Purple line", "deutan": "Red line"},
    # Additional vanishing plates
    {"num": 30, "normal": "97", "protan": "None", "deutan": "None"},
    {"num": 31, "normal": "45", "protan": "None", "deutan": "None"},
    {"num": 32, "normal": "5", "protan": "None", "deutan": "None"},
    {"num": 33, "normal": "7", "protan": "None", "deutan": "None"},
    # Additional transformation plates
    {"num": 34, "normal": "16", "protan": "6", "deutan": "1"},
    {"num": 35, "normal": "73", "protan": "3", "deutan": "7"},
    # Additional control plates
    {"num": 36, "normal": "12", "protan": "12", "deutan": "12"},
    {"num": 37, "normal": "8", "protan": "8", "deutan": "8"},
    {"num": 38, "normal": "29", "protan": "29", "deutan": "29"},
]

def create_placeholder_plate(plate_num, normal_text, plate_type="standard"):
    """Create a placeholder Ishihara-style plate image."""
    # Create a 600x600 image with light background
    img = Image.new('RGB', (600, 600), color=(240, 240, 230))
    draw = ImageDraw.Draw(img)
    
    # Add random colored circles to simulate Ishihara style
    np.random.seed(plate_num * 42)  # Deterministic randomness
    
    # Background circles
    for _ in range(300):
        x = np.random.randint(0, 600)
        y = np.random.randint(0, 600)
        r = np.random.randint(8, 20)
        
        # Random pastel colors
        color = (
            np.random.randint(100, 255),
            np.random.randint(100, 255),
            np.random.randint(100, 255)
        )
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=color)
    
    # Add the main number/pattern (what normal vision sees)
    try:
        # Try to use a large font
        font = ImageFont.truetype("arial.ttf", 200)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Display the normal answer in the center
    if normal_text and normal_text != "None":
        # For line/pattern plates, show simplified text
        if "line" in normal_text.lower():
            display_text = f"{plate_num}"
        else:
            display_text = normal_text
        
        # Calculate text position to center it
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (600 - text_width) // 2
        y = (600 - text_height) // 2
        
        # Draw text with contrasting color (darker for visibility)
        draw.text((x, y), display_text, fill=(80, 120, 80), font=font)
    
    # Add plate number in corner
    try:
        small_font = ImageFont.truetype("arial.ttf", 24)
    except:
        small_font = ImageFont.load_default()
    
    draw.text((10, 10), f"Plate {plate_num}", fill=(60, 60, 60), font=small_font)
    draw.text((10, 570), "PLACEHOLDER - Replace with licensed plate", fill=(180, 60, 60), font=small_font)
    
    return img

def main():
    """Generate all missing Ishihara plates."""
    output_dir = "backend/static/ishihara"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating placeholder Ishihara plates 9-38...")
    print("⚠️  These are PLACEHOLDERS. Replace with actual licensed Ishihara plates for clinical use.\n")
    
    for config in PLATES_CONFIG:
        plate_num = config["num"]
        normal_text = config["normal"]
        
        filename = f"plate_{plate_num:02d}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Check if plate already exists
        if os.path.exists(filepath):
            print(f"✓ Plate {plate_num} already exists, skipping...")
            continue
        
        # Determine plate type
        if "line" in normal_text.lower():
            plate_type = "classification"
        elif normal_text == "None":
            plate_type = "hidden"
        else:
            plate_type = "standard"
        
        # Generate the plate
        img = create_placeholder_plate(plate_num, normal_text, plate_type)
        img.save(filepath)
        print(f"✓ Created {filename} (Normal: {normal_text})")
    
    print(f"\n✅ All 38 Ishihara plates are now available in {output_dir}/")
    print("\n📋 Next steps:")
    print("1. Replace placeholder images with actual licensed Ishihara plates")
    print("2. Ensure image filenames match: plate_01.png through plate_38.png")
    print("3. Test with: GET http://localhost:8000/ishihara/plates?mode=comprehensive")

if __name__ == "__main__":
    main()
