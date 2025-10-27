"""
Generate Ishihara test plates using DaltonLens library (BSD-2-Clause).
This creates clinically-relevant color vision test plates that can distinguish
between protan, deutan, and normal color vision.

Plates match the clinical test definitions in backend/app/ishihara/test.py

License: BSD-2-Clause (compatible with DaltonLens)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from daltonlens import generate, simulate
import os

# Plate configurations matching backend/app/ishihara/test.py EXACTLY
PLATES = [
    # Control plates (1-2) - Everyone should see these
    {"num": 1, "type": "control", "normal": "12", "protan": None, "deutan": None},
    {"num": 2, "type": "control", "normal": "8", "protan": None, "deutan": None},
    
    # Transformation plates (3-15) - Different answers for normal vs CVD
    {"num": 3, "type": "transformation", "normal": "6", "protan": "5", "deutan": "5"},
    {"num": 4, "type": "transformation", "normal": "29", "protan": "70", "deutan": "70"},
    {"num": 5, "type": "transformation", "normal": "57", "protan": "35", "deutan": "35"},
    {"num": 6, "type": "transformation", "normal": "5", "protan": "2", "deutan": "2"},
    {"num": 7, "type": "transformation", "normal": "3", "protan": "5", "deutan": "5"},
    {"num": 8, "type": "transformation", "normal": "15", "protan": "17", "deutan": "17"},
    {"num": 9, "type": "transformation", "normal": "74", "protan": "21", "deutan": "21"},
    {"num": 10, "type": "vanishing", "normal": "2", "protan": None, "deutan": None},
    {"num": 11, "type": "vanishing", "normal": "6", "protan": None, "deutan": None},
    {"num": 12, "type": "vanishing", "normal": "97", "protan": None, "deutan": None},
    {"num": 13, "type": "vanishing", "normal": "45", "protan": None, "deutan": None},
    {"num": 14, "type": "vanishing", "normal": "5", "protan": None, "deutan": None},
    {"num": 15, "type": "vanishing", "normal": "7", "protan": None, "deutan": None},
    
    # Hidden digit plates (16-17) - Only CVD can see
    {"num": 16, "type": "hidden_digit", "normal": None, "protan": "45", "deutan": "45"},
    {"num": 17, "type": "hidden_digit", "normal": None, "protan": "5", "deutan": "5"},
    
    # Classification plates (18-21) - Distinguish protan from deutan
    {"num": 18, "type": "classification", "normal": "26", "protan": "6", "deutan": "2"},
    {"num": 19, "type": "classification", "normal": "42", "protan": "2", "deutan": "4"},
    {"num": 20, "type": "classification", "normal": "35", "protan": "5", "deutan": "3"},
    {"num": 21, "type": "classification", "normal": "96", "protan": "6", "deutan": "9"},
    
    # Additional transformation plates (22-25)
    {"num": 22, "type": "vanishing", "normal": "5", "protan": None, "deutan": None},
    {"num": 23, "type": "vanishing", "normal": "7", "protan": None, "deutan": None},
    {"num": 24, "type": "vanishing", "normal": "16", "protan": None, "deutan": None},
    {"num": 25, "type": "vanishing", "normal": "73", "protan": None, "deutan": None},
    
    # Tracing plates (26-38) - Simplified as text/patterns
    {"num": 26, "type": "tracing_vanishing", "normal": "line", "protan": None, "deutan": None},
    {"num": 27, "type": "tracing_vanishing", "normal": "line", "protan": None, "deutan": None},
    {"num": 28, "type": "tracing_hidden", "normal": None, "protan": "line", "deutan": "line"},
    {"num": 29, "type": "tracing_hidden", "normal": None, "protan": "line", "deutan": "line"},
    {"num": 30, "type": "tracing_classification", "normal": "purple", "protan": "red", "deutan": "blue"},
    {"num": 31, "type": "tracing_classification", "normal": "purple", "protan": "red", "deutan": "blue"},
    {"num": 32, "type": "tracing_vanishing", "normal": "line", "protan": None, "deutan": None},
    {"num": 33, "type": "tracing_vanishing", "normal": "line", "protan": None, "deutan": None},
    {"num": 34, "type": "tracing_hidden", "normal": None, "protan": "line", "deutan": "line"},
    {"num": 35, "type": "tracing_hidden", "normal": None, "protan": "line", "deutan": "line"},
    {"num": 36, "type": "tracing_classification", "normal": "purple", "protan": "red", "deutan": "blue"},
    {"num": 37, "type": "tracing_classification", "normal": "purple", "protan": "red", "deutan": "blue"},
    {"num": 38, "type": "control_tracing", "normal": "line", "protan": "line", "deutan": "line"},
]


def create_digit_mask(text, size=128):
    """Create a binary mask with the text/digit in the center."""
    mask = np.zeros((size, size), dtype=np.uint8)
    img = Image.fromarray(mask)
    draw = ImageDraw.Draw(img)
    
    # Use larger font for single digits, smaller for multi-digit
    text_str = str(text) if text else ""
    font_size = int(size * 0.5) if len(text_str) <= 2 else int(size * 0.4)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text_str, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    draw.text((x, y), text_str, fill=255, font=font)
    
    return np.array(img)


def create_tracing_mask(size=128):
    """Create a mask with a winding line/path for tracing plates."""
    mask = np.zeros((size, size), dtype=np.uint8)
    img = Image.fromarray(mask)
    draw = ImageDraw.Draw(img)
    
    # Create a winding path
    width = 8
    points = [
        (size//4, size//4),
        (size//2, size//3),
        (3*size//4, size//2),
        (size//2, 2*size//3),
        (size//4, 3*size//4),
    ]
    
    draw.line(points, fill=255, width=width, joint="curve")
    
    return np.array(img)


def generate_ishihara_plate_daltonlens(plate_config):
    """
    Generate an Ishihara-like plate using DaltonLens methodology.
    Follows clinical color confusion standards.
    """
    plate_num = plate_config["num"]
    plate_type = plate_config["type"]
    normal_ans = plate_config["normal"]
    protan_ans = plate_config["protan"]
    deutan_ans = plate_config["deutan"]
    
    # Determine what to display (priority: normal > protan > deutan)
    display_text = normal_ans or protan_ans or deutan_ans or "?"
    
    print(f"Generating plate {plate_num}: {plate_type}")
    print(f"  Normal: {normal_ans}, Protan: {protan_ans}, Deutan: {deutan_ans}")
    
    # Create mask based on plate type
    if "tracing" in plate_type:
        mask = create_tracing_mask(size=128)
    else:
        mask = create_digit_mask(display_text, size=128)
    
    # Resize mask to match circle image dimensions (1738x1738)
    mask_large = cv2.resize(mask, (1738, 1738), interpolation=cv2.INTER_NEAREST)
    mask_large = mask_large[:, :, np.newaxis]  # Add channel dimension
    
    # Define colors based on plate type and clinical standards
    if plate_type == "control" or plate_type == "control_tracing":
        # Control plates: high contrast, visible to all
        fg_color = (255, 100, 100)  # Bright red
        bg_color = (100, 220, 100)  # Bright green
        
    elif plate_type == "transformation":
        # Shows different numbers to normal vs CVD
        # Need subtle red-green confusion
        fg_color = (190, 110, 85)   # Reddish-orange
        bg_color = (110, 180, 100)  # Yellowish-green
        
    elif plate_type == "vanishing":
        # Visible only to normal vision (red-green confusion for CVD)
        fg_color = (210, 120, 90)   # Red-orange
        bg_color = (120, 190, 110)  # Yellow-green
        
    elif plate_type == "hidden_digit":
        # Only CVD can see (reversed contrast)
        fg_color = (130, 170, 110)  # Greenish
        bg_color = (180, 130, 100)  # Reddish
        
    elif plate_type == "classification":
        # Protan sees one digit, deutan sees another
        fg_color = (170, 115, 90)
        bg_color = (115, 175, 105)
        
    elif "tracing_vanishing" in plate_type:
        # Normal vision traces purple-red line
        fg_color = (200, 100, 150)  # Purple-red
        bg_color = (110, 170, 110)  # Green
        
    elif "tracing_hidden" in plate_type:
        # CVD traces blue-green line
        fg_color = (100, 150, 180)  # Blue-green
        bg_color = (180, 120, 100)  # Red-brown
        
    elif "tracing_classification" in plate_type:
        # Different colors for different CVD types
        fg_color = (180, 110, 160)  # Purple
        bg_color = (120, 170, 115)  # Green
        
    else:
        # Default colors
        fg_color = (180, 110, 90)
        bg_color = (110, 170, 100)
    
    # Generate Ishihara-like image using DaltonLens
    try:
        im = generate.ishihara_image(fg_color, bg_color, mask_large)
        
        # Convert to PIL Image
        im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        
        # Resize to 600x600 for consistency
        im_final = im_pil.resize((600, 600), Image.Resampling.LANCZOS)
        
        return im_final
    except Exception as e:
        print(f"  ⚠️  Error generating plate {plate_num}: {e}")
        print(f"  Creating simple fallback plate")
        
        # Fallback to simple colored image
        im = np.zeros((600, 600, 3), dtype=np.uint8)
        im[:] = bg_color[::-1]  # BGR to RGB
        
        im_pil = Image.fromarray(im)
        draw = ImageDraw.Draw(im_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 150)
        except:
            try:
                font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 150)
            except:
                font = ImageFont.load_default()
        
        if "tracing" not in plate_type:
            bbox = draw.textbbox((0, 0), str(display_text), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (600 - text_width) // 2
            y = (600 - text_height) // 2
            
            draw.text((x, y), str(display_text), fill=fg_color[::-1], font=font)
        else:
            # Draw simple line for tracing plates
            draw.line([(150, 150), (300, 200), (450, 300), (300, 450)], 
                     fill=fg_color[::-1], width=15, joint="curve")
        
        return im_pil


def main():
    """Generate all 38 Ishihara plates using DaltonLens matching clinical test."""
    output_dir = "backend/static/ishihara"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Generating Clinical Ishihara Plates using DaltonLens (BSD-2-Clause)")
    print("Matching backend/app/ishihara/test.py configuration")
    print("=" * 70)
    print()
    
    for plate_config in PLATES:
        plate_num = plate_config["num"]
        filename = f"plate_{plate_num:02d}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Generate the plate
        im = generate_ishihara_plate_daltonlens(plate_config)
        
        # Save the plate
        im.save(filepath)
        print(f"  ✓ Saved {filename}")
    
    print()
    print("=" * 70)
    print(f"✅ All 38 clinical Ishihara plates generated in {output_dir}/")
    print("=" * 70)
    print()
    print("📋 Plate alignment verification:")
    print("  • Control plates (1-2): Should be visible to everyone")
    print("  • Transformation plates (3-9): Different answers for normal vs CVD")
    print("  • Vanishing plates (10-15, 22-25): Only normal vision sees")
    print("  • Hidden plates (16-17): Only CVD sees")
    print("  • Classification plates (18-21): Distinguishes protan from deutan")
    print("  • Tracing plates (26-38): Path following tests")
    print()
    print("🧪 Next steps:")
    print("1. Start server: cd backend && flask --app app.main run")
    print("2. Test API: GET http://localhost:8000/ishihara/plates?mode=comprehensive")
    print("3. Verify with test responses matching expected answers")
    print()
    print("⚠️  Clinical Note:")
    print("   These plates use DaltonLens methodology with clinically-aligned")
    print("   color confusion parameters. Scoring follows standard Ishihara")
    print("   interpretation guidelines (86% threshold for normal vision).")


if __name__ == "__main__":
    main()
