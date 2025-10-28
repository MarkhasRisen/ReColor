# Ishihara Test Plate Setup Guide

## 📋 What You Need

To complete the calibration feature, you need **Ishihara color blindness test plate images**.

## 🎯 Image Requirements

### File Format:
- **Format**: PNG or JPG
- **Resolution**: 500x500 to 1000x1000 pixels recommended
- **Quality**: High resolution, clear circles

### Naming Convention:
Save images with this pattern in `backend/static/ishihara/`:
```
plate_01.png  → First plate (usually "12")
plate_02.png  → Second plate
plate_03.png  → Third plate
...
plate_08.png  → Eighth plate
```

## 📥 Where to Get Images

### Option 1: Purchase Official Plates
- **Ishihara's Tests for Colour Deficiency** (official book)
- Available from medical supply companies
- Most legally compliant option

### Option 2: Educational Resources
- Public domain medical education materials
- University ophthalmology departments
- Open educational resources (OER)

### Option 3: Create Synthetic Plates
Generate simplified test patterns programmatically:
```python
# Example: Generate a simple test pattern
from PIL import Image, ImageDraw
import random

def generate_test_plate(number_to_show, background_color, number_color, filename):
    """Generate a simplified Ishihara-style test plate."""
    img = Image.new('RGB', (800, 800), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw random circles (background)
    for _ in range(200):
        x = random.randint(0, 800)
        y = random.randint(0, 800)
        r = random.randint(10, 30)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=background_color)
    
    # Draw number shape (simplified)
    # ... (implement number drawing logic)
    
    img.save(f"backend/static/ishihara/{filename}")

# Example usage:
generate_test_plate("12", (255, 200, 200), (200, 100, 100), "plate_01.png")
```

## ⚖️ Legal Considerations

⚠️ **IMPORTANT**: Ishihara test plates are copyrighted material.

- ✅ **Legal**: Using purchased official plates for personal/educational use
- ✅ **Legal**: Creating your own synthetic patterns
- ✅ **Legal**: Using public domain alternatives
- ❌ **Illegal**: Copying copyrighted plates without permission for commercial use

### Recommended Approach:
1. **For Development/Testing**: Use placeholder images or synthetic patterns
2. **For Production**: Purchase official plates or license from copyright holder
3. **For Research**: Check if your institution has licensed access

## 🔧 Current Setup

### API Endpoint:
```
GET /static/ishihara/<filename>
```

Example:
```
http://192.168.1.9:8000/static/ishihara/plate_01.png
```

### Calibration Response:
The API currently returns plate data like:
```json
{
  "plates": [
    {
      "id": "p1",
      "image_url": "/static/ishihara/plate_01.png",
      "correct_answer": "12",
      "description": "Plate 1"
    }
  ]
}
```

## 🚀 Quick Start

### 1. Add Placeholder Images (For Testing)

Create simple colored circles as placeholders:
```python
from PIL import Image, ImageDraw, ImageFont

for i in range(1, 9):
    img = Image.new('RGB', (600, 600), 'lightgray')
    draw = ImageDraw.Draw(img)
    
    # Draw circle background
    for _ in range(150):
        x = random.randint(50, 550)
        y = random.randint(50, 550)
        r = random.randint(15, 25)
        color = (random.randint(100, 255), random.randint(100, 200), random.randint(100, 150))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    # Add text "PLATE {i}"
    text = f"PLATE {i}"
    draw.text((250, 280), text, fill='black')
    
    img.save(f"backend/static/ishihara/plate_{i:02d}.png")
    print(f"Created plate_{i:02d}.png")
```

### 2. Update Calibration Endpoint

Modify `backend/app/routes/calibration.py` to return image URLs:
```python
plates_data = [
    {"id": "p1", "image_url": "/static/ishihara/plate_01.png", "correct": "12"},
    {"id": "p2", "image_url": "/static/ishihara/plate_02.png", "correct": "8"},
    # ... more plates
]
```

### 3. Test in Web Interface

Open `test_web_app.html` and verify images load in calibration tab.

## 📊 Standard Ishihara Plates

Traditional Ishihara test includes:

| Plate | Visible to All | Protan/Deutan | Tritan | Purpose |
|-------|---------------|---------------|--------|---------|
| 1     | 12            | 12            | 12     | Control |
| 2     | 8             | 3 (weak)      | 8      | Red-Green |
| 3     | 6             | 5 (weak)      | 6      | Red-Green |
| 4     | 29            | 70            | 29     | Red-Green |
| 5     | 57            | 35            | 57     | Red-Green |
| 6     | 5             | 2             | 5      | Red-Green |
| 7     | 3             | 5             | 3      | Red-Green |
| 8     | 15            | 17            | 15     | Red-Green |

## 🎨 Next Steps After Adding Images

1. **Update calibration endpoint** to return actual image URLs
2. **Test in web interface** - verify images display correctly
3. **Update mobile app** - ensure images load on mobile devices
4. **Configure correct answers** - match plate numbers to expected responses
5. **Add image optimization** - compress images for mobile performance

## 📝 Files to Update

After adding images:
- ✅ `backend/static/ishihara/*.png` - Plate images (you add these)
- ⏳ `backend/app/routes/calibration.py` - Return image URLs in response
- ⏳ `test_web_app.html` - Display images instead of text
- ⏳ `mobile/src/screens/Calibration.tsx` - Load and display images

## 💡 Pro Tips

1. **Start small**: Begin with 3-4 test plates, expand later
2. **Use CDN**: For production, host images on CDN for faster loading
3. **Lazy loading**: Load images on-demand to reduce initial load time
4. **Fallback text**: Keep text descriptions for accessibility
5. **Cache images**: Configure proper cache headers for repeated tests

---

**Ready to add images?** Place your PNG/JPG files in:
```
C:\Users\markr\Downloads\Daltonization\backend\static\ishihara\
```

Then test with:
```
http://192.168.1.9:8000/static/ishihara/plate_01.png
```
