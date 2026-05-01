"""
Render PlantUML diagrams from chapter3_testing_diagrams.md into PNG files
using the PlantUML public server API.
"""

import re, zlib, string, urllib.request, os, ssl

# ── PlantUML encoding ──────────────────────────────────────────────────
PLANTUML_ALPHABET = (
    string.digits
    + string.ascii_uppercase
    + string.ascii_lowercase
    + "-_"
)
BASE64_ALPHABET = (
    string.ascii_uppercase
    + string.ascii_lowercase
    + string.digits
    + "+/"
)

def plantuml_encode(text):
    """Encode text for PlantUML server URL."""
    compressed = zlib.compress(text.encode("utf-8"))[2:-4]  # raw deflate
    result = []
    for i in range(0, len(compressed), 3):
        chunk = compressed[i:i+3]
        if len(chunk) == 3:
            b0, b1, b2 = chunk
            result.append(PLANTUML_ALPHABET[b0 >> 2])
            result.append(PLANTUML_ALPHABET[((b0 & 0x3) << 4) | (b1 >> 4)])
            result.append(PLANTUML_ALPHABET[((b1 & 0xF) << 2) | (b2 >> 6)])
            result.append(PLANTUML_ALPHABET[b2 & 0x3F])
        elif len(chunk) == 2:
            b0, b1 = chunk
            result.append(PLANTUML_ALPHABET[b0 >> 2])
            result.append(PLANTUML_ALPHABET[((b0 & 0x3) << 4) | (b1 >> 4)])
            result.append(PLANTUML_ALPHABET[(b1 & 0xF) << 2])
        elif len(chunk) == 1:
            b0 = chunk[0]
            result.append(PLANTUML_ALPHABET[b0 >> 2])
            result.append(PLANTUML_ALPHABET[(b0 & 0x3) << 4])
    return "".join(result)


def render_diagram(puml_text, output_path):
    """Send PlantUML text to server and save PNG."""
    encoded = plantuml_encode(puml_text)
    url = f"https://www.plantuml.com/plantuml/png/{encoded}"
    
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        data = resp.read()
    
    with open(output_path, "wb") as f:
        f.write(data)
    
    size_kb = len(data) / 1024
    print(f"  [OK] {os.path.basename(output_path)} ({size_kb:.1f} KB)")


# ── Extract diagrams from markdown ─────────────────────────────────────
MD_PATH = r"c:\Users\markr\OneDrive\Desktop\Local Project Filee\ReColor\evaluation\chapter3_testing_diagrams.md"
OUT_DIR = r"c:\Users\markr\OneDrive\Desktop\Local Project Filee\ReColor\evaluation\diagrams"

os.makedirs(OUT_DIR, exist_ok=True)

with open(MD_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Find all ```plantuml ... ``` blocks
pattern = r"```plantuml\s*\n(@startuml\s+(\w+).*?@enduml)"
matches = re.findall(pattern, content, re.DOTALL)

print(f"Found {len(matches)} PlantUML diagrams. Rendering...\n")

for i, (puml_text, diagram_name) in enumerate(matches, 1):
    output_path = os.path.join(OUT_DIR, f"{diagram_name}.png")
    print(f"[{i}/{len(matches)}] Rendering {diagram_name}...")
    try:
        render_diagram(puml_text, output_path)
    except Exception as e:
        print(f"  [FAIL] {e}")

print(f"\nDone. All PNGs saved to: {OUT_DIR}")
