"""Render the test case hierarchy as a WBS diagram (PlantUML mind map fallback)."""
import zlib, string, urllib.request, ssl, os

PLANTUML_ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase + "-_"

def plantuml_encode(text):
    compressed = zlib.compress(text.encode("utf-8"))[2:-4]
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

puml = r"""@startwbs test_case_hierarchy
!theme plain
skinparam backgroundColor #FFFFFF

title ReColor Test Case Classification (457 Total)

* **457 Test Cases**
** Part A: Module Tests (233)
*** RC-AU: Authentication (20)
*** RC-NAV: Navigation (12)
*** RC-CAM: Camera (39)
*** RC-CC: Color Correction (27)
*** RC-SIM: CVD Simulation (7)
*** RC-CI: Color Identifier (20)
*** RC-ISH: Ishihara Screening (47)
*** RC-HF: Haptic Feedback (9)
*** RC-AF: Audio Feedback (11)
*** RC-CA: Career Awareness (8)
*** RC-SET: Settings (9)
*** RC-FB: Firebase (12)
*** RC-GAL: Gallery (12)
** Part B: Functionality Tests (107)
*** FN-SP: Splash Screen (3)
*** FN-OB: App Onboarding (6)
*** FN-LG: Login and Auth (6)
*** FN-HM: Home Screen (5)
*** FN-IT: Ishihara Intro (6)
*** FN-TE: Test Execution (17)
*** FN-RS: Results and Scoring (13)
*** FN-CE: Camera Enhancement (14)
*** FN-CI: Color Identifier (6)
*** FN-SIM: CVD Simulation (5)
*** FN-GL: CVD Gallery (4)
*** FN-ED: Education and Articles (6)
*** FN-SV: Survey (3)
*** FN-ST: Settings (8)
*** FN-HI: History (3)
*** FN-AD: Admin and Research (6)
** Part C: Android Core (80)
*** VE: Visual Experience (20)
*** AF: Android Functionality (20)
*** PS: Performance and Stability (20)
*** SC: Privacy and Security (20)
** Part D: Compatibility (37)
*** CM-AV: Android Versions (7)
*** CM-SS: Screen Sizes (8)
*** CM-OR: Orientation (2)
*** CM-CH: Camera Hardware (7)
*** CM-NW: Network Conditions (8)
*** CM-DV: Device Edge Cases (7)
@endwbs"""

OUT = r"c:\Users\markr\OneDrive\Desktop\Local Project Filee\ReColor\evaluation\diagrams\test_case_hierarchy.png"

encoded = plantuml_encode(puml)
url = f"https://www.plantuml.com/plantuml/png/{encoded}"
ctx = ssl.create_default_context()
req = urllib.request.Request(url)
req.add_header("User-Agent", "Mozilla/5.0")

with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
    data = resp.read()
with open(OUT, "wb") as f:
    f.write(data)
print(f"[OK] test_case_hierarchy.png ({len(data)/1024:.1f} KB)")
