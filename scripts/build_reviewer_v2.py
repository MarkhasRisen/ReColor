"""Generate ReColor_Defense_Reviewer_v2.docx — grouped by camera mode."""
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

for section in doc.sections:
    section.top_margin    = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin   = Cm(2)
    section.right_margin  = Cm(2)

style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

# ── Helpers ──
def H1(text):
    p = doc.add_heading(text, level=1)
    for r in p.runs: r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x93)
    return p

def H2(text):
    p = doc.add_heading(text, level=2)
    for r in p.runs: r.font.color.rgb = RGBColor(0x2E, 0x50, 0xB0)
    return p

def H3(text):
    p = doc.add_heading(text, level=3)
    for r in p.runs: r.font.color.rgb = RGBColor(0x50, 0x70, 0xC0)
    return p

def P(content, bold=False, italic=False):
    p = doc.add_paragraph()
    if isinstance(content, str):
        r = p.add_run(content); r.bold = bold; r.italic = italic
    else:
        for text, opts in content:
            r = p.add_run(text)
            r.bold = opts.get('bold', False)
            r.italic = opts.get('italic', False)
            if opts.get('mono'):
                r.font.name = 'Consolas'; r.font.size = Pt(10)
            if opts.get('color'):
                r.font.color.rgb = RGBColor(*opts['color'])
    return p

def BULLET(text):
    return doc.add_paragraph(text, style='List Bullet')

def NUM(text):
    return doc.add_paragraph(text, style='List Number')

def CODE(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    r = p.add_run(text)
    r.font.name = 'Consolas'; r.font.size = Pt(9.5)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), 'F2F2F2')
    p._p.get_or_add_pPr().append(shd)
    return p

def QUOTE(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.right_indent = Cm(0.75)
    r = p.add_run(text); r.italic = True
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return p

def TABLE(headers, rows, widths=None):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = 'Light Grid Accent 1'
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]; cell.text = ''
        r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[1+ri].cells[ci]; cell.text = ''
            r = cell.paragraphs[0].add_run(str(val)); r.font.size = Pt(10)
    if widths:
        for row in t.rows:
            for ci, w in enumerate(widths):
                row.cells[ci].width = Cm(w)
    return t

def QA(q, a):
    p = doc.add_paragraph()
    r = p.add_run(q); r.bold = True; r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x93)
    QUOTE(a)

# ═══════════════════════════════════════════════════════════════════
# COVER
# ═══════════════════════════════════════════════════════════════════
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run('ReColor Thesis Defense\nAlgorithm Reviewer — By Camera Mode')
r.bold = True; r.font.size = Pt(22); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x93)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run('Shared Infrastructure · Color Identifier · CVD Simulation · Camera Enhancement')
r.italic = True; r.font.size = Pt(12); r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_paragraph()
intro = doc.add_paragraph()
intro.add_run(
    "This reviewer is organized by the three camera modes in the ReColor app. Each section covers "
    "the full pipeline end-to-end: how the camera captures a frame, how the pixels are processed, "
    "what each algorithm does with what values, and which panelist questions to expect. Read it in "
    "order — the shared infrastructure in Part 0 is referenced by all three mode-specific parts."
)
doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════════════
H1('Contents')
BULLET('PART 0 — General / Shared Infrastructure (applies to all 3 cameras)')
BULLET('PART 1 — Color Identifier (CIELAB Delta-E nearest-neighbor)')
BULLET('PART 2 — CVD Simulation (Viénot + GPU shader, freeze pattern)')
BULLET('PART 3 — Camera Enhancement (Daltonization + Hue Rotation, freeze pattern)')
BULLET('PART 4 — Cross-Mode Panelist Questions')
BULLET('PART 5 — Cheatsheet + Night-Before Rehearsal')
doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 0 — GENERAL / SHARED INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════
H1('PART 0 — General Infrastructure (applies to all 3 cameras)')
P(
    "Before diving into each mode, master these foundations. All three cameras share them. "
    "A panelist who asks 'how does the camera work?' expects an answer from this section."
)

H2('0.1 — The VisionCamera capture pipeline')
P(
    "All three camera screens use the same React Native VisionCamera library. The lifecycle is "
    "identical:"
)
NUM('useCameraPermission() — requests CAMERA permission from Android/iOS.')
NUM('useCameraDevice(position) — picks back or front camera.')
NUM('useIsFocused() — detects whether the screen is currently visible (deactivates camera on navigation away).')
NUM('<Camera ref={cameraRef} isActive={isFocused} photo={true} /> — renders the live preview.')
NUM('cameraRef.current.takePhoto() — captures a high-resolution still (returns a file path).')

P('The captured photo is always processed off-preview — no live pixel manipulation in JavaScript.', bold=True)

H2('0.2 — The HAL race guard (why all 3 cameras have a 700-900ms delay)')
P(
    "On Android, the Hardware Abstraction Layer (HAL) takes time to release a camera when the "
    "previous screen unmounts. If the next camera screen activates too quickly, the app crashes "
    "with a native-side camera-open conflict. Two guards fix this:"
)

P('Guard 1 — Delayed camera activation:', bold=True)
CODE("""useEffect(() => {
  const timer = setTimeout(() => {
    if (isMountedRef.current) setCameraReady(true);
  }, 700); // 900ms for Simulation, 700ms for Identifier/Enhance
  return () => clearTimeout(timer);
}, []);""")

P('Guard 2 — beforeRemove listener (deactivate before unmount):', bold=True)
CODE("""useEffect(() => {
  const unsub = navigation.addListener('beforeRemove', () => {
    setCameraReady(false);
  });
  return unsub;
}, [navigation]);""")

P(
    "The beforeRemove event fires BEFORE React removes the screen from the tree, giving the camera "
    "a clean teardown window. The delay-to-activate on the next screen then gives the HAL time to "
    "fully release the previous device handle."
)

H2('0.3 — sRGB gamma correction (foundational)')
P(
    "A JPEG pixel stored as 128 is NOT half the light of 255. sRGB is a non-linear encoding "
    "optimized for perceptual uniformity. Any algorithm that models physical light (Viénot cone "
    "response, CIELAB XYZ conversion, daltonization error math) MUST operate in linear space."
)

P('Exact piecewise (used in Color Identifier only, tensorHelper.js:137-139):', bold=True)
CODE("""if c ≤ 0.04045:  linear = c / 12.92
else:            linear = ((c + 0.055) / 1.055) ^ 2.4""")

P('Approximation (used in Simulation shader + Daltonization hot loop):', bold=True)
CODE("""linear = c ^ 2.2         (decode, sRGB → linear)
srgb   = linear ^ 0.4545  (encode, linear → sRGB; 0.4545 = 1/2.2)""")

P([('Error vs exact: ', {}), ('<1%', {'bold': True}),
   (', speed gain ~3× on millions of pixels. Identifier uses exact because LAB class assignment is '
    'sensitive to systematic bias; Simulation/Enhancement use the approximation because output is '
    're-encoded anyway.', {})])

H2('0.4 — The 10-class color system')
P(
    "The app uses a fixed taxonomy of 10 perceptual color classes, defined in "
    "tensorHelper.js:20-31. All algorithms — color naming, CVD confusion sets, daltonization "
    "gating (historical) — agree on these 10 names:"
)
CODE("""0: Neutral   1: Red    2: Orange   3: Yellow   4: Green
5: Cyan      6: Blue   7: Violet   8: Pink     9: Brown""")

P('Which classes confuse which CVD type (tensorHelper.js:37-41):')
TABLE(
    ['CVD type', 'Confused classes', 'Physiological reason'],
    [
        ['Protan', 'Red, Orange, Violet, Brown', 'All contain significant red → collapses to dark for missing L-cone'],
        ['Deutan', 'Red, Orange, Green, Brown',  'Classic red-green axis via missing M-cone'],
        ['Tritan', 'Yellow, Cyan, Blue, Pink',   'All contain blue component → lost with missing S-cone'],
    ],
    widths=[3, 6, 8]
)

H2('0.5 — JPEG decode and data-URI encode (shared utilities)')
P(
    "All mode-specific pipelines call the same two helpers in tensorHelper.js:"
)
BULLET('decodeJpegBase64(base64) → { data: Uint8Array RGBA, width, height } — uses jpeg-js library')
BULLET('encodeToDataUri(rgbaPixels, w, h) → "data:image/jpeg;base64,..." — chunked conversion to avoid O(n²) string concat')

P(
    "The RGBA buffer stores 4 bytes per pixel: [R, G, B, A, R, G, B, A, ...]. The index for "
    "pixel (x, y) is (y * width + x) * 4. Every algorithm iterates this buffer with i += 4 stride."
)

H2('0.6 — Error boundaries and crash recovery')
P(
    "All three camera screens are wrapped in a ScreenErrorBoundary (React ErrorBoundary). "
    "Global handlers in App.js:168-191 catch BOTH unhandled exceptions AND unhandled promise "
    "rejections, log them via AppLog to a persistent file (~/.recolor_debug.log), and show a "
    "recovery UI instead of a full-app crash."
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 1 — COLOR IDENTIFIER
# ═══════════════════════════════════════════════════════════════════
H1('PART 1 — Color Identifier')
P('Location: App.js:2228-2470 (ColorIdentifierScreenInner), tensorHelper.js:148-336 (algorithms).',
  italic=True)

H2('1.1 — What it does')
P(
    "User taps anywhere on the live camera preview. A crosshair appears at the tap point. The app "
    "captures a photo, reads the pixel under the crosshair, and displays the color name plus a "
    "confidence percentage. Four names per class are possible (e.g., Red / Crimson / Firebrick / "
    "Indian Red). This is the only mode that does NOT use the freeze pattern — each tap produces "
    "a fresh one-shot identification."
)

H2('1.2 — Full data flow (App.js:2287-2366)')
NUM('User taps screen → handleTouchEnd captures (locationX, locationY) — the crosshair pixel in screen coordinates.')
NUM('runDetection(cx, cy) fires. A guard (isProcessingRef) prevents concurrent taps from racing.')
NUM('cameraRef.current.takePhoto({ qualityPrioritization: "speed" }) — fast capture, not the highest quality because we only need ONE pixel.')
NUM('ImageManipulator.manipulateAsync resizes to 640px wide with base64: true — small size because we only read a 10×10 patch.')
NUM('decodeJpegBase64(resized.base64) → RGBA Uint8Array at image resolution (imgW, imgH).')
NUM('Cover-mode coordinate mapping → convert (cx, cy) from screen space to image pixel (pixX, pixY). This is the subtle step.')
NUM('Average a 10×10 pixel neighborhood around (pixX, pixY) to reduce noise.')
NUM('identifyColor(avgR, avgG, avgB) — the main algorithm, runs CIELAB Delta-E nearest-neighbor against 77 reference points.')
NUM('Display: color name, confidence %, and the actual sampled hex swatch (so user can spot mismatches).')

H2('1.3 — Cover-mode coordinate mapping (critical, panelists LOVE this)')
P(
    "The camera preview uses style={StyleSheet.absoluteFill} with default object-fit: cover — "
    "the image fills the screen, and the sides or top/bottom are CROPPED. The camera's photo "
    "resolution does NOT match the screen's aspect ratio. We must map screen coordinates to the "
    "correct pixel in the source image:"
)

CODE("""const screenAspect = width / screenHeight;
const photoAspect  = imgW / imgH;

let pixX, pixY;
if (photoAspect > screenAspect) {
  // Photo is WIDER than screen → left/right cropped; height fills
  const visibleW = screenAspect * imgH;
  const offsetX  = (imgW - visibleW) / 2;
  pixX = Math.round(offsetX + (cx / width) * visibleW);
  pixY = Math.round((cy / screenHeight) * imgH);
} else {
  // Photo is TALLER than screen → top/bottom cropped; width fills
  const visibleH = imgW / screenAspect;
  const offsetY  = (imgH - visibleH) / 2;
  pixX = Math.round((cx / width) * imgW);
  pixY = Math.round(offsetY + (cy / screenHeight) * visibleH);
}""")

P([('Without this mapping, ', {'bold': True}),
   ('the color identified would be WRONG — the user touches "red apple" on screen but the '
    'algorithm samples the "green leaf" because screen (0,0) maps to image (offset, 0) after '
    'cover-cropping. This is a hidden land mine; verify it works by tapping the exact edges of '
    'the preview.', {})])

H2('1.4 — 10×10 neighborhood averaging (App.js:2332-2351)')
CODE("""const half = 5;
const x0 = max(0, pixX - half);
const y0 = max(0, pixY - half);
const x1 = min(imgW, pixX + half);
const y1 = min(imgH, pixY + half);

let avgR = 0, avgG = 0, avgB = 0, count = 0;
for (let py = y0; py < y1; py++) {
  for (let px = x0; px < x1; px++) {
    const idx = (py * imgW + px) * 4;
    avgR += decoded.data[idx];
    avgG += decoded.data[idx+1];
    avgB += decoded.data[idx+2];
    count++;
  }
}
avgR /= count; avgG /= count; avgB /= count;""")

P([('Why 10×10 and not 1×1? ', {'bold': True}),
   ('Camera sensor noise in a single pixel can swing RGB values by ±10-15 in low light. Averaging '
    'a 100-pixel patch reduces noise variance by √100 = 10×, producing stable readings. The patch '
    'is small enough (~0.3 mm on screen) that it still samples a single color surface.', {})])

H2('1.5 — sRGB → CIELAB conversion (tensorHelper.js:148-170)')
P('Step 1 — sRGB gamma decode (EXACT IEC 61966-2-1 piecewise):', bold=True)
CODE("""function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}""")
P(
    "Identifier uses the exact piecewise curve (not the 2.2 approximation). Color name "
    "classification is sensitive to small systematic biases — a 1% shift in linear R can push a "
    "borderline orange into red."
)

P('Step 2 — linear RGB → XYZ (D65 white point matrix):', bold=True)
CODE("""X = (0.4124564·R + 0.3575761·G + 0.1804375·B) / 0.95047
Y = (0.2126729·R + 0.7151522·G + 0.0721750·B) / 1.00000
Z = (0.0193339·R + 0.1191920·G + 0.9503041·B) / 1.08883""")

P([
    ('The division by ', {}),
    ('(0.95047, 1.0, 1.08883)', {'mono': True}),
    (' normalizes to the D65 illuminant so that a pure white pixel (1,1,1) maps to XYZ = (1,1,1). ', {}),
    ('D65', {'bold': True}),
    (' = standard daylight at 6504K, the reference illuminant for sRGB. Any other white point '
     'would shift all LAB values.', {})
])

P('Step 3 — XYZ → LAB (nonlinear f-function):', bold=True)
CODE("""f(t) = t^(1/3)           if t > 0.008856
f(t) = 7.787·t + 16/116  otherwise  // linear near 0 to avoid cbrt(0) instability

L = 116·f(Y) - 16       // lightness  [0, 100]
a = 500·(f(X) - f(Y))   // green-red axis, roughly [-128, +128]
b = 200·(f(Y) - f(Z))   // blue-yellow axis, roughly [-128, +128]""")

P(
    "LAB is a perceptually uniform color space: equal Euclidean distances ≈ equal perceived color "
    "differences. This is the KEY property — RGB distance does NOT match perception, so a naive "
    "RGB nearest-neighbor classifier would group colors incorrectly."
)

H2('1.6 — Weighted Delta-E (tensorHelper.js:177-184)')
CODE("""const L_WEIGHT = 0.5;
function deltaE(lab1, lab2) {
  return sqrt(
    L_WEIGHT * (lab1[0] - lab2[0])² +
               (lab1[1] - lab2[1])² +
               (lab1[2] - lab2[2])²
  );
}""")
P([('Why down-weight L* (lightness) by 0.5? ', {'bold': True}),
   ('Because a "dark red" and a "bright red" are both "Red" — the class name does not depend on '
    'how bright the pixel is. Full-weight L* would make "dark red" closer to "dark green" than to '
    '"bright red," which breaks naming. Hue and chroma (a* and b*) determine color identity, so '
    'they stay at full weight.', {})])

P('Delta-E scale reference (cite if asked):')
TABLE(['Delta-E', 'Perceptual meaning'],
      [['< 1.0', 'Imperceptible to the human eye'],
       ['< 2.0', 'Perceptible only under close inspection'],
       ['2 – 10', 'Noticeable at a glance'],
       ['10 – 49', 'Different colors'],
       ['100', 'Exact opposites']],
      widths=[3, 12])

H2('1.7 — Chroma gate (tensorHelper.js:193, 314)')
CODE("""const NEUTRAL_CHROMA_THRESHOLD = 12;
const chroma = Math.sqrt(lab[1]² + lab[2]²);  // C* distance from neutral axis
const isChromatic = chroma >= 12;

for (const entry of IDENTIFIER_DB) {
  if (isChromatic && entry.name === 'Neutral') continue;  // skip grays
  const dist = deltaE(lab, entry.lab);
  if (dist < bestDist) { bestDist = dist; bestName = entry.name; ... }
}""")

P([('Why a chroma gate? ', {'bold': True}),
   ('Without it, a dim gray wall (small a*/b* values due to sensor noise) could be classified as '
    '"dark red" because its a* happens to be slightly positive. The gate forces: if C* < 12, you '
    'are truly gray → only match against neutral entries (black/gray/white). If C* ≥ 12, skip '
    'neutrals entirely — the pixel has real color.', {})])

H2('1.8 — Confidence score (tensorHelper.js:333)')
CODE('const confidence = Math.max(0, Math.round(100 - bestDist * 2));')
P(
    "Linear scaling: Delta-E 0 = 100% confidence, Delta-E 50 = 0% confidence, anything higher is "
    "clamped to 0. The factor of 2 is calibrated so that Delta-E 10 (noticeable difference) = 80% "
    "confidence, matching the user's perception of reliability during testing."
)

H2('1.9 — The 77-entry reference database (tensorHelper.js:195-298)')
P(
    "Each class has 6-10 representative entries covering saturated, muted, dark, and soft variants. "
    "Example for 'Red' class (9 entries):"
)
TABLE(
    ['Hex', 'RGB', 'Named variant'],
    [
        ['#FF0000', '(255, 0, 0)',    'Pure Red'],
        ['#CC0000', '(204, 0, 0)',    'Saturated Red (muted)'],
        ['#8B0000', '(139, 0, 0)',    'Dark Red'],
        ['#DC143C', '(220, 20, 60)',  'Crimson'],
        ['#B22222', '(178, 34, 34)',  'Firebrick'],
        ['#FF3333', '(255, 51, 51)',  'Light Red'],
        ['#CD5C5C', '(205, 92, 92)',  'Indian Red'],
        ['#8B3A3A', '(139, 58, 58)',  'Dark Muted Red'],
        ['#E06060', '(224, 96, 96)',  'Soft Red'],
    ],
    widths=[3, 5, 7],
)
P(
    "All entries classify into the SAME class name (or a named variant). When Delta-E finds the "
    "nearest of the 9, we return both 'Red' (class) and potentially the variant label (e.g. "
    "'Crimson') for display."
)

H2('1.10 — Identifier-specific panelist questions')
QA('"Why only one pixel\'s worth of information? Isn\'t that error-prone?"',
   "We sample a 10×10 = 100-pixel neighborhood and average, reducing sensor noise by √100 = 10×. "
   "A single-pixel read would swing ±15 per channel in low light; the average is stable within ±2.")

QA('"Why not use a deep-learning color classifier?"',
   "Three reasons. (1) Determinism — the reference database is inspectable and reproducible; "
   "panelists can verify classification by hand. (2) No training data required — our 10 classes "
   "are defined from color-vision literature, not learned. (3) CIELAB Delta-E is the industry "
   "standard for color-difference measurement (ISO 12647), used in print, paint, and textile "
   "industries. An ML classifier would be a black box with no accuracy advantage for a "
   "well-defined 10-class problem.")

QA('"What if the user taps a color boundary (e.g., red-and-green edge)?"',
   "The 10×10 patch averages across the boundary, producing a muddy intermediate color (e.g., "
   "olive-brown for red+green). Delta-E then matches against the nearest reference, which may be "
   "'Brown' — technically wrong but reasonable given the input. This is an inherent limitation of "
   "point-sampling; we could add a homogeneity check, but it would slow the interaction.")

QA('"Why not use HSV instead of LAB?"',
   "HSV is convenient for hue manipulation (which we use in the Hue Rotation algorithm), but it "
   "is NOT perceptually uniform. Equal numerical distances in HSV can look wildly different — "
   "e.g., (H=30, S=1, V=1) vs (H=60, S=1, V=1) is 'orange to yellow,' but (H=30, S=0.1, V=1) vs "
   "(H=60, S=0.1, V=1) is two near-whites. LAB handles this correctly; HSV does not.")

QA('"How does the cover-mode mapping handle front camera vs back camera?"',
   "VisionCamera reports photo dimensions (photo.width, photo.height) already accounting for "
   "orientation, and ImageManipulator applies EXIF rotation on resize. So (imgW, imgH) are always "
   "the correct display-orientation dimensions. The front camera's mirroring is purely a preview "
   "effect — the captured photo is NOT mirrored — so no extra handling is needed.")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 2 — CVD SIMULATION
# ═══════════════════════════════════════════════════════════════════
H1('PART 2 — CVD Simulation')
P('Location: App.js:2460+ (CVDSimulationScreenInner), App.js:84-100 (SkSL shader), tensorHelper.js:49-109 (matrices).',
  italic=True)

H2('2.1 — What it does')
P(
    "Shows a normal-vision user what a person with color vision deficiency sees. Tap the snowflake "
    "button to freeze a photo, then tap Protan/Deutan/Tritan to apply the simulation. Switching "
    "between CVD types is INSTANT because it only changes GPU shader uniforms on the cached image."
)
P(
    "Simulation is NOT an assistive feature — it is educational. It shows accessibility designers "
    "what a poster or UI looks like to CVD users, or helps CVD users explain their experience to "
    "family members."
)

H2('2.2 — Full data flow (freeze pattern)')
NUM('User taps snowflake button → handleFreeze().')
NUM('cameraRef.current.takePhoto({ qualityPrioritization: "quality" }) — high quality because this becomes the displayed image.')
NUM('ImageManipulator resize to 1040px longest edge (see Part 2.3 for why 1040).')
NUM('setFrozenUri(resized.uri) + setFrozen(true) — stops the live camera, saves file URI to state.')
NUM('useImage(frozenUri) hook in react-native-skia loads the JPEG into a GPU texture. This is ASYNC — skImage starts null, becomes a GPU-resident SkImage once loaded.')
NUM('<Canvas>...<SkiaImage image={skImage}>...<RuntimeShader /></SkiaImage>...</Canvas> renders the frozen photo through the CVD shader on every React render.')
NUM('When user taps Protan/Deutan/Tritan, setCvdType(m) changes the uniforms passed to RuntimeShader. React re-renders. The GPU applies the new matrix on the next frame. No re-capture, no JS pixel loop.')
NUM('Save: canvasRef.current.makeImageSnapshot() captures the rendered GPU canvas (INCLUDING the shader output) to PNG. Write to FileSystem, then MediaLibrary.saveToLibraryAsync.')

H2('2.3 — Why resize to 1040p?')
BULLET('Quality ceiling: 1040p is the highest resolution where individual pixels are still imperceptible on a phone screen at normal viewing distance.')
BULLET('GPU memory: a 1040×780 RGBA texture = 3.2 MB; full 1080p = 8.3 MB; 4K = 33 MB. Mobile GPUs have limited texture memory.')
BULLET('Shader throughput: the shader runs per texture sample, so 4× fewer pixels = 4× faster snapshot on save.')

H2('2.4 — The Viénot-Brettel-Mollon 1999 matrices (tensorHelper.js:49-65)')
P('The core mathematical model. MEMORIZE the Protan row 0 at minimum.')

P('Protan (missing L cone):', bold=True)
CODE("""[ 0.152286  1.052583  -0.204868]    ← R_out = 0.15·R + 1.05·G - 0.20·B
[ 0.114503  0.786281   0.099216]    ← G_out
[-0.003882 -0.048116   1.051998]    ← B_out""")

P('Deutan (missing M cone):', bold=True)
CODE("""[ 0.367322  0.860646  -0.227968]
[ 0.280085  0.672501   0.047413]
[-0.011820  0.042940   0.968881]""")

P('Tritan (missing S cone):', bold=True)
CODE("""[ 1.255528 -0.076749  -0.178779]
[-0.078411  0.930809   0.147602]
[ 0.004733  0.691367   0.303900]""")

P([('Key insight: ', {'bold': True}),
   ('Protan row 0 — perceived R = 15% of original R + 105% of G − 20% of B. MOST of the red '
    'signal a Protan perceives comes from the GREEN channel reading through their intact '
    'M-cone. That is the defining feature of protanopia.', {})])

H2('2.5 — Matrix properties panelists love')
P([('Property 1 — Rows sum to ~1.0: ', {'bold': True}),
   ('Protan row 0: 0.152 + 1.053 − 0.205 = 1.000. This preserves grays: input (v,v,v) → output '
    '(v,v,v). Answer when asked "what happens to a gray pixel?" → "grays stay gray, always."', {})])

P([('Property 2 — Negative entries: ', {'bold': True}),
   ('Mathematically valid (projection coefficients onto the dichromatic confusion plane in LMS '
    'space), but can produce out-of-range intermediate values. The shader clamps with '
    'clamp(sim, 0, 1) on line 96.', {})])

P([('Property 3 — Tritan row 0 has coefficient > 1: ', {'bold': True}),
   ('1.255 on R. This reflects that Tritans "compensate" by reading blue/yellow info through '
    'the remaining L and M cones, with an effective gain > 1. This is why Tritan outputs can '
    'look more saturated than source in warm regions.', {})])

H2('2.6 — The SkSL shader line-by-line (App.js:84-100)')
CODE("""uniform shader contents;                      // input texture
uniform half3 row0;                            // CVD matrix row 0
uniform half3 row1;
uniform half3 row2;

half4 main(float2 coord) {
  half4 c = contents.eval(coord);              // sample input pixel (sRGB)
  half3 lin = pow(c.rgb, half3(2.2));          // sRGB → linear (gamma decode)
  half3 sim = half3(                            // 3×3 matrix as three dot products
    dot(row0, lin),
    dot(row1, lin),
    dot(row2, lin));
  sim = clamp(sim, half3(0.0), half3(1.0));    // clip out-of-gamut projections
  return half4(pow(sim, half3(0.4545)), c.a);  // linear → sRGB (gamma encode)
}""")

P([('Why GPU and not JS? ', {'bold': True}),
   ('1040×780 = 811,200 pixels. In JS: ~400ms. On GPU: ~4ms. The GPU runs the shader in parallel '
    'across thousands of pixel-shader cores. On-screen, it feels instantaneous — switching CVD '
    'types updates in under 16ms (one frame at 60Hz).', {})])

P([('Why half precision? ', {'bold': True}),
   ('half = 16-bit float, enough for 8-bit display output. Mobile GPUs run half-precision ops '
    '~2× faster than float (single-precision), and there is no visible quality loss.', {})])

H2('2.7 — Switching CVD types is INSTANT (no re-capture)')
P(
    "When the user taps Deutan after Protan, here's what happens:"
)
NUM('setCvdType("Deutan") changes React state.')
NUM('cvdUniforms = useMemo(() => getCVDRows(cvdType), [cvdType]) recomputes — returns the Deutan rows.')
NUM('React re-renders <RuntimeShader source={CVD_EFFECT} uniforms={cvdUniforms} />.')
NUM('react-native-skia passes the new uniforms to the GPU.')
NUM('The GPU re-runs the shader on the SAME texture with the NEW matrix. One frame later, the screen updates.')

P(
    "The frozen image is decoded ONCE (by useImage). Switching modes never re-decodes, never "
    "re-takes a photo, never touches the JS pixel buffer. This is why it feels instant."
)

H2('2.8 — Saving the simulated image')
P('canvasRef.current.makeImageSnapshot() is THE key technique. It reads back the rendered '
  'GPU canvas as an SkImage, which we then encodeToBase64() and write to a temp PNG:')
CODE("""const snapshot = canvasRef.current.makeImageSnapshot();
const b64 = snapshot.encodeToBase64();
const tmpPath = `${FileSystem.cacheDirectory}recolor_sim_${Date.now()}.png`;
await FileSystem.writeAsStringAsync(tmpPath, b64, { encoding: Base64 });
await MediaLibrary.saveToLibraryAsync(tmpPath);""")

P(
    "The saved PNG contains the POST-SHADER output — i.e., what the user is seeing on screen, "
    "including the CVD simulation. If we saved the original JPEG instead, the user would get an "
    "unsimulated copy, defeating the purpose."
)

H2('2.9 — Simulation-specific panelist questions')
QA('"How is your simulation different from online simulators like Coblis?"',
   "Coblis also uses Viénot/Brettel matrices, but many online implementations skip gamma "
   "correction — they apply the matrix directly to sRGB values. This produces a washed-out, "
   "under-simulated output, especially for Protan and Tritan where the matrix entries are far "
   "from identity. Our shader decodes to linear, applies the matrix, and re-encodes to sRGB — "
   "this is the colorimetrically correct implementation.")

QA('"Why freeze instead of live simulation?"',
   "We attempted live simulation with useSkiaFrameProcessor — the GPU shader is fast enough to "
   "run on every video frame. However, on Android this combination introduced HAL race conditions "
   "between VisionCamera and react-native-skia: the app crashed when transitioning between camera "
   "screens. The freeze pattern eliminates the live frame processor, which removes the crash "
   "source. Trade-off: the user sees the normal camera, freezes, then sees the simulation — not "
   "a live AR view. But reliability > novelty for an assistive app.")

QA('"What prevents you from adding severity control (mild CVD)?"',
   "The Viénot matrices are derived for severity=1.0 (complete dichromacy). To support anomalous "
   "trichromacy (partial deficiency), you would switch to Machado 2009 which parameterizes "
   "severity from 0.0 to 1.0. This is straightforward future work — 11 matrices per CVD type "
   "instead of 1, with user-facing severity slider. We deferred because (a) Viénot's worst case "
   "still benefits milder cases, and (b) adding a slider adds cognitive load for an unclear UX win.")

QA('"Why PNG for the saved simulation and JPEG for the original capture?"',
   "makeImageSnapshot() returns a lossless raster. Re-encoding to JPEG would introduce compression "
   "artifacts on top of the shader output. PNG is lossless. File size is ~3× JPEG, but for a "
   "one-off save that is acceptable.")

QA('"How does the shader know the image orientation?"',
   "ImageManipulator applies EXIF rotation during resize. The saved JPEG has the correct "
   "orientation baked into the pixel layout. useImage decodes those pixels directly — no "
   "additional rotation needed. If we skipped ImageManipulator and loaded the raw camera output, "
   "we'd need to read EXIF ourselves.")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 3 — CAMERA ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════
H1('PART 3 — Camera Enhancement')
P('Location: App.js:1828-2211 (CameraEnhanceScreenInner), tensorHelper.js:473-629 (algorithms).',
  italic=True)

H2('3.1 — What it does')
P(
    "The only truly ASSISTIVE mode. Captures a photo, then processes the pixels so that a CVD "
    "user can distinguish colors they normally confuse. Two algorithms are selectable: "
    "Daltonization (the default, physically motivated) and Hue Rotation (a comparative baseline). "
    "Users can toggle between them on the same captured image to compare effects."
)

H2('3.2 — Full data flow (freeze + JS processing)')
NUM('User taps snowflake → handleFreeze().')
NUM('cameraRef.current.takePhoto({ qualityPrioritization: "quality" }).')
NUM('ImageManipulator resize to 1040p + base64: true — we need base64 because jpeg-js decodes FROM base64 to RGBA.')
NUM('frozenUriRef.current = resized.uri — save the ORIGINAL file URI for the before/after long-press toggle.')
NUM('decoded = decodeJpegBase64(resized.base64) → RGBA Uint8Array. Stored in decodedRef.current (NEVER mutated — applyDaltonization copies internally).')
NUM('runEnhancement(cvdType, algorithm) → picks applyDaltonization OR applyHueRotation, runs in JS, returns a new RGBA Uint8Array.')
NUM('encodeToDataUri(out, w, h) → data URI. setResultUri(uri). The <Image source={{ uri: resultUri }} /> displays it.')
NUM('Switching CVD type or algorithm: decodedRef is still cached → skip steps 2-5, go straight to step 6. ~400ms per re-enhancement.')

P(
    "Compare to Simulation: Simulation processes on GPU (fast, uses shader uniforms). Enhancement "
    "processes in JS (slower but the algorithms are too complex for a shader — especially hue "
    "rotation's HSV conversion). JS is ~400ms per 1040p image, which is acceptable for a "
    "tap-to-re-enhance interaction but unusable as live AR."
)

H2('3.3 — Why the decodedRef is never mutated')
P([('applyDaltonization begins with ', {}),
   ('const pixels = new Uint8Array(rawImageData.data)', {'mono': True}),
   (' — a COPY. We never overwrite the original decoded buffer. This means switching from Protan '
    'Daltonization to Deutan Hue Rotation always starts from the same clean input, never a '
    'double-processed pixel soup. If we mutated in place, switching CVD type twice would produce '
    'incorrect output.', {})])

H2('3.4 — ALGORITHM A: Daltonization (Brettel / Fidaner)')

H3('3.4.1 — The three-step intuition')
P('Daltonization is an assistive INVERSE of simulation. Given a CVD user who cannot see some colors:')
NUM('SIMULATE what they see: sim = SIM · original  (how the CVD user would perceive it without help)')
NUM('COMPUTE the error (what they LOSE): err = original - sim')
NUM('REDISTRIBUTE the error into channels they CAN see: out = original + E · err')

P(
    "For a pixel they already perceive correctly (e.g., pure blue for a Protan), "
    "sim ≈ original, so err ≈ 0, so out ≈ original. The pixel is unchanged. "
    "For a pixel in their confusion zone (e.g., red for Protan), err is large, and the error "
    "gets shifted into G and B so the Protan sees a tinted (yellowish or magenta) color "
    "distinct from other reds/browns."
)

H3('3.4.2 — The error-redistribution matrices (tensorHelper.js:116-132)')

P('Protan E:', bold=True)
CODE("""[ 0.0  0.0  0.0 ]    ← R stays untouched (Protan cannot see R well anyway)
[ 0.7  1.0  0.0 ]    ← G gets 70% of R_err + 100% of G_err
[ 0.7  0.0  1.0 ]    ← B gets 70% of R_err + 100% of B_err""")

P('Deutan E:', bold=True)
CODE("""[ 1.0  0.6  0.0 ]    ← R gets 60% of G_err
[ 0.0  0.0  0.0 ]    ← G useless — Deutan cannot see G well
[ 0.0  0.6  1.0 ]    ← B gets 60% of G_err""")

P('Tritan E:', bold=True)
CODE("""[ 1.0  0.0  0.7 ]    ← R gets 70% of B_err
[ 0.0  1.0  0.7 ]    ← G gets 70% of B_err
[ 0.0  0.0  0.0 ]    ← B useless — Tritan cannot see B well""")

P([('The rule: ', {'bold': True}),
   ('The row corresponding to the MISSING cone\'s channel is zero (no point redistributing there — '
    'the user cannot perceive it). The 0.7 and 0.6 weights are Fidaner\'s empirical sweet spots — '
    'higher than 1.0 oversaturates (pixels go unnaturally vivid), lower than 0.5 under-corrects '
    '(the CVD user still cannot discriminate).', {})])

H3('3.4.3 — The full per-pixel loop (tensorHelper.js:485-531)')
CODE("""for (let i = 0; i < numPixels; i++) {
  if (useGate && !confusionSet.has(mask[i])) continue;  // mask param optional

  const rIdx = i * 4;
  // sRGB → linear (gamma decode with pow 2.2 approximation)
  const r = Math.pow(pixels[rIdx]   / 255.0, 2.2);
  const g = Math.pow(pixels[rIdx+1] / 255.0, 2.2);
  const b = Math.pow(pixels[rIdx+2] / 255.0, 2.2);

  // Simulate CVD perception in linear space
  const rSim = SIM[0][0]*r + SIM[0][1]*g + SIM[0][2]*b;
  const gSim = SIM[1][0]*r + SIM[1][1]*g + SIM[1][2]*b;
  const bSim = SIM[2][0]*r + SIM[2][1]*g + SIM[2][2]*b;

  // Error = what got lost
  const rErr = r - rSim, gErr = g - gSim, bErr = b - bSim;

  // Redistribute error into surviving channels
  const rOut = r + ERR[0][0]*rErr + ERR[0][1]*gErr + ERR[0][2]*bErr;
  const gOut = g + ERR[1][0]*rErr + ERR[1][1]*gErr + ERR[1][2]*bErr;
  const bOut = b + ERR[2][0]*rErr + ERR[2][1]*gErr + ERR[2][2]*bErr;

  // linear → sRGB (gamma encode, clamp to [0, 255])
  pixels[rIdx]   = clamp(round(pow(clamp01(rOut), 1/2.2) * 255), 0, 255);
  pixels[rIdx+1] = clamp(round(pow(clamp01(gOut), 1/2.2) * 255), 0, 255);
  pixels[rIdx+2] = clamp(round(pow(clamp01(bOut), 1/2.2) * 255), 0, 255);
}""")

H3('3.4.4 — THE SELF-GATING PROPERTY (your biggest talking point)')
P(
    "The mask parameter is OPTIONAL. In the original paper, a pixel classifier decides which "
    "pixels to daltonize (only pixels in confusion classes get touched). We removed that "
    "classifier — passing mask=null means every pixel runs the math. The algorithm still works "
    "correctly because:"
)
BULLET('For a pixel the CVD user ALREADY perceives correctly → sim ≈ original → err ≈ 0 → out = original. Pixel unchanged.')
BULLET('For a confused pixel → err is large → out is shifted. Correction applied.')

P([('This means ', {}),
   ('the error magnitude itself acts as the gate', {'bold': True}),
   ('. No CNN, no classifier, no extra model file needed. Daltonization is SELF-GATING.', {})])

P(
    "Removing the CNN: 7× speedup (2.8s → 0.4s), eliminated a ~8MB model file, removed a failure "
    "mode (classifier misclassification would incorrectly skip a pixel that needed correction)."
)

H3('3.4.5 — Gamma correctness in daltonization')
P(
    "Same concern as simulation: the Viénot matrices model linear light. If we applied them to "
    "sRGB values, the error estimate would be wrong → redistribution would be wrong → the output "
    "pixels would be subtly off. Our loop does sRGB → linear → simulate/error/redistribute → "
    "linear → sRGB. The 2.2 approximation is adequate here because output is re-encoded anyway; "
    "any <1% error is visually undetectable."
)

H2('3.5 — ALGORITHM B: Hue Rotation (comparative baseline)')

H3('3.5.1 — Why this exists')
P(
    "The paper needs a comparative analysis. Daltonization alone cannot be claimed 'best' without "
    "comparing to a representative of a different algorithm family. Hue Rotation represents "
    "rule-based / heuristic methods (as opposed to Daltonization's physically-motivated "
    "error-redistribution)."
)

H3('3.5.2 — The config (tensorHelper.js:543-550)')
CODE("""HUE_ROTATION_CONFIG = {
  Protan: { center:   0, range: 60, shift:  40 },   // reds → oranges/yellows
  Deutan: { center:   0, range: 60, shift:  40 },   // reds → oranges
  Tritan: { center: 240, range: 60, shift: -30 },   // blues → cyan/purple
};""")
P('Parameter meanings:')
BULLET('center = hue angle (in degrees, 0-360) where confusion is strongest. 0° = red. 240° = blue.')
BULLET('range = half-width of the affected band in degrees. 60 means hues within 60° of center get rotated. Outside that band: untouched.')
BULLET('shift = rotation amount AT band center (degrees). Positive = clockwise on the color wheel (toward yellow). Negative = counterclockwise (toward violet). Linearly tapers to 0 at the band edges.')

H3('3.5.3 — The HSV color model')
P(
    "Hue Rotation operates in HSV, not RGB. Why? Because HSV isolates color (H) from brightness "
    "(V) and colorfulness (S) — we can rotate H without affecting the other two. You cannot "
    "cleanly rotate 'color' in RGB without also shifting brightness."
)
TABLE(['Component', 'Range', 'Meaning'],
      [['H (hue)',        '0–360°',  '0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta'],
       ['S (saturation)', '0–1',     '0=gray, 1=pure color'],
       ['V (value)',      '0–1',     '0=black, 1=max brightness']],
      widths=[3, 3, 10])

H3('3.5.4 — Circular hue distance (handles 360° wrap)')
CODE("""function circularHueDistance(h, center) {
  const d = Math.abs(h - center);
  return d > 180 ? 360 - d : d;
}""")
P(
    "Hue wraps at 360°. Naively, hue 350° seems far from center=0° (distance 350), but on the "
    "color wheel it is only 10° away. This function gives the shortest arc distance."
)

H3('3.5.5 — The per-pixel loop (tensorHelper.js:565-625)')
CODE("""for (let i = 0; i < numPixels; i++) {
  const r = pixels[idx]/255, g = pixels[idx+1]/255, b = pixels[idx+2]/255;

  // RGB → HSV
  const max = max(r,g,b), min = min(r,g,b), d = max - min;
  const v = max;
  const s = max === 0 ? 0 : d / max;

  if (s < SAT_MIN) continue;  // skip near-grays (unreliable hue)

  let h;
  if (d === 0) h = 0;
  else if (max === r) h = 60 * ((g - b) / d % 6);
  else if (max === g) h = 60 * ((b - r) / d + 2);
  else              h = 60 * ((r - g) / d + 4);
  if (h < 0) h += 360;

  const dist = circularHueDistance(h, cfg.center);
  if (dist > cfg.range) continue;  // outside affected band → untouched

  // Linear taper: full shift at center, 0 at band edge
  const weight = 1 - dist / cfg.range;
  let newH = h + cfg.shift * weight;
  newH = ((newH % 360) + 360) % 360;  // wrap to [0, 360)

  // HSV → RGB (inline, no allocation)
  const c = v * s;
  const hp = newH / 60;
  const x = c * (1 - abs((hp % 2) - 1));
  let nr=0, ng=0, nb=0;
  if (hp < 1) { nr=c; ng=x; }
  else if (hp < 2) { nr=x; ng=c; }
  else if (hp < 3) { ng=c; nb=x; }
  else if (hp < 4) { ng=x; nb=c; }
  else if (hp < 5) { nr=x; nb=c; }
  else             { nr=c; nb=x; }
  const m = v - c;

  pixels[idx]   = round((nr + m) * 255);
  pixels[idx+1] = round((ng + m) * 255);
  pixels[idx+2] = round((nb + m) * 255);
}""")

H3('3.5.6 — Saturation gate (SAT_MIN = 0.15)')
P(
    "Near-gray pixels have an unreliable hue — tiny RGB noise (e.g., (128, 128, 130) vs "
    "(128, 130, 128)) produces wildly different hue values. Rotating the 'hue' of gray produces "
    "colored noise. We skip any pixel with s < 0.15 to preserve neutrals."
)

H3('3.5.7 — Linear taper')
P(
    "Without the taper, every pixel inside the band would get the SAME 40° shift, while pixels "
    "just outside the band would get 0° shift. This creates a visible hard edge in gradients — "
    "'banding artifact.' Linear taper (weight = 1 − dist/range) smoothly blends the rotation from "
    "full at center to zero at the edge, eliminating visible discontinuities."
)

H2('3.6 — Why Daltonization beats Hue Rotation (rehearse this)')
TABLE(
    ['Property', 'Daltonization', 'Hue Rotation'],
    [
        ['Theoretical basis',      'Physical cone-response model (Viénot 1999)', 'Heuristic rule'],
        ['Self-gating',            'Yes (error ≈ 0 for perceivable pixels)',     'No (manual band definition)'],
        ['Preserves identity',     'High — only shifts what is lost',            'Low — shifts entire band'],
        ['Handles all tones',      'Yes',                                         'Band-limited (range=60°)'],
        ['Parameter count',        '9 (SIM matrix) + 9 (ERR matrix) per CVD',    '3 (center, range, shift) per CVD'],
        ['Achromatic axis',        'Preserved by matrix design',                  'Preserved by sat gate'],
    ],
    widths=[4, 6, 6]
)

H2('3.7 — Instant re-enhancement via cached decodedRef')
P(
    "When the user changes CVD type or algorithm while frozen, we do NOT re-capture or re-decode. "
    "The original RGBA buffer is still in decodedRef.current. We just call runEnhancement with "
    "the new params, get a fresh Uint8Array, and encode it:"
)
CODE("""const handleCvdChange = (newType) => {
  if (newType === cvdType) return;      // no-op on same selection
  setCvdType(newType);
  setProcessing(true);
  setTimeout(() => {                     // 50ms delay → lets React paint the spinner
    const uri = runEnhancement(newType, algorithm);
    setResultUri(uri);
    setProcessing(false);
  }, 50);
};""")
P(
    "Total time: ~400ms for the pixel loop + ~50ms for the re-paint. Total UX: under half a "
    "second from tap to updated image. If we re-captured, it would be 2+ seconds (camera takes "
    "time, ImageManipulator takes time, decoder takes time)."
)

H2('3.8 — Before/after long-press toggle')
P(
    "The app provides a long-press zone (App.js:2069-2077) — a precisely-positioned rectangle "
    "that does NOT cover the CVD buttons, algorithm buttons, or nav controls. When held, "
    "setShowOriginal(true); when released, setShowOriginal(false). The display swaps between "
    "resultUri (enhanced) and frozenUriRef.current (original) — no reprocessing, just swap the "
    "<Image source>. This is purely a UX feature, no algorithm."
)

H2('3.9 — Enhancement-specific panelist questions')
QA('"Why do you run daltonization in JavaScript instead of on GPU?"',
   "Daltonization requires per-pixel branching (mask check), conditional gamma, and 3×3 matrix "
   "multiply PLUS 3×3 error matrix multiply. Expressing that in SkSL would be complex and would "
   "not produce meaningful speedup because the bottleneck is the data movement (decoding JPEG, "
   "encoding JPEG), not the pixel math. JS ~400ms is acceptable because enhancement is a "
   "tap-to-re-enhance interaction, not real-time. A GPU port is possible future work if we ever "
   "need sub-100ms refresh.")

QA('"Your system supports two enhancement algorithms. Isn\'t that confusing for users?"',
   "For end users, the default is Daltonization — they never need to touch Hue Rotation unless "
   "they want to compare. The dual-algorithm mode exists primarily for RESEARCH comparison (the "
   "thesis contribution). In a shipped product, we would likely hide Hue Rotation behind an "
   "'Advanced' toggle or remove it entirely.")

QA('"The 0.7 in your error matrix — why that specific number?"',
   "Fidaner's 2005 RGB daltonization paper explores this empirically. 1.0 produces oversaturation: "
   "the corrected pixels look like artificial neon. Below 0.5 produces under-correction: the CVD "
   "user still cannot discriminate adjacent confusion pixels. 0.7 is the empirical sweet spot "
   "balancing 'visible correction' against 'preserved naturalism.' It is a literature-standard "
   "value, not one we tuned ourselves.")

QA('"Can Daltonization fail? Show me a case where it does nothing."',
   "A photo of pure blue sky for a Protan. Blue is not in Protan's confusion set — they perceive "
   "blue correctly. sim ≈ original, err ≈ 0, out = original. The algorithm correctly identifies "
   "'nothing to fix' via self-gating. Similarly, a photo of a grayscale newspaper: rows sum to ~1, "
   "so grays stay gray, err = 0, output unchanged. This is a FEATURE: the algorithm does not over-"
   "correct when the input does not need help.")

QA('"Why does the freeze take 2 seconds? That feels slow."',
   "Breakdown: takePhoto ~600ms (camera hardware), ImageManipulator resize ~400ms (JPEG re-encode), "
   "decodeJpegBase64 ~200ms (base64 decode + JPEG decode), daltonization ~400ms (per-pixel math), "
   "encodeToDataUri ~300ms (RGBA → JPEG → base64). Total ~2000ms. The bottleneck is actually "
   "takePhoto and JPEG round-trip, not our algorithm. An optimization: use YUV frame grabs "
   "directly instead of JPEG capture — future work.")

QA('"What happens with a completely dark photo (indoor, low light)?"',
   "Camera sensor noise dominates dark regions — RGB values jump around ±15. Daltonization's "
   "gamma decode pushes dark values even further toward zero (pow 2.2 compresses the low end), "
   "so noise becomes negligible in linear space. But the corrected output is also mostly dark, "
   "and the user cannot distinguish colors in darkness regardless of correction. Low-light "
   "enhancement is a separate problem not addressed by daltonization.")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 4 — CROSS-MODE PANELIST QUESTIONS
# ═══════════════════════════════════════════════════════════════════
H1('PART 4 — Cross-Mode Panelist Questions')
P(
    "These questions are not mode-specific — they probe your architectural and design decisions "
    "across the whole system. Expect at least two of these.",
    italic=True
)

QA('"What is the difference between Simulation and Enhancement? They both use the same matrices."',
   "Opposite directions. SIMULATION takes a normal image and outputs 'what a CVD user sees' — "
   "useful for accessibility designers or for CVD users to show family members. ENHANCEMENT takes "
   "a normal image and outputs a version 'a CVD user can discriminate' — useful for the CVD user "
   "themselves. Simulation applies SIM matrix directly. Enhancement simulates, subtracts to get "
   "the error, then redistributes that error into surviving channels. Same matrices, different "
   "mathematical role.")

QA('"Why three separate camera screens instead of one camera with mode switching?"',
   "Each mode has fundamentally different data requirements. Color Identifier takes a photo on "
   "every tap (fast, small). Simulation freezes a photo and re-renders via GPU shader. "
   "Enhancement freezes, decodes to JS buffer, and processes per-pixel. Combining into one screen "
   "would require conditional camera behavior + conditional render trees + conditional state "
   "management — more complex with no user-facing benefit. Three clean screens is more "
   "maintainable.")

QA('"Your paper claims \'adaptive\' daltonization. What\'s adaptive about it?"',
   "Three adaptive behaviors: (1) Self-gating — the algorithm skips pixels that don't need "
   "correction, adapting per-pixel based on error magnitude. (2) CVD type selection — user picks "
   "which deficiency they have, algorithm adapts matrices. (3) Algorithm selection — user picks "
   "Daltonization or Hue Rotation based on which gives better results for THEIR specific "
   "deficiency and THEIR specific image. These are the three levels of adaptation.")

QA('"How did you test correctness? Did you have CVD users validate it?"',
   "Be honest: 'We validated ALGORITHMIC correctness against published reference outputs — our "
   "Viénot simulation matches the Coblis online simulator when both are configured for "
   "severity=1.0 with gamma correction. We have NOT conducted clinical validation with CVD "
   "participants; that is listed as future work. The thesis scope is implementation and "
   "comparative algorithmic analysis, not clinical efficacy.' Never claim unverified user "
   "validation. Panelists respect honest limitations.")

QA('"What happens if the user picks the wrong CVD type? (e.g., Protan button but they are Deutan)"',
   "Partial correction. Protan and Deutan are both red-green deficiencies with overlapping "
   "confusion sets (Red, Orange, Brown in common). A Deutan using Protan mode still gets SOME "
   "benefit — reds get shifted toward yellow, which helps. But it's not optimal. A Tritan using "
   "Protan mode gets no benefit because their confusion sets don't overlap (Yellow, Cyan, Blue, "
   "Pink vs Red, Orange, Violet, Brown). Proper CVD diagnosis (via Ishihara test) is a prerequisite "
   "for optimal correction.")

QA('"Why Viénot 1999 and not the newer Machado 2009?"',
   "Machado offers continuous severity control (0.0 to 1.0), modeling anomalous trichromats who "
   "have SHIFTED cones rather than absent ones. We chose Viénot because: (a) single matrix per "
   "CVD type is simpler to explain and defend; (b) worst-case simulation (severity=1.0) still "
   "benefits milder cases because the correction is a superset; (c) Machado requires a user-facing "
   "severity slider, adding cognitive load for a non-clinical UX. We list Machado as future work "
   "for anomalous trichromacy support.")

QA('"How does your app handle a color the user has never seen? E.g., a teal-turquoise gradient?"',
   "For Color Identifier: the nearest CIELAB reference is 'Cyan' — the app classifies it "
   "correctly into a named class. For Enhancement: Daltonization operates per-pixel independent of "
   "class, so every pixel in the gradient gets individually corrected. The user will see the "
   "gradient preserved but shifted if it was in their confusion zone.")

QA('"The CNN you removed — could you resurrect it as a quality signal?"',
   "Theoretically yes. The CNN outputs per-pixel confidence for each color class. You could use "
   "that confidence to BLEND daltonization — fully daltonize high-confidence confused pixels, "
   "partially daltonize low-confidence ones. This might reduce false-corrections on edge pixels. "
   "But: (a) adds 8MB model + 2s inference time, (b) daltonization's self-gating already handles "
   "this implicitly via error magnitude, and (c) we have no evidence it would improve perceived "
   "quality. Retained the simpler design.")

QA('"Show me the most important single line of code in your entire system."',
   "Depends on framing, but two strong candidates: "
   "(a) App.js:98 — 'return half4(pow(sim, half3(0.4545)), c.a);' — the gamma-correct sRGB encode "
   "that makes Simulation colorimetrically valid. "
   "(b) tensorHelper.js:486 — 'if (useGate && !confusionSet.has(mask[i])) continue;' — the "
   "optional mask that demonstrates self-gating is now just a performance hint, not a correctness "
   "requirement.")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 5 — CHEATSHEET + NIGHT BEFORE
# ═══════════════════════════════════════════════════════════════════
H1('PART 5 — Cheatsheet + Night-Before Rehearsal')

H2('5.1 — One-liner cheatsheet (say on demand)')
TABLE(
    ['Concept', 'One-liner'],
    [
        ['sRGB gamma',          'Non-linear encoding; pixel 128 is not half the light of 255'],
        ['Linear space',        'Where physical models (Viénot, daltonization) live'],
        ['Viénot matrices',     '3×3 projections of LMS space onto dichromatic confusion planes'],
        ['Row sum ≈ 1',         'Preserves grays: (v,v,v) input → (v,v,v) output'],
        ['Daltonization',       'Simulate → subtract → redistribute error to surviving channels'],
        ['Self-gating',         'Error magnitude acts as its own classifier; no CNN needed'],
        ['Hue rotation',        'Rule-based comparative baseline in HSV'],
        ['Linear taper',        'Prevents banding artifacts at rotation band edges'],
        ['CIELAB',              'Perceptually uniform space, D65-referenced'],
        ['Delta-E',             'Perceptual distance in LAB; L* down-weighted for naming'],
        ['Chroma gate 12',      'Separates true grays from noisy low-chroma colors'],
        ['Cover-mode mapping',  'Converts screen tap to image pixel accounting for aspect crop'],
        ['10×10 averaging',     'Reduces sensor noise by √100 = 10× before identifying'],
        ['Freeze pattern',      'Avoids JS per-frame processing impossible at 30fps'],
        ['HAL race guard',      'beforeRemove + activation delay prevents camera open conflict'],
        ['GPU shader',          '~4ms for 1040p vs ~400ms in JS'],
        ['makeImageSnapshot',   'Reads back GPU canvas to save post-shader output'],
        ['decodedRef cache',    'Avoids re-decoding when user switches CVD type or algorithm'],
    ],
    widths=[4, 13]
)

H2('5.2 — What to memorize the night before')

P('Tier 1 — MUST know cold:', bold=True)
NUM('One row of the Protan matrix: [0.152, 1.053, -0.205]')
NUM('The daltonization formula: out = original + E · (original - SIM · original)')
NUM('Why gamma matters: Viénot models linear cone response; sRGB is non-linear')
NUM('The self-gating claim: "Error ≈ 0 for already-perceivable pixels"')
NUM('The three modes in one sentence: "Identifier names, Simulation shows, Enhancement helps"')

P('Tier 2 — Know the numbers:', bold=True)
NUM('Prevalence: Protan 1%, Deutan 5%, Tritan <0.01% of males')
NUM('Error matrix weight: 0.7 (Fidaner empirical optimum)')
NUM('Chroma gate threshold: 12 (separates gray from colored)')
NUM('L* weight in Delta-E: 0.5 (down-weighted so dark-red ≈ bright-red)')
NUM('1040p resize cap (GPU memory + processing budget)')
NUM('10×10 sample patch for Identifier (noise averaging)')

P('Tier 3 — Be ready to narrate end-to-end:', bold=True)
BULLET('The red apple walkthrough (Part 3.4.3 — values 240,30,30 → linear → sim → err → out)')
BULLET('The cover-mode coordinate mapping (Part 1.3 — why screen taps need aspect correction)')
BULLET('The HAL race guard (Part 0.2 — why there is a 700ms delay before camera activates)')

H2('5.3 — Opening statement (rehearse verbatim)')
QUOTE(
    "ReColor is a mobile assistive tool for color vision deficiency, built on three published "
    "algorithms integrated into a React Native application. The simulation mode uses the Viénot-"
    "Brettel-Mollon 1999 matrices to show normal-vision users what a CVD user sees. The enhancement "
    "mode uses Brettel-Fidaner daltonization to redistribute lost color information into channels "
    "the CVD user can perceive, with self-gating via the error term eliminating the need for a "
    "pixel classifier. The color identifier uses CIELAB Delta-E nearest-neighbor classification "
    "against a 77-entry reference database. All three modes share a gamma-correct processing "
    "pipeline, a freeze-then-process pattern for stability, and sRGB-to-linear-space conversion "
    "where physical color models demand it. The contribution is not a new algorithm, but the "
    "correct, mobile-deployed integration of these algorithms as a practical assistive tool."
)

H2('5.4 — If a panelist traps you')
BULLET('If asked a question you do not know: "That is a good question. I did not cover that depth in my implementation, but here is my reasoning..."')
BULLET('If asked about a limitation: acknowledge it immediately. Never defend weaknesses.')
BULLET('If asked about future work: have THREE ready — Machado 2009 severity, clinical user validation, GPU-port of daltonization.')
BULLET('If the panelist tries to lead you into a wrong answer: pause, ask them to clarify. Do not agree under pressure.')

doc.add_paragraph()
closing = doc.add_paragraph()
closing.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = closing.add_run('— End of Reviewer —')
r.italic = True; r.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

# ── Save ──
out_path = r'c:\xampp\htdocs\Evala\ReColor\ReColor_Defense_Reviewer_v2.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
