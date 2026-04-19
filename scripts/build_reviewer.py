"""Generate ReColor_Defense_Reviewer.docx from structured content."""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ── Page margins ──
for section in doc.sections:
    section.top_margin    = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin   = Cm(2)
    section.right_margin  = Cm(2)

# ── Base style ──
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

def H1(text):
    p = doc.add_heading(text, level=1)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x93)
    return p

def H2(text):
    p = doc.add_heading(text, level=2)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x2E, 0x50, 0xB0)
    return p

def H3(text):
    p = doc.add_heading(text, level=3)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x50, 0x70, 0xC0)
    return p

def P(text_or_runs, bold=False, italic=False):
    """Add paragraph. Pass a string, or a list of (text, {style}) tuples."""
    p = doc.add_paragraph()
    if isinstance(text_or_runs, str):
        r = p.add_run(text_or_runs)
        r.bold = bold
        r.italic = italic
    else:
        for text, opts in text_or_runs:
            r = p.add_run(text)
            r.bold   = opts.get('bold', False)
            r.italic = opts.get('italic', False)
            if opts.get('mono'):
                r.font.name = 'Consolas'
                r.font.size = Pt(10)
            if opts.get('color'):
                r.font.color.rgb = RGBColor(*opts['color'])
    return p

def BULLET(text):
    p = doc.add_paragraph(text, style='List Bullet')
    return p

def NUM(text):
    p = doc.add_paragraph(text, style='List Number')
    return p

def CODE(text):
    """Monospaced code block in a shaded paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    r = p.add_run(text)
    r.font.name = 'Consolas'
    r.font.size = Pt(9.5)
    # shading
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F2F2F2')
    p._p.get_or_add_pPr().append(shd)
    return p

def QUOTE(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.75)
    p.paragraph_format.right_indent = Cm(0.75)
    r = p.add_run(text)
    r.italic = True
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return p

def TABLE(headers, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Light Grid Accent 1'
    # header
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(10)
    # body
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[1 + ri].cells[ci]
            cell.text = ''
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(10)
    if widths:
        for row in t.rows:
            for ci, w in enumerate(widths):
                row.cells[ci].width = Cm(w)
    return t

# ═══════════════════════════════════════════════════════════════════
# COVER
# ═══════════════════════════════════════════════════════════════════
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run('ReColor Thesis Defense\nComplete Algorithm Reviewer')
r.bold = True
r.font.size = Pt(22)
r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x93)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run('Viénot Simulation · Brettel/Fidaner Daltonization · Hue Rotation · CIELAB Delta-E')
r.italic = True
r.font.size = Pt(12)
r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_paragraph()
intro = doc.add_paragraph()
intro.add_run(
    "Everything in ReColor lives in three files: the runtime shader in App.js (lines 84-100), "
    "the color science in tensorHelper.js, and the orchestration in the screen components. "
    "You have four distinct algorithms: Viénot CVD Simulation, Brettel/Fidaner Daltonization, "
    "Hue Rotation (comparative), and CIELAB Delta-E Color Identification — all glued together "
    "by sRGB gamma correction."
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 0 — BIOLOGY
# ═══════════════════════════════════════════════════════════════════
H1('PART 0 — Biology Primer (Ground your answers here)')

P('The human eye has three cone photoreceptors, each tuned to different wavelengths:')

TABLE(
    ['Cone', 'Peak λ', 'Common name', 'Deficiency name'],
    [
        ['L', '~564 nm', 'Red-sensing',   'Protanopia (L missing)'],
        ['M', '~534 nm', 'Green-sensing', 'Deuteranopia (M missing)'],
        ['S', '~420 nm', 'Blue-sensing',  'Tritanopia (S missing)'],
    ],
    widths=[2, 3, 4, 6],
)

P([
    ('Dichromacy', {'bold': True}),
    (' = one cone type absent (complete CVD). ', {}),
    ('Anomalous trichromacy', {'bold': True}),
    (' = one cone shifted (partial). ReColor simulates the ', {}),
    ('dichromatic', {'italic': True}),
    (' worst case, which also benefits anomalous trichromats because the compensation is stronger than they strictly need.', {}),
])

P([
    ('Prevalence', {'bold': True}),
    (' (memorize these): Protan ~1% of males, Deutan ~5% of males, Tritan <0.01% (both sexes). '
     'Red-green CVD (Protan + Deutan together) = ~8% of males, ~0.5% of females.', {}),
])

P([
    ('Why red-green is most common: ', {'bold': True}),
    ('L and M cone genes are on the X chromosome → males (XY) have no backup. '
     'S is on chromosome 7 → equally expressed both sexes, but far less common to break.', {}),
])

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 1 — GAMMA
# ═══════════════════════════════════════════════════════════════════
H1('PART 1 — sRGB Gamma Correction (The Foundation)')

P('Every panelist who does image processing will test you on this. Master it.', bold=True)

H3('The problem')
P(
    "A JPEG pixel stored as 128 is NOT half the light of 255. Display hardware and human vision "
    "are both non-linear. sRGB stores pixels in a perceptually-uniform encoding (more precision "
    "for dark tones where the eye is sensitive), so equal numeric steps give equal perceived steps, "
    "not equal light steps."
)

H3('The math (IEC 61966-2-1 standard)')
P('Exact piecewise decode (tensorHelper.js lines 137-139):')
CODE("""if c ≤ 0.04045:  linear = c / 12.92
else:            linear = ((c + 0.055) / 1.055) ^ 2.4""")

P([
    ('Approximation', {'bold': True}),
    (' used in the hot loops and shader: ', {}),
    ('linear = c ^ 2.2', {'mono': True}),
    (' (gamma decode), ', {}),
    ('srgb = linear ^ (1/2.2) = linear ^ 0.4545', {'mono': True}),
    (' (gamma encode). Error vs exact: ', {}),
    ('<1%', {'bold': True}),
    (', and the speed gain is ~3× on millions of pixels.', {}),
])

H3('Why we use exact in Color Identifier but approximate in Simulation/Daltonization')
BULLET('Color ID matches pixels against reference LAB values — any systematic error shifts class assignment, so we use the exact piecewise function.')
BULLET('Simulation / Daltonization runs on every pixel at interactive rates; a <1% pow-2.2 error is visually undetectable because the output gets re-encoded back anyway.')

H3('If a panelist asks: "why does gamma matter for CVD simulation?"')
QUOTE(
    "The Viénot matrices were derived against LINEAR light — they model photon absorption by cones, "
    "which is linear. Applying them directly to sRGB values (as many open-source implementations do) "
    "produces a washed-out, weakened simulation, especially for protanopia and tritanopia. Our shader "
    "decodes to linear, applies the matrix, clamps to [0,1], then re-encodes. That is the single most "
    "important correctness decision in the simulation path."
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 2 — CVD SIMULATION
# ═══════════════════════════════════════════════════════════════════
H1('PART 2 — CVD Simulation (Viénot-Brettel-Mollon 1999)')

H3('Location')
BULLET('Matrices: tensorHelper.js lines 49-65 (CVD_COMBINED)')
BULLET('Row accessor: tensorHelper.js lines 105-109 (getCVDRows)')
BULLET('Shader: App.js lines 84-100')

H3('The matrices (MEMORIZE one row of one matrix — panelists love specifics)')

P('Protan (no L cone):', bold=True)
CODE("""[ 0.152286  1.052583  -0.204868]    R_out
[ 0.114503  0.786281   0.099216]    G_out
[-0.003882 -0.048116   1.051998]    B_out""")

P('Deutan (no M cone):', bold=True)
CODE("""[ 0.367322  0.860646  -0.227968]
[ 0.280085  0.672501   0.047413]
[-0.011820  0.042940   0.968881]""")

P('Tritan (no S cone):', bold=True)
CODE("""[ 1.255528 -0.076749  -0.178779]
[-0.078411  0.930809   0.147602]
[ 0.004733  0.691367   0.303900]""")

H3('What each number means')
QUOTE(
    "Row 0 says: 'The red channel the CVD user perceives = 0.15 × original_R + 1.05 × original_G "
    "− 0.20 × original_B.' For Protan, MOST of the perceived red comes from the green channel — "
    "that's the defining feature of protanopia: they read red values through their M-cone response."
)

H3('Why the rows ≈ sum to 1')
P([
    ('Each row sum ≈ 1.0 (Protan row 0: 0.152 + 1.053 − 0.205 = ', {}),
    ('1.000', {'bold': True}),
    ('). This preserves luminance on the achromatic axis (grays stay gray). A panelist asking '
     '"what happens to a pure gray pixel?" — you can answer: gray → gray, because any (v,v,v) '
     'input multiplied by row-sum 1 returns v.', {}),
])

H3('Why some entries are negative')
P(
    "The matrices are derived by projecting the dichromatic confusion axis onto the colorimetric "
    "subspace spanned by two remaining cones. Geometrically it's a projection onto a plane in LMS "
    "space; negative coefficients are mathematically valid (they're the projection coefficients), "
    "though they can push intermediate values outside [0,1] — which is why the shader has "
    "clamp(sim, 0, 1) on line 96."
)

H3('How Viénot differs from Machado 2009')
BULLET('Viénot 1999 = complete dichromacy only (severity = 1.0, total cone loss)')
BULLET('Machado 2009 = continuous severity parameter (0.0 → 1.0), models anomalous trichromacy too')

P([
    ('ReColor uses Viénot because: (a) simpler — one matrix per CVD type; (b) worst-case '
     'simulation also benefits mild cases; (c) Machado\'s tables are 11× larger (one matrix per '
     'severity level). ', {}),
    ('If a panelist says "why not Machado?" ', {'bold': True}),
    ('— tell them you chose Viénot for interpretability and to keep the defense surface small; '
     'adding Machado is listed as future work and would require the user to self-report severity.', {}),
])

H3('The SkSL shader line-by-line (App.js 84-100)')
CODE("""uniform shader contents;       // input texture (the frozen photo)
uniform half3 row0;            // row 0 of CVD matrix as a 3-component vector
uniform half3 row1;
uniform half3 row2;

half4 main(float2 coord) {
  half4 c = contents.eval(coord);         // sample input pixel (sRGB)
  half3 lin = pow(c.rgb, half3(2.2));     // sRGB → linear (gamma decode)
  half3 sim = half3(                       // apply 3x3 matrix as three dot products
    dot(row0, lin),
    dot(row1, lin),
    dot(row2, lin));
  sim = clamp(sim, half3(0.0), half3(1.0)); // clip out-of-gamut projections
  return half4(pow(sim, half3(0.4545)), c.a); // linear → sRGB (gamma encode)
}""")

P([
    ('Why GPU shader and not JS loop? ', {'bold': True}),
    ('1080×1920 pixels = 2 million ops per frame at 30fps = 60M pow() calls/sec. JS would drop to '
     '1-2 fps. The GPU does it in parallel on every texture sample.', {}),
])

P([
    ('Why half (16-bit float) and not float? ', {'bold': True}),
    ('Mobile GPU perf; visually indistinguishable for 8-bit display output.', {}),
])

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 3 — DALTONIZATION
# ═══════════════════════════════════════════════════════════════════
H1('PART 3 — Daltonization (Brettel / Fidaner RGB variant)')

H3('Location')
BULLET('Pipeline: tensorHelper.js lines 473-534 (applyDaltonization)')
BULLET('Simulation matrix: same CVD_COMBINED as above')
BULLET('Error redistribution: tensorHelper.js lines 116-132 (CVD_ERR_SHIFT)')

H3('The three-step idea')
P('Daltonization is an assistive INVERSE of simulation. The intuition:')

NUM('Simulate what the CVD user sees: sim = SIM · original')
NUM('Compute the error — the information that got lost: err = original − sim')
NUM('Redistribute that error into channels the user CAN see: out = original + E · err')

P(
    "If a pixel was already perceivable (red = red for a normal observer = red for a protan — that's "
    "fine), the error err ≈ 0 and the pixel passes through unchanged. If the pixel was in a confusion "
    "zone, the error is large, and you shift it into the surviving channels so the CVD user perceives "
    "a distinguishable hue."
)

H3('The error-shift matrices (MEMORIZE this for Protan)')
CODE("""Protan E = [ 0.0  0.0  0.0 ]    ← R stays: they can't see it anyway
           [ 0.7  1.0  0.0 ]    ← G gets 70% of R error + 100% of G error
           [ 0.7  0.0  1.0 ]    ← B gets 70% of R error + 100% of B error""")

QUOTE(
    "Interpretation: 'For a Protan, the R channel is useless to redistribute INTO (row 0 is zero), "
    "but the R error (lost red info) is pushed into G and B at 70% weight. Reds become perceptibly "
    "tinted toward yellow (R+G) and magenta (R+B), which Protans CAN distinguish from pure green.'"
)

CODE("""Deutan E = [ 1.0  0.6  0.0 ]    ← R gets 60% of G error
           [ 0.0  0.0  0.0 ]    ← G useless for them
           [ 0.0  0.6  1.0 ]    ← B gets 60% of G error

Tritan E = [ 1.0  0.0  0.7 ]    ← R gets 70% of B error
           [ 0.0  1.0  0.7 ]    ← G gets 70% of B error
           [ 0.0  0.0  0.0 ]    ← B useless""")

P([
    ('Rule: ', {'bold': True}),
    ('the row corresponding to the missing cone\'s channel is zero (no point redistributing there). '
     'The 0.7 vs 1.0 is empirical — higher than 1 oversaturates, lower than 0.5 doesn\'t help enough. '
     'Fidaner\'s original paper suggests these as starting values; we kept them.', {}),
])

H3('The full loop (tensorHelper.js 485-531)')
CODE("""for (let i = 0; i < numPixels; i++) {
  const r = pow(pixels[rIdx]/255, 2.2);                    // gamma decode
  const g = pow(pixels[rIdx+1]/255, 2.2);
  const b = pow(pixels[rIdx+2]/255, 2.2);

  const rSim = SIM[0][0]*r + SIM[0][1]*g + SIM[0][2]*b;    // simulate
  const gSim = SIM[1][0]*r + SIM[1][1]*g + SIM[1][2]*b;
  const bSim = SIM[2][0]*r + SIM[2][1]*g + SIM[2][2]*b;

  const rErr = r - rSim, gErr = g - gSim, bErr = b - bSim; // error

  const rOut = r + ERR[0][0]*rErr + ERR[0][1]*gErr + ERR[0][2]*bErr;
  const gOut = g + ERR[1][0]*rErr + ERR[1][1]*gErr + ERR[1][2]*bErr;
  const bOut = b + ERR[2][0]*rErr + ERR[2][1]*gErr + ERR[2][2]*bErr;

  pixels[rIdx]   = pow(clamp01(rOut), 1/2.2) * 255;        // gamma encode
  pixels[rIdx+1] = pow(clamp01(gOut), 1/2.2) * 255;
  pixels[rIdx+2] = pow(clamp01(bOut), 1/2.2) * 255;
}""")

H3('The self-gating property (CRITICAL — be ready for this)')

P([
    ('The mask parameter is ', {}),
    ('optional', {'bold': True}),
    (' (tensorHelper.js lines 473-482). We removed the CNN classifier because the error term ', {}),
    ('IS its own gate', {'bold': True}),
    (':', {}),
])

BULLET('For a pixel that is neither red nor green (e.g., pure blue for a Protan), sim ≈ original, so err ≈ 0, so out = original + E·0 = original. The pixel is untouched without any classifier needed.')
BULLET('For a pixel in the confusion zone, err is large, and the correction kicks in proportionally.')

P(
    "This is the single most clever thing to claim in your defense. Most papers use a classifier "
    "to decide which pixels to daltonize. Ours doesn't need one because the math already knows.",
    bold=True
)

H3('Why we operate in linear space (and Fidaner did too)')
P(
    "Because simulation is a linear model of cone photon absorption. Applying it in sRGB gamma space "
    "means you're applying a physical model to non-physical values → the error estimate is wrong → "
    "the redistribution is wrong. This is the same gamma concern from Part 1, applied to a different "
    "algorithm."
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 4 — HUE ROTATION
# ═══════════════════════════════════════════════════════════════════
H1('PART 4 — Hue Rotation (Comparative Algorithm)')

H3('Location')
BULLET('Config: tensorHelper.js lines 543-550')
BULLET('Implementation: tensorHelper.js lines 557-629')

H3('Why this exists')
P(
    "It's the comparative baseline. Your paper compares Daltonization (physically-motivated, "
    "error-redistribution) against Hue Rotation (rule-based, hue-manipulation). This gives you a "
    "quantitative claim: 'Daltonization preserves more of the original color identity than "
    "rotation-based methods.'"
)

H3('The HSV color model (know this cold)')
BULLET('H (Hue, 0-360°) — pure color: 0°=red, 60°=yellow, 120°=green, 180°=cyan, 240°=blue, 300°=magenta.')
BULLET('S (Saturation, 0-1) — colorfulness vs gray.')
BULLET('V (Value, 0-1) — brightness.')
P('HSV is chosen because hue is a single parameter that wraps at 360° — perfect for "rotate the color wheel."')

H3('The config (tensorHelper.js 543-550)')
CODE("""Protan: { center:   0, range: 60, shift:  40 }
Deutan: { center:   0, range: 60, shift:  40 }
Tritan: { center: 240, range: 60, shift: -30 }""")

P('Parameter meanings:', bold=True)
BULLET('center = the hue where confusion is strongest. For Protan/Deutan, reds (0°) collapse into greens. For Tritan, blues (240°) collapse into cyan.')
BULLET('range = half-width of the band. 60° means hues 300°→60° (wrapping through 0°) get rotated. Outside that band: untouched.')
BULLET('shift = rotation amount AT band center, linearly tapered to 0 at the edges. Protan 40° = "pure red shifts toward yellow (0° → 40°)."')

H3('Circular distance (handles the 360° wrap) — tensorHelper.js 552-555')
CODE("""function circularHueDistance(h, center) {
  const d = Math.abs(h - center);
  return d > 180 ? 360 - d : d;
}""")
P(
    "Without this, the hue at 350° would appear 'far' from center=0° when it's actually only 10° "
    "away on the wheel."
)

H3('Linear taper (tensorHelper.js 591-592)')
CODE("""const weight = 1 - dist / cfg.range;
let newH = h + cfg.shift * weight;""")

P(
    "A pixel exactly at center gets the full shift. A pixel at the band edge gets zero shift. This "
    "avoids BANDING ARTIFACTS — if you applied a constant shift within the band and nothing outside, "
    "viewers would see a hard color discontinuity at the band boundary."
)

H3('The saturation gate (tensorHelper.js 563, 578)')
CODE("""const SAT_MIN = 0.15;
if (s < SAT_MIN) continue;""")
P(
    "Near-gray pixels have an unreliable hue (tiny noise in RGB produces huge hue swings). Rotating "
    "the 'hue' of a gray pixel produces colored noise. Skipping them preserves neutrals."
)

H3('Why hue rotation is weaker than daltonization (rehearse this comparison)')
TABLE(
    ['Property', 'Daltonization', 'Hue Rotation'],
    [
        ['Physical basis',           'Cone-response model (Viénot)', 'Heuristic rule'],
        ['Self-gating',              'Yes (via error magnitude)',    'No (manual band definition)'],
        ['Preserves achromatic axis','Yes',                          'Yes (via sat gate)'],
        ['Handles all pixel tones',  'Yes',                          'No (band-limited)'],
        ['Color identity preservation','High (only shifts what is lost)', 'Low (shifts entire band)'],
    ],
)

P([
    ('Scripted answer if asked "why keep a weaker algorithm?": ', {'bold': True}),
])
QUOTE(
    "Hue rotation is the comparative baseline. Without it, the paper has no quantitative evidence "
    "that daltonization is the better choice. Comparative analysis against a representative of the "
    "rule-based family is methodologically required."
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 5 — COLOR IDENTIFIER
# ═══════════════════════════════════════════════════════════════════
H1('PART 5 — Color Identifier (CIELAB + Delta-E)')

H3('Location')
BULLET('LAB conversion: tensorHelper.js lines 148-170')
BULLET('Delta-E: tensorHelper.js lines 177-184')
BULLET('Reference database: tensorHelper.js lines 195-298')
BULLET('Main entry: tensorHelper.js lines 309-336')

H3('Why CIELAB and not RGB distance?')
P(
    "RGB Euclidean distance doesn't match human perception. Two pairs of colors with equal RGB "
    "distance can look wildly different in perceptual distance. CIELAB is approximately perceptually "
    "uniform — equal distances ≈ equal perceived differences."
)

H3('sRGB → LAB pipeline (tensorHelper.js 148-170)')

P('Step 1: sRGB → linear RGB (exact IEC piecewise, not the 2.2 approx).', bold=True)

P('Step 2: linear RGB → XYZ using the D65 matrix:', bold=True)
CODE("""X = 0.4124564·R + 0.3575761·G + 0.1804375·B
Y = 0.2126729·R + 0.7151522·G + 0.0721750·B
Z = 0.0193339·R + 0.1191920·G + 0.9503041·B""")

P([
    ('Normalized by the D65 white point ', {}),
    ('(0.95047, 1.0, 1.08883)', {'mono': True}),
    (' so that reference white = (1,1,1) in XYZ — required for the LAB formula.', {}),
])

P('Step 3: XYZ → LAB using the nonlinear f() function:', bold=True)
CODE("""f(t) = t^(1/3)           if t > 0.008856
f(t) = 7.787·t + 16/116  otherwise

L = 116·f(Y) − 16       (lightness, 0-100)
a = 500·(f(X) − f(Y))   (green-red axis, roughly ±128)
b = 200·(f(Y) − f(Z))   (blue-yellow axis, roughly ±128)""")

P([
    ('D65', {'bold': True}),
    (' = standard daylight illuminant at 6504K color temperature. Why this specific white point? '
     'sRGB is defined relative to D65; using any other white would shift everything.', {}),
])

H3('Weighted Delta-E (tensorHelper.js 177-184)')
CODE("""const L_WEIGHT = 0.5;
function deltaE(lab1, lab2) {
  return Math.sqrt(
    L_WEIGHT * (lab1[0] - lab2[0])² +
               (lab1[1] - lab2[1])² +
               (lab1[2] - lab2[2])²
  );
}""")

P([
    ('Why down-weight L*? ', {'bold': True}),
    ('We want "dark red" and "bright red" to both identify as "Red." L* is brightness; a* and b* '
     'encode hue/chroma (the identity of the color). Full weight on L* would put "dark red" closer '
     'to "dark green" than to "bright red" — obviously wrong for naming.', {}),
])

P('Delta-E interpretation scale (cite this if asked):', bold=True)
BULLET('< 1.0 — imperceptible to the human eye')
BULLET('< 2.0 — perceptible only by close inspection')
BULLET('2-10 — noticeable at a glance')
BULLET('10-49 — different colors')
BULLET('100 — colors are exact opposites')

H3('Chroma gate (tensorHelper.js 193, 314)')
CODE("""const NEUTRAL_CHROMA_THRESHOLD = 12;
const chroma = Math.sqrt(lab[1]² + lab[2]²);
const isChromatic = chroma >= 12;""")

P([
    ('Chroma ', {'bold': True}),
    ('C* = sqrt(a² + b²)', {'mono': True}),
    (' is the distance from the neutral axis. Near-neutral pixels (low chroma) must match against '
     'black/gray/white reference entries, not colored ones. Without this gate, a dim gray wall could '
     'be classified as "dark red" because its small a*/b* noise happens to be positive red. The gate '
     'forces: "if chroma < 12, you\'re gray — match only against neutral entries."', {}),
])

H3('Confidence score (tensorHelper.js 333)')
CODE('const confidence = Math.max(0, Math.round(100 - bestDist * 2));')

P(
    "Linear scaling: Delta-E 0 = 100% confidence, Delta-E 50+ = 0%. The factor 2 is calibrated so "
    "that Delta-E 10 (noticeable difference) = 80% confidence, which matches the user-perceived "
    "reliability on testing."
)

H3('The reference database (77 entries across 10 classes)')
P('Design decisions:', bold=True)
NUM('Multiple entries per class — "Red" has 9 reference points covering saturated, muted, dark, and soft variants. A pixel finds the nearest of those 9; they all map to class "Red."')
NUM('Named variants (Crimson, Firebrick, Indian Red) — labels pre-computed so the UI can show "Crimson" for a specifically-crimson pixel.')
NUM('Classes chosen from CVD literature — the 10 classes match the confusion-set definitions in CONFUSION_CLASSES (which was originally for the CNN).')

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 6 — SYSTEM-LEVEL DECISIONS
# ═══════════════════════════════════════════════════════════════════
H1('PART 6 — System-Level Decisions (panelists love these)')

H3('Why freeze-then-process for both Simulation and Enhancement?')
P(
    "Processing 2 million pixels in JavaScript per frame is impossible at 30fps. Daltonization on "
    "mobile JS takes ~400ms for a 1040p image. Running it live would give 2.5fps (unusable) and "
    "drain battery. Freeze-then-process gives: (a) interactive preview of a still frame, (b) instant "
    "CVD-type switching on the cached decoded pixels, (c) save-quality output without compromise."
)

H3('Why 1040p resize before processing?')
BULLET('Quality ceiling: at 1040p, individual pixels are imperceptible at normal viewing distance on a phone screen.')
BULLET('Processing budget: 1040×780 = ~0.8M pixels → daltonization in ~400ms. At 1080p full-res (2M pixels) it\'s 1s — borderline unacceptable UX.')
BULLET('Memory: Uint8Array of 2M·4bytes = 8MB per copy; processing creates 2-3 copies. At 4k (33MB/copy), risk of OOM on low-end devices.')

H3('Why not use the CNN we originally built?')
P(
    "We removed it. The self-gating property of daltonization (error ≈ 0 for correctly-perceivable "
    "pixels) makes the classifier redundant. Removing it: (a) cut processing time 7× (2.8s → 0.4s), "
    "(b) eliminated a ~8MB model file, (c) removed a failure mode (classifier misclassification "
    "overriding correct daltonization)."
)

QUOTE(
    "If asked: 'We initially used a CNN segmentation model. During testing we observed the "
    "error-redistribution math itself gates the correction. Removing the CNN improved both speed "
    "and correctness. The current implementation uses daltonization's mathematical self-gating "
    "instead of external classification.'"
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 7 — PANELIST QUESTIONS
# ═══════════════════════════════════════════════════════════════════
H1('PART 7 — Anticipated Panelist Questions + Scripted Answers')

def QA(q, a):
    p = doc.add_paragraph()
    r = p.add_run(q)
    r.bold = True
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x93)
    QUOTE(a)

QA(
    'Q1: "This isn\'t novel — these matrices are from 1999."',
    "Correct. ReColor's contribution is NOT a new color-science algorithm; it's the INTEGRATION, "
    "OPTIMIZATION, and DEPLOYMENT of published algorithms as a mobile real-time assistive tool. "
    "Specific contributions: (1) gamma-correct GPU shader implementation of Viénot matrices — many "
    "open-source implementations omit this and produce weakened output; (2) empirical validation "
    "that daltonization's error-redistribution self-gates, removing the need for a pixel classifier; "
    "(3) comparative analysis framework pitting physically-motivated daltonization against "
    "rule-based hue rotation."
)

QA(
    'Q2: "How did you validate that it actually works for CVD users?"',
    "Be honest: 'We validated correctness against published reference images — our gamma-corrected "
    "output matches Viénot et al.'s original plates and the Coblis CVD simulator. Clinical "
    "validation with CVD participants is listed as future work; the thesis scope is technical "
    "implementation and comparative algorithmic analysis.' If pushed: 'The gold-standard validation "
    "is Ishihara plate testing with actual CVD participants; we acknowledge this as a limitation.'"
)

QA(
    'Q3: "Why linear space? JPEGs are sRGB already."',
    "Because the CVD matrices are derived from a physical model of cone photon absorption, which is "
    "linear. Applying them to sRGB gamma-encoded values means applying a physical model to "
    "non-physical values — the simulation is weakened because dark tones get under-transformed and "
    "bright tones get over-transformed. This is why our shader explicitly decodes to linear, applies "
    "the matrix, then re-encodes to sRGB on output."
)

QA(
    'Q4: "What if the panelist tests a gray pixel?"',
    "Gray passes through unchanged. Viénot matrix rows sum to approximately 1.0, so (v,v,v) input "
    "returns (v,v,v) output regardless of CVD type. The daltonization error is zero for grays "
    "because simulated gray equals original gray. The hue rotation skips them via the saturation "
    "gate (s < 0.15)."
)

QA(
    'Q5: "Why 0.7 in the error matrix and not 0.5 or 1.0?"',
    "Those weights come from Fidaner's original RGB daltonization paper. 1.0 is oversaturated — "
    "pixels become unnaturally colored. Below 0.5 is under-corrected — the CVD user still can't "
    "discriminate the confusing colors. 0.7 is the empirical sweet spot preserving identity while "
    "restoring discriminability."
)

QA(
    'Q6: "Why only three CVD types? There\'s anomalous trichromacy too."',
    "We model full dichromacy (Viénot severity = 1.0) as worst case. Anomalous trichromats "
    "(protanomaly, deuteranomaly, tritanomaly) see somewhere between normal and dichromatic; the "
    "correction helps them but is stronger than strictly needed. Adding severity control would "
    "require Machado 2009 matrices — future work."
)

QA(
    'Q7: "Your Color Identifier uses a hardcoded 77-entry database. Why not machine learning?"',
    "Three reasons. (1) A reference-database approach is deterministic and explainable — panelists "
    "can inspect the database and verify classification. (2) Training a color classifier requires a "
    "labeled dataset; our 10 classes are defined from color-vision literature and directly match the "
    "CVD confusion sets. (3) CIELAB Delta-E is perceptually uniform — the same metric used in "
    "color-science standards (ISO 12647, Delta-E 2000). An ML classifier would be a black box "
    "offering no accuracy gain for such a well-defined problem."
)

QA(
    'Q8: "What\'s the computational complexity?"',
    "All pixel-level algorithms are O(W·H) — single pass, constant work per pixel. "
    "Simulation on GPU: ~16ms per frozen 1040p image. "
    "Daltonization in JS: ~400ms per 1040p image (~0.2µs per pixel). "
    "Hue rotation in JS: ~300ms (lighter per-pixel math, no matrix multiply). "
    "Color ID: O(1) — single pixel queried against 77-entry DB = 77 Delta-E evals = <1ms."
)

QA(
    'Q9: "Why freeze? Why not live AR?"',
    "JavaScript cannot daltonize 2 million pixels at 30fps on a phone. The GPU shader CAN run "
    "Simulation live — we initially implemented that — but the React Native Skia live frame "
    "processor introduces stability issues on Android (HAL race conditions with the VisionCamera "
    "library). The freeze pattern trades the live preview for: (a) reliable operation, (b) instant "
    "CVD-type switching on cached pixels, (c) save-quality output, (d) reduced battery drain."
)

QA('Q10: "What are the limitations?"',
    "Rehearse these — panelists LOVE when you volunteer your own limitations:")
BULLET('Freeze-frame only — no live preview of daltonization')
BULLET('Viénot models full dichromacy; anomalous trichromacy gets over-corrected')
BULLET('Reference database in Color ID is English-only; no localization')
BULLET('No user calibration — all Protans get the same correction')
BULLET('Error matrices (the 0.7 / 0.6 weights) are literature-standard, not personalized')
BULLET('No clinical validation with actual CVD users — future work')
BULLET('Processing up to 1040p max; higher resolutions downsampled')

QA(
    'Q11: "Why gamma 2.2 and not 2.4 with offset?"',
    "2.2 is a power-law approximation to the IEC 61966-2-1 piecewise curve. The piecewise curve "
    "uses 2.4 with an offset and a linear segment near zero. Approximation error is under 1%, "
    "visually undetectable for 8-bit output. In the Color Identifier path we use the exact piecewise "
    "curve because small LAB errors shift class assignments; in the simulation hot loop we use 2.2 "
    "because speed matters and the output is re-encoded anyway."
)

QA(
    'Q12: "What\'s the difference between Simulation and Enhancement?"',
    "Opposite directions. Simulation shows a normal-vision user WHAT A CVD USER SEES (educational, "
    "accessibility testing). Enhancement (Daltonization) shows a CVD user AN IMAGE MODIFIED SO THEY "
    "CAN DISTINGUISH COLORS they normally confuse (assistive). Same underlying matrices, inverse "
    "use case."
)

QA(
    'Q13: "What would break your system?"',
    "Low-light input: noise in dark pixels can produce unreliable hue, shifting Color ID "
    "classifications. Monochromatic input (e.g., photo of a sunset): works mathematically but offers "
    "no enhancement because there's no confusion to resolve. Images with heavy JPEG compression "
    "artifacts: color banding in gradient regions gets amplified by daltonization's error "
    "redistribution. Screen color calibration: if the user's display isn't sRGB-calibrated, the "
    "output is technically wrong — but this affects all color-managed software equally."
)

QA(
    'Q14: "Walk me through what happens when a Protan user daltonizes a red apple."',
    "(1) The pixel (240, 30, 30) is gamma-decoded to linear (0.84, 0.013, 0.013). "
    "(2) Viénot Protan matrix simulates: R_sim = 0.15·0.84 + 1.05·0.013 − 0.20·0.013 ≈ 0.125. "
    "The Protan sees red as very dark. "
    "(3) Error: R_err = 0.84 − 0.125 = 0.715, huge. "
    "(4) Redistribution: Protan E has [0.7, 0.7] in the G-row-R and B-row-R entries. So R_out stays "
    "(E row 0 is zero), G_out = 0.013 + 0.7·0.715 ≈ 0.513, B_out = 0.013 + 0.7·0.715 ≈ 0.513. "
    "(5) Gamma-encode → the pixel appears yellow-pinkish to a normal observer, but to the Protan, "
    "the added G and B content is perceivable where the pure red was not — distinguishing the apple "
    "from green foliage."
)

QA(
    'Q15: "You mentioned confusion classes — how were they defined?"',
    "Protan confuses Red, Orange, Violet, Brown — all colors with significant red content that "
    "collapses into darkness. Deutan confuses Red, Orange, Green, Brown — the classic red-green "
    "axis. Tritan confuses Yellow, Cyan, Blue, Pink — anything with a blue component. These sets "
    "come from ophthalmological color-vision literature (Ishihara test design, Farnsworth D-15) and "
    "are used in the original CNN gate; with the self-gating removal they're documentation of which "
    "colors benefit most from the correction."
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 8 — CHEATSHEET
# ═══════════════════════════════════════════════════════════════════
H1('PART 8 — Quick-Reference Cheatsheet')
P('One-liners to say on demand.', italic=True)

TABLE(
    ['Concept', 'One-liner'],
    [
        ['Viénot matrices',        '3×3 linear projections of LMS color space onto the dichromatic confusion subspace'],
        ['Why gamma correct',      'Matrices model linear cone response; sRGB is non-linear'],
        ['Daltonization strategy', 'Simulate → extract error → redistribute error to working channels'],
        ['Self-gating',            'Error magnitude acts as its own classifier; no CNN needed'],
        ['Hue rotation role',      'Rule-based comparative baseline'],
        ['CIELAB',                 'Perceptually uniform color space, D65-referenced'],
        ['Delta-E',                'Perceptual distance in LAB space'],
        ['L* weight 0.5',          'Down-weights lightness so "dark red" and "bright red" both match "Red"'],
        ['Chroma gate 12',         'Separates truly achromatic pixels from low-chroma colored ones'],
        ['0.7 error weight',       "Fidaner's empirical optimum — avoids oversaturation"],
        ['Freeze pattern',         "JS can't process 2M pixels/frame at 30fps"],
    ],
    widths=[5, 11],
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════
# PART 9 — NIGHT BEFORE
# ═══════════════════════════════════════════════════════════════════
H1('PART 9 — What to Rehearse the Night Before')

P('Memorize, in order:', bold=True)

NUM('"The four algorithms": Viénot Simulation, Brettel/Fidaner Daltonization, Hue Rotation (comparative), CIELAB Delta-E Identification.')
NUM('One row of the Protan matrix: [0.152, 1.053, -0.205] → "protan perceived red is 15% of original red plus 105% of green minus 20% of blue — most red information is read through the green channel."')
NUM('The daltonization formula: out = original + E·(original − SIM·original).')
NUM('Why gamma matters: cone response is linear; sRGB is non-linear.')
NUM('The self-gating claim: "Error is zero for already-perceivable pixels."')
NUM('One CIELAB fact: L* is 0-100 lightness; a* and b* are red-green and blue-yellow axes.')
NUM('Your limitations list (Q10) — say them BEFORE the panel does.')

P([
    ('If you can narrate ', {}),
    ('Q14 (the red apple walkthrough) ', {'bold': True}),
    ('end-to-end from memory, you\'ve mastered daltonization. If you can explain ', {}),
    ('why gamma matters (Q3) ', {'bold': True}),
    ('without the file open, you\'ve mastered the numerical pipeline. Those two are the hardest '
     'and the ones panelists will lean on.', {}),
])

doc.add_paragraph()
closing = doc.add_paragraph()
closing.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = closing.add_run('— End of Reviewer —')
r.italic = True
r.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

# ── Save ──
out_path = r'c:\xampp\htdocs\Evala\ReColor\ReColor_Defense_Reviewer.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
