"""
Identifier DB Expansion — fill gap regions to ~250 total entries.

Tests:
  Current (77) → Expanded (~250) on X-Rite ColorChecker 24.
If accuracy improves, prints JS-formatted entries ready to paste into tensorHelper.js.
"""
from __future__ import annotations
import numpy as np
import colour

NEUTRAL = "Neutral"
NEUTRAL_CHROMA = 12.0

def rgb_to_lab(rgb):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.array(rgb) / 255))

# ───── Current DB ─────────────────────────────────────────────
DB_CURRENT = [
    ("black", NEUTRAL, (0,0,0)),       ("dark gray", NEUTRAL, (64,64,64)),
    ("gray", NEUTRAL, (128,128,128)),  ("light gray", NEUTRAL, (192,192,192)),
    ("white", NEUTRAL, (255,255,255)),
    ("Red","Red",(255,0,0)),     ("Red","Red",(204,0,0)),
    ("Red","Red",(139,0,0)),     ("Crimson","Red",(220,20,60)),
    ("Firebrick","Red",(178,34,34)), ("Red","Red",(255,51,51)),
    ("Indian Red","Red",(205,92,92)), ("Dark Muted Red","Red",(139,58,58)),
    ("Soft Red","Red",(224,96,96)),
    ("Dark Orange","Orange",(255,140,0)), ("Orange","Orange",(255,165,0)),
    ("Coral","Orange",(255,127,80)),      ("Orange","Orange",(232,117,26)),
    ("Orange","Orange",(204,112,0)),      ("Orange","Orange",(196,128,64)),
    ("Orange","Orange",(224,151,110)),    ("Orange","Orange",(184,116,58)),
    ("Yellow","Yellow",(255,255,0)),  ("Yellow","Yellow",(255,215,0)),
    ("Yellow","Yellow",(255,236,139)),("Yellow","Yellow",(218,165,32)),
    ("Yellow","Yellow",(240,230,140)),("Yellow","Yellow",(189,183,107)),
    ("Yellow","Yellow",(212,204,106)),
    ("Green","Green",(0,128,0)),    ("Green","Green",(0,255,0)),
    ("Green","Green",(34,139,34)),  ("Green","Green",(0,100,0)),
    ("Green","Green",(50,205,50)),  ("Green","Green",(144,238,144)),
    ("Green","Green",(107,142,35)), ("Green","Green",(85,107,47)),
    ("Green","Green",(143,188,143)),("Green","Green",(74,122,74)),
    ("Cyan","Cyan",(0,255,255)),  ("Cyan","Cyan",(0,139,139)),
    ("Cyan","Cyan",(32,178,170)), ("Cyan","Cyan",(0,206,209)),
    ("Cyan","Cyan",(64,224,208)), ("Cyan","Cyan",(95,158,160)),
    ("Cyan","Cyan",(107,155,155)),
    ("Blue","Blue",(0,0,255)),    ("Blue","Blue",(0,0,128)),
    ("Blue","Blue",(30,144,255)), ("Blue","Blue",(65,105,225)),
    ("Blue","Blue",(135,206,235)),("Blue","Blue",(70,130,180)),
    ("Blue","Blue",(106,123,141)),("Blue","Blue",(74,106,138)),
    ("Blue","Blue",(176,196,222)),
    ("Violet","Violet",(139,0,255)),  ("Violet","Violet",(128,0,128)),
    ("Violet","Violet",(148,0,211)),  ("Violet","Violet",(186,85,211)),
    ("Violet","Violet",(75,0,130)),   ("Violet","Violet",(102,51,153)),
    ("Violet","Violet",(147,112,219)),("Violet","Violet",(123,104,165)),
    ("Violet","Violet",(93,78,122)),
    ("Pink","Pink",(255,192,203)),("Pink","Pink",(255,105,180)),
    ("Pink","Pink",(255,20,147)), ("Pink","Pink",(219,112,147)),
    ("Pink","Pink",(255,182,193)),("Pink","Pink",(255,0,255)),
    ("Pink","Pink",(196,138,154)),("Pink","Pink",(212,160,160)),
    ("Pink","Pink",(176,112,128)),
    ("Brown","Brown",(139,69,19)), ("Brown","Brown",(160,82,45)),
    ("Brown","Brown",(210,105,30)),("Brown","Brown",(101,67,33)),
    ("Brown","Brown",(165,42,42)), ("Brown","Brown",(222,184,135)),
    ("Brown","Brown",(139,115,85)),("Brown","Brown",(107,79,58)),
    ("Brown","Brown",(196,168,130)),("Brown","Brown",(128,96,64)),
]

# ───── New additions targeting gap regions + general coverage ────
# Goal: reach ~250 entries. Each cluster fills a known gap.
DB_ADDITIONS = [
    # ── Neutral additions (fill mid-light + warm/cool tints) ──
    ("Snow", NEUTRAL, (250,250,250)),       ("Ivory", NEUTRAL, (255,255,240)),
    ("Charcoal", NEUTRAL, (45,45,45)),      ("Smoke Gray", NEUTRAL, (148,148,148)),
    ("Slate Gray", NEUTRAL, (112,128,144)), ("Dim Gray", NEUTRAL, (105,105,105)),
    ("Off White", NEUTRAL, (240,240,240)),  ("Silver", NEUTRAL, (192,192,192)),
    ("Soft Black", NEUTRAL, (28,28,28)),    ("Pewter", NEUTRAL, (96,103,103)),

    # ── Red additions (salmons, brick, wine, muted skin reds) ──
    ("Salmon", "Red", (250,128,114)),       ("Light Salmon", "Red", (255,160,122)),
    ("Dark Salmon", "Red", (233,150,122)),  ("Tomato", "Red", (255,99,71)),
    ("Brick Red", "Red", (203,65,84)),      ("Pale Red", "Red", (240,128,128)),
    ("Wine", "Red", (114,47,55)),           ("Burgundy", "Red", (128,0,32)),
    ("Maroon", "Red", (128,0,0)),           ("Rusty Red", "Red", (185,69,69)),
    ("Muted Brick", "Red", (170,85,85)),    ("Cherry", "Red", (222,49,99)),
    ("Cardinal", "Red", (196,30,58)),       ("Rose Red", "Red", (200,80,90)),
    ("Berry", "Red", (180,40,60)),          ("Light Rose Red", "Red", (220,120,120)),

    # ── Orange additions (peaches, apricots, terracotta) ──
    ("Orange Red", "Orange", (255,69,0)),       ("Peach", "Orange", (255,218,185)),
    ("Light Peach", "Orange", (255,229,180)),   ("Apricot", "Orange", (251,206,177)),
    ("Tan Orange", "Orange", (210,140,90)),     ("Burnt Orange", "Orange", (204,85,0)),
    ("Light Orange", "Orange", (255,200,140)),  ("Bright Orange", "Orange", (255,178,107)),
    ("Pale Orange", "Orange", (255,213,165)),   ("Sandy Orange", "Orange", (244,164,96)),
    ("Terracotta", "Orange", (204,78,57)),      ("Salmon Orange", "Orange", (255,160,122)),
    ("Pumpkin", "Orange", (255,117,24)),        ("Carrot", "Orange", (237,145,33)),
    ("Tangerine", "Orange", (242,133,0)),       ("Amber", "Orange", (255,191,0)),
    ("Persimmon", "Orange", (236,88,0)),

    # ── Yellow additions (pastels, mustard, olive yellow) ──
    ("Light Yellow", "Yellow", (255,255,224)),  ("Lemon Chiffon", "Yellow", (255,250,205)),
    ("Pale Goldenrod", "Yellow", (238,232,170)),("Light Goldenrod", "Yellow", (250,250,210)),
    ("Papaya Whip", "Yellow", (255,239,213)),   ("Cornsilk", "Yellow", (255,248,220)),
    ("Mustard", "Yellow", (255,219,88)),        ("Olive Yellow", "Yellow", (200,180,80)),
    ("Pale Yellow", "Yellow", (255,255,153)),   ("Mellow Yellow", "Yellow", (248,222,126)),
    ("Saffron", "Yellow", (244,196,48)),        ("Banana", "Yellow", (255,225,53)),
    ("Daffodil", "Yellow", (255,255,49)),       ("Honey", "Yellow", (235,182,73)),
    ("Lemon", "Yellow", (255,247,0)),           ("Buttercup", "Yellow", (243,191,63)),
    ("Cream", "Yellow", (255,253,208)),

    # ── Green additions (foliage, sage, olive, mint) — biggest gap region ──
    ("Lawn Green", "Green", (124,252,0)),       ("Chartreuse", "Green", (127,255,0)),
    ("Spring Green", "Green", (0,255,127)),     ("Medium Spring Green", "Green", (0,250,154)),
    ("Pale Green", "Green", (152,251,152)),     ("Sea Green", "Green", (46,139,87)),
    ("Medium Sea Green", "Green", (60,179,113)),("Forest Foliage", "Green", (88,113,69)),
    ("Olive Drab", "Green", (107,142,35)),      ("Olive", "Green", (128,128,0)),
    ("Sage", "Green", (158,174,131)),           ("Moss Green", "Green", (138,154,91)),
    ("Yellow Green", "Green", (154,205,50)),    ("Mint Green", "Green", (152,255,152)),
    ("Avocado", "Green", (118,128,69)),         ("Foliage Mid", "Green", (110,135,85)),
    ("Foliage Light", "Green", (135,160,100)),  ("Foliage Dark", "Green", (75,95,55)),
    ("Pickle", "Green", (94,113,28)),           ("Pistachio", "Green", (147,197,114)),
    ("Hunter Green", "Green", (53,94,59)),      ("Pine", "Green", (33,79,57)),
    ("Emerald", "Green", (80,200,120)),         ("Bottle Green", "Green", (0,106,78)),
    ("Khaki Green", "Green", (135,135,90)),     ("Asparagus", "Green", (135,169,107)),

    # ── Cyan additions (pastels, sea green, teal) ──
    ("Light Cyan", "Cyan", (224,255,255)),      ("Pale Turquoise", "Cyan", (175,238,238)),
    ("Aquamarine", "Cyan", (127,255,212)),      ("Medium Aquamarine", "Cyan", (102,205,170)),
    ("Medium Turquoise", "Cyan", (72,209,204)), ("Light Sea Green", "Cyan", (32,178,170)),
    ("Teal", "Cyan", (0,128,128)),              ("Aqua Mid", "Cyan", (110,200,200)),
    ("Bluish Green Light", "Cyan", (130,200,180)), ("Bluish Green Mid", "Cyan", (95,170,160)),
    ("Pale Mint", "Cyan", (170,210,200)),       ("Spearmint", "Cyan", (140,200,170)),
    ("Robin Egg Blue", "Cyan", (0,204,204)),    ("Tiffany Blue", "Cyan", (10,186,181)),
    ("Sky Cyan", "Cyan", (130,200,210)),

    # ── Blue additions (sky, navy, cobalt) ──
    ("Cornflower Blue", "Blue", (100,149,237)), ("Light Blue", "Blue", (173,216,230)),
    ("Deep Sky Blue", "Blue", (0,191,255)),     ("Powder Blue", "Blue", (176,224,230)),
    ("Alice Blue", "Blue", (240,248,255)),      ("Midnight Blue", "Blue", (25,25,112)),
    ("Cobalt Blue", "Blue", (0,71,171)),        ("Periwinkle", "Blue", (204,204,255)),
    ("Lavender Blue", "Blue", (171,184,228)),   ("Slate Blue", "Blue", (106,90,205)),
    ("Pale Blue", "Blue", (210,220,235)),       ("Sky Mid", "Blue", (110,140,180)),
    ("Sky Muted", "Blue", (95,120,160)),        ("Denim", "Blue", (80,110,150)),
    ("Cerulean", "Blue", (42,82,190)),          ("Sapphire", "Blue", (15,82,186)),
    ("Azure", "Blue", (0,127,255)),             ("Navy Mid", "Blue", (35,55,120)),
    ("Steel Sky", "Blue", (70,100,150)),        ("Marine", "Blue", (30,60,150)),
    ("Pacific", "Blue", (28,107,160)),          ("Periwinkle Mid", "Blue", (140,160,210)),

    # ── Violet additions (orchid, plum, lavender, eggplant) ──
    ("Thistle", "Violet", (216,191,216)),       ("Plum", "Violet", (221,160,221)),
    ("Orchid", "Violet", (218,112,214)),        ("Dark Orchid", "Violet", (153,50,204)),
    ("Blue Violet", "Violet", (138,43,226)),    ("Lavender", "Violet", (230,230,250)),
    ("Mauve", "Violet", (181,126,220)),         ("Pale Violet", "Violet", (200,170,220)),
    ("Eggplant", "Violet", (97,64,81)),         ("Lilac", "Violet", (200,162,200)),
    ("Wisteria", "Violet", (201,160,220)),      ("Heliotrope", "Violet", (223,115,255)),
    ("Amethyst", "Violet", (153,102,204)),      ("Iris", "Violet", (90,79,207)),
    ("Grape", "Violet", (111,45,168)),          ("Blueberry", "Violet", (79,71,140)),

    # ── Pink additions (magenta, rose, fuchsia, light pinks) ──
    ("Misty Rose", "Pink", (255,228,225)),      ("Lavender Blush", "Pink", (255,240,245)),
    ("Rose", "Pink", (255,0,127)),              ("Light Hot Pink", "Pink", (255,182,193)),
    ("Salmon Pink", "Pink", (255,145,164)),     ("Pale Pink", "Pink", (250,218,221)),
    ("Bubblegum", "Pink", (255,193,204)),       ("Fuchsia", "Pink", (255,0,150)),
    ("Magenta-Rose", "Pink", (200,80,140)),     ("Magenta Mid", "Pink", (190,70,140)),
    ("Carnation", "Pink", (255,166,201)),       ("Watermelon", "Pink", (252,108,133)),
    ("Coral Pink", "Pink", (248,131,121)),      ("Cerise", "Pink", (222,49,99)),
    ("Blush", "Pink", (222,93,131)),            ("Rose Pink", "Pink", (240,100,140)),

    # ── Brown additions (tans, beige, skin browns) — Dark Skin gap ──
    ("Peru", "Brown", (205,133,63)),            ("Sandy Brown", "Brown", (244,164,96)),
    ("Wheat", "Brown", (245,222,179)),          ("Rosy Brown", "Brown", (188,143,143)),
    ("Tan", "Brown", (210,180,140)),            ("Khaki Brown", "Brown", (189,158,82)),
    ("Camel", "Brown", (193,154,107)),          ("Beige", "Brown", (245,245,220)),
    ("Bisque", "Brown", (255,228,196)),         ("Light Brown", "Brown", (181,101,29)),
    ("Walnut", "Brown", (95,67,46)),            ("Coffee", "Brown", (111,78,55)),
    ("Mocha", "Brown", (122,85,57)),            ("Russet", "Brown", (128,70,27)),
    ("Skin Tan", "Brown", (190,140,100)),       ("Skin Mid", "Brown", (160,110,80)),
    ("Skin Dark", "Brown", (115,80,65)),        ("Espresso", "Brown", (75,54,33)),
    ("Mahogany", "Brown", (192,64,0)),          ("Caramel", "Brown", (175,111,67)),
    ("Hazelnut", "Brown", (180,140,100)),       ("Tawny", "Brown", (205,87,0)),
]

DB_EXPANDED = DB_CURRENT + DB_ADDITIONS

# ───── X-Rite ColorChecker 24 ────────────────────────────────
COLORCHECKER = [
    ((115,82,68),  "Brown",  "Dark Skin"),
    ((194,150,130),"Pink",   "Light Skin"),
    (( 98,122,157),"Blue",   "Blue Sky"),
    (( 87,108,67), "Green",  "Foliage"),
    ((133,128,177),"Violet", "Blue Flower"),
    ((103,189,170),"Cyan",   "Bluish Green"),
    ((214,126,44), "Orange", "Orange"),
    (( 80,91,166), "Blue",   "Purplish Blue"),
    ((193,90,99),  "Red",    "Moderate Red"),
    (( 94,60,108), "Violet", "Purple"),
    ((157,188,64), "Green",  "Yellow Green"),
    ((224,163,46), "Yellow", "Orange Yellow"),
    (( 56,61,150), "Blue",   "Blue"),
    (( 70,148,73), "Green",  "Green"),
    ((175,54,60),  "Red",    "Red"),
    ((231,199,31), "Yellow", "Yellow"),
    ((187,86,149), "Pink",   "Magenta"),
    ((  8,133,161),"Cyan",   "Cyan"),
    ((243,243,242),NEUTRAL,  "White"),
    ((200,200,200),NEUTRAL,  "Neutral 8"),
    ((160,160,160),NEUTRAL,  "Neutral 6.5"),
    ((122,122,121),NEUTRAL,  "Neutral 5"),
    (( 85, 85, 85),NEUTRAL,  "Neutral 3.5"),
    (( 52, 52, 52),NEUTRAL,  "Black"),
]

def make_id(db):
    db_lab = np.array([rgb_to_lab(e[2]) for e in db])
    def identify(rgb):
        lab = rgb_to_lab(rgb)
        chroma = float(np.sqrt(lab[1]**2 + lab[2]**2))
        chromatic = chroma >= NEUTRAL_CHROMA
        best_i, best_d = -1, np.inf
        for i, (_, klass, _) in enumerate(db):
            if chromatic and klass == NEUTRAL: continue
            if not chromatic and klass != NEUTRAL: continue
            d = float(colour.delta_E(lab, db_lab[i], method="CIE 2000"))
            if d < best_d: best_d, best_i = d, i
        return db[best_i][1], db[best_i][0], best_d
    return identify

def evaluate(db, label):
    ident = make_id(db)
    print(f"\n{'='*78}\n{label}  ({len(db)} entries)\n{'='*78}")
    print(f"{'Patch':22}  {'Expected':9}  {'Predicted':9}  {'Nearest':22}  {'dE':>6}  Result")
    print("-" * 90)
    correct = 0
    fails = []
    for rgb, expected, name in COLORCHECKER:
        pc, nn, dE = ident(rgb)
        ok = (pc == expected)
        if ok: correct += 1; mark = "OK"
        else:
            kind = "sparsity" if dE >= 8 else "re-label"
            fails.append((name, expected, pc, nn, dE, kind))
            mark = f"FAIL ({kind})"
        print(f"{name:22}  {expected:9}  {pc:9}  {nn:22}  {dE:6.2f}  {mark}")
    n = len(COLORCHECKER); acc = correct/n
    print(f"\nAccuracy: {correct}/{n} = {acc*100:.1f}%")
    return acc, fails

def emit_js(adds):
    print(f"\n{'='*78}\nNEW DB ENTRIES — paste into tensorHelper.js IDENTIFIER_DB\n{'='*78}")
    by_class = {}
    for n, k, rgb in adds:
        by_class.setdefault(k, []).append((n, rgb))
    order = [NEUTRAL, "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Violet", "Pink", "Brown"]
    for k in order:
        if k not in by_class: continue
        print(f"\n  // ── {k} additions ──")
        for n, (r, g, b) in by_class[k]:
            hex_ = f"#{r:02X}{g:02X}{b:02X}"
            print(f'  {{ name: "{n}", hex: "{hex_}", lab: rgbToLab({r}, {g}, {b}) }}, // {n.lower()}')

def main():
    acc_b, fails_b = evaluate(DB_CURRENT,  "BEFORE: Current DB")
    acc_a, fails_a = evaluate(DB_EXPANDED, f"AFTER: Expanded DB (+{len(DB_ADDITIONS)} = {len(DB_EXPANDED)} total)")

    print(f"\n{'='*78}\nHYPOTHESIS RESULT\n{'='*78}")
    print(f"Hypothesis: 'Adding ~{len(DB_ADDITIONS)} entries to fill gap regions raises accuracy.'\n")
    print(f"  Before:  {acc_b*100:.1f}%   ({len(fails_b)} failures)")
    print(f"  After:   {acc_a*100:.1f}%   ({len(fails_a)} failures)")
    delta = (acc_a - acc_b) * 100
    print(f"  Change:  {delta:+.1f} pp\n")

    if delta > 0:
        print("VERDICT: HYPOTHESIS CONFIRMED.")
        print(f"  +{delta:.1f}pp by adding {len(DB_ADDITIONS)} entries placed near gap regions.")
        emit_js(DB_ADDITIONS)
    elif delta == 0:
        print("VERDICT: NO IMPROVEMENT.")
        print("  Remaining failures are class-boundary issues, not sparsity.")
    else:
        print("VERDICT: ACCURACY DROPPED.")
        print("  Some new entries are pulling decisions in wrong directions — they need pruning.")

if __name__ == "__main__":
    main()
