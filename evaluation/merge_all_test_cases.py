"""
Merge all 4 ReColor test case documents into one comprehensive Word file
with coverage assessment.
"""

import re
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

# ── Paths ──────────────────────────────────────────────────────────────
BASE = r"C:\Users\markr\.gemini\antigravity\brain\09b7a398-5f89-40b9-a86d-6c7983fe12c2"
FILES = [
    (f"{BASE}\\recolor_test_cases_consolidated.md", "Part A: Consolidated Module Test Cases",
     "233 test cases organized by application module (Authentication, Navigation, Camera, Color Correction, CVD Simulation, Color Identifier, Ishihara Screening, Haptic Feedback, Audio Feedback, Career Awareness, Settings, Firebase, Gallery). Uses RC-prefixed IDs."),
    (f"{BASE}\\test_cases_functionality.md", "Part B: Functionality Test Cases",
     "107 test cases organized by screen/feature (Splash, Onboarding, Login, Home, Ishihara Intro/Execution/Results, Camera Enhancement, Color Identifier, CVD Simulation, CVD Gallery, Education/Articles, Survey, Settings, History, Admin/Research). Uses FN-prefixed IDs."),
    (f"{BASE}\\test_cases_android_core.md", "Part C: Android Core App Test Cases",
     "80 test cases across four quality dimensions: Visual Experience (VE), Android Functionality (AF), Performance & Stability (PS), and Privacy & Security (SC)."),
    (f"{BASE}\\test_cases_compatibility.md", "Part D: Compatibility Test Cases",
     "37 test cases covering Android version compatibility, screen size/resolution, orientation, camera hardware, network conditions, and device-specific edge cases. Uses CM-prefixed IDs."),
]
OUT = r"c:\Users\markr\OneDrive\Desktop\Local Project Filee\ReColor\evaluation\recolor_all_test_cases_merged.docx"


def set_cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def style_header_row(row, bg="1F3864"):
    for cell in row.cells:
        set_cell_shading(cell, bg)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.size = Pt(8)
                r.font.name = "Calibri"


def style_data_cell(cell):
    for p in cell.paragraphs:
        pf = p.paragraph_format
        pf.space_before = Pt(1)
        pf.space_after = Pt(1)
        for r in p.runs:
            r.font.size = Pt(8)
            r.font.name = "Calibri"


def add_zebra(table):
    for i, row in enumerate(table.rows):
        if i == 0:
            continue
        if i % 2 == 0:
            for cell in row.cells:
                set_cell_shading(cell, "D6E4F0")


def parse_md(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sections = []
    current = None
    for line in lines:
        raw = line.rstrip("\n\r")
        m = re.match(r"^(#{1,4})\s+(.+)$", raw)
        if m:
            if current and current.get("_in_table"):
                current["tables"].append(current["_current_table"])
                current["_in_table"] = False
            current = {"heading": m.group(2).strip(), "level": len(m.group(1)),
                        "tables": [], "_in_table": False, "_current_table": []}
            sections.append(current)
            continue
        if current is None:
            continue
        if raw.strip() == "---" or raw.startswith(">"):
            continue
        if "|" in raw and raw.strip().startswith("|"):
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            if not current["_in_table"]:
                current["_in_table"] = True
                current["_current_table"] = []
            current["_current_table"].append(cells)
        else:
            if current.get("_in_table"):
                current["tables"].append(current["_current_table"])
                current["_current_table"] = []
                current["_in_table"] = False
    if current and current.get("_in_table"):
        current["tables"].append(current["_current_table"])
    return sections


def add_sections_to_doc(doc, sections):
    for sec in sections:
        if not sec["heading"]:
            continue
        level = min(sec["level"] + 1, 4)  # shift down since Part heading is level 1
        h = doc.add_heading(sec["heading"], level=level)
        for r in h.runs:
            r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64) if level <= 2 else RGBColor(0x2E, 0x74, 0xB5)

        for tbl_data in sec["tables"]:
            if not tbl_data:
                continue
            num_cols = len(tbl_data[0])
            table = doc.add_table(rows=0, cols=num_cols)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.style = "Table Grid"
            for row_cells in tbl_data:
                row = table.add_row()
                for ci, val in enumerate(row_cells):
                    if ci < num_cols:
                        row.cells[ci].text = val
                        style_data_cell(row.cells[ci])
            if table.rows:
                style_header_row(table.rows[0])
            add_zebra(table)
            doc.add_paragraph()


def add_coverage_assessment(doc):
    doc.add_page_break()
    h = doc.add_heading("Part E: Coverage Assessment", level=1)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    # Intro
    p = doc.add_paragraph()
    run = p.add_run("This section assesses whether the combined 457 test cases fully cover the ReColor system by mapping test cases against the application's screens, components, utilities, and key functional areas.")
    run.font.size = Pt(10)
    run.font.name = "Calibri"

    # ── Screen Coverage ──
    h2 = doc.add_heading("Screen-Level Coverage", level=2)
    for r in h2.runs:
        r.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    screens = [
        ("SplashScreen.js", "Yes", "FN-SP-001 to 003, VE-001, PS-001/002"),
        ("AppOnboarding.js", "Yes", "FN-OB-001 to 006, VE-002"),
        ("LoginScreen.js", "Yes", "RC-AU-008 to 013, FN-LG-001 to 006, SC-001 to 003"),
        ("SignUp.js", "Yes", "RC-AU-001 to 007"),
        ("HomeScreen.js", "Yes", "RC-NAV-001 to 002, FN-HM-001 to 005"),
        ("IshiharaIntroScreen.js", "Yes", "FN-IT-001 to 003"),
        ("IshiharaOnboarding.js", "Yes", "FN-IT-004 to 006"),
        ("TestScreen.js", "Yes", "RC-ISH-001 to 047, FN-TE-001 to 017, VE-004 to 006, PS-010/014"),
        ("ResultsScreen.js", "Yes", "FN-RS-001 to 013, VE-013"),
        ("CameraEnhanceScreen.js", "Yes", "RC-CAM-001 to 039, RC-CC-001 to 027, FN-CE-001 to 014, PS-003 to 005/016"),
        ("ColorIdentifierScreen.js", "Yes", "RC-CI-001 to 020, FN-CI-001 to 006, VE-014/015, PS-008"),
        ("CVDSimulationScreen.js", "Yes", "RC-SIM-001 to 007, FN-SIM-001 to 005, PS-006/007"),
        ("CVDGalleryScreen.js", "Yes", "RC-GAL-001 to 012, FN-GL-001 to 004, PS-009"),
        ("EducationListScreen.js", "Yes", "FN-ED-001 to 002"),
        ("ArticleScreen.js", "Yes", "FN-ED-003 to 006, VE-017"),
        ("CareerDetail.js", "Partial", "RC-CA-001 to 008 cover Career broadly; CareerDetail screen not explicitly tested"),
        ("SettingsScreen.js", "Yes", "RC-SET-001 to 009, FN-ST-001 to 008, VE-016"),
        ("HistoryScreen.js", "Yes", "FN-HI-001 to 003"),
        ("ProfileScreen.js", "Partial", "AF-008 mentions tab preservation; no dedicated ProfileScreen tests"),
        ("SurveyScreen.js", "Yes", "FN-SV-001 to 002, SC-009"),
        ("SurveySuccessScreen.js", "Yes", "FN-SV-003"),
        ("AdminLoginScreen.js", "Yes", "FN-AD-001 to 002"),
        ("AdminHubScreen.js", "Yes", "FN-AD-003 to 004"),
        ("ResearchDashboardScreen.js", "Yes", "FN-AD-005 to 006, VE-019, SC-017"),
    ]

    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0]
    for i, txt in enumerate(["Screen File", "Covered?", "Test Case References", "Notes"]):
        hdr.cells[i].text = txt
    style_header_row(hdr)

    covered_count = 0
    partial_count = 0
    for name, status, refs in screens:
        row = table.add_row()
        row.cells[0].text = name
        row.cells[1].text = status
        row.cells[2].text = refs
        note = ""
        if status == "Partial":
            note = "Needs additional dedicated test cases"
            partial_count += 1
        else:
            covered_count += 1
        row.cells[3].text = note
        for c in row.cells:
            style_data_cell(c)
        if status == "Partial":
            set_cell_shading(row.cells[1], "FFF3CD")

    add_zebra(table)
    doc.add_paragraph()

    p = doc.add_paragraph()
    run = p.add_run(f"Result: {covered_count}/24 screens fully covered, {partial_count}/24 partially covered, 0/24 missing.")
    run.bold = True
    run.font.size = Pt(10)
    run.font.name = "Calibri"

    # ── Component & Utility Coverage ──
    h2 = doc.add_heading("Component & Utility Coverage", level=2)
    for r in h2.runs:
        r.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    components = [
        ("ModeSelector.js", "Yes", "FN-CE-014, CM-CH-007, RC-CAM-034 to 039"),
        ("BackgroundBubbles.js", "Yes", "VE-007"),
        ("Card.js", "Implicit", "Used across screens; tested indirectly via VE-003, VE-012"),
        ("DisclaimerBanner.js", "Yes", "VE-020, FN-TE-017, SC-020"),
        ("Header.js", "Implicit", "Tested indirectly via screen-level tests"),
        ("ProgressBar.js", "Yes", "FN-TE-014, VE-019"),
        ("colorLogic.js", "Yes", "RC-ISH-032 to 042 (scoring), FN-RS-001 to 008 (diagnosis)"),
        ("logger.js (AppLog)", "Yes", "FN-ST-005/006, PS-020, VE-016"),
        ("constants.js", "Implicit", "Configuration consumed by other modules"),
        ("firebaseConfig.js", "Yes", "RC-FB-001 to 012, FN-RS-010/011, SC-006 to 009/019, AF-020"),
        ("tensorHelper.js", "Yes", "RC-CC-005 to 027, FN-CE-004/005, RC-SIM-001 to 005"),
        ("MainTabNavigator.js", "Yes", "FN-HM-005, AF-008"),
    ]

    table2 = doc.add_table(rows=1, cols=3)
    table2.style = "Table Grid"
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr2 = table2.rows[0]
    for i, txt in enumerate(["File", "Covered?", "Test Case References"]):
        hdr2.cells[i].text = txt
    style_header_row(hdr2)

    for name, status, refs in components:
        row = table2.add_row()
        row.cells[0].text = name
        row.cells[1].text = status
        row.cells[2].text = refs
        for c in row.cells:
            style_data_cell(c)
    add_zebra(table2)
    doc.add_paragraph()

    # ── Quality Dimension Coverage ──
    h2 = doc.add_heading("Quality Dimension Coverage", level=2)
    for r in h2.runs:
        r.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    dims = [
        ("Functional Correctness", "Yes", "Parts A + B", "457 TCs"),
        ("Visual / UI Quality", "Yes", "Part C Section A", "20 TCs (VE-001 to 020)"),
        ("Android Platform Behavior", "Yes", "Part C Section B", "20 TCs (AF-001 to 020)"),
        ("Performance & Stability", "Yes", "Part C Section C", "20 TCs (PS-001 to 020)"),
        ("Privacy & Security", "Yes", "Part C Section D + Part A Firebase", "20 + 12 TCs"),
        ("Device Compatibility", "Yes", "Part D", "37 TCs (CM-*)"),
        ("Accessibility (a11y)", "No", "---", "No test cases for screen readers, contrast ratios, touch target sizes"),
        ("Localization / i18n", "N/A", "---", "App is English-only; not applicable"),
        ("Usability / UX Heuristics", "Partial", "---", "Covered implicitly by functionality tests but no formal usability heuristics"),
    ]

    table3 = doc.add_table(rows=1, cols=4)
    table3.style = "Table Grid"
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr3 = table3.rows[0]
    for i, txt in enumerate(["Quality Dimension", "Covered?", "Source", "Count / Notes"]):
        hdr3.cells[i].text = txt
    style_header_row(hdr3)

    for dim, status, src, notes in dims:
        row = table3.add_row()
        row.cells[0].text = dim
        row.cells[1].text = status
        row.cells[2].text = src
        row.cells[3].text = notes
        for c in row.cells:
            style_data_cell(c)
        if status == "No":
            set_cell_shading(row.cells[1], "F8D7DA")
        elif status == "Partial":
            set_cell_shading(row.cells[1], "FFF3CD")
    add_zebra(table3)
    doc.add_paragraph()

    # ── Gaps ──
    h2 = doc.add_heading("Identified Gaps", level=2)
    for r in h2.runs:
        r.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    gaps = [
        ("ProfileScreen.js", "No dedicated test cases for the Profile tab screen (user info display, avatar, Manage Settings button)."),
        ("CareerDetail.js", "Career module tests (RC-CA) cover the list screen but not the detail/drill-down screen (roles, stats, color boxes, facts)."),
        ("Accessibility", "No test cases for TalkBack/screen reader compatibility, minimum touch target sizes (48dp), or WCAG contrast compliance."),
        ("Ishihara Plate Image Integrity", "Tests verify plates render but do not validate that the correct Ishihara image file is mapped to each plate number."),
        ("Data Deletion / Account Management", "No test cases for user requesting data deletion or account removal (GDPR/privacy consideration)."),
        ("SignUp Confirm Password", "SignUp.js has a confirm-password field, but RC-AU test cases do not test password mismatch scenario on the SignUp screen specifically."),
    ]

    for title, desc in gaps:
        p = doc.add_paragraph()
        run_b = p.add_run(f"{title}: ")
        run_b.bold = True
        run_b.font.size = Pt(10)
        run_b.font.name = "Calibri"
        run_d = p.add_run(desc)
        run_d.font.size = Pt(10)
        run_d.font.name = "Calibri"

    # ── Verdict ──
    h2 = doc.add_heading("Overall Verdict", level=2)
    for r in h2.runs:
        r.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    verdict_lines = [
        "The combined 457 test cases provide STRONG coverage of the ReColor system.",
        "",
        "Coverage Strengths:",
        "  - All 24 screens are covered (22 fully, 2 partially)",
        "  - All major functional flows are tested end-to-end",
        "  - Camera pipeline thoroughly tested (39 camera + 27 correction + 7 simulation TCs)",
        "  - Ishihara scoring logic comprehensively validated (47 module + 17 functionality TCs)",
        "  - Firebase dual-write privacy model explicitly tested",
        "  - Android lifecycle, permissions, and offline behavior well covered",
        "  - Performance benchmarks with specific latency thresholds defined",
        "  - Compatibility tested across Android 12-15, multiple screen sizes, and network conditions",
        "",
        "Coverage Gaps (Minor):",
        "  - ProfileScreen and CareerDetail need dedicated test cases",
        "  - Accessibility testing (TalkBack, touch targets) is absent",
        "  - SignUp confirm-password mismatch not explicitly tested",
        "  - No data deletion / account removal tests",
        "",
        "Recommendation: The test suite is thesis-ready. The identified gaps are minor and do not represent critical functional risks. Adding 5-10 additional test cases for the gaps above would bring coverage to comprehensive.",
    ]

    for line in verdict_lines:
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.size = Pt(10)
        run.font.name = "Calibri"
        if line.startswith("The combined") or line.startswith("Recommendation:"):
            run.bold = True

    # ── Summary Table ──
    h2 = doc.add_heading("Test Case Summary", level=2)
    for r in h2.runs:
        r.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    summary = [
        ("Part A: Consolidated Module Tests", "RC-*", "233"),
        ("Part B: Functionality Tests", "FN-*", "107"),
        ("Part C: Android Core Tests", "VE/AF/PS/SC-*", "80"),
        ("Part D: Compatibility Tests", "CM-*", "37"),
        ("TOTAL", "", "457"),
    ]

    table4 = doc.add_table(rows=1, cols=3)
    table4.style = "Table Grid"
    table4.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr4 = table4.rows[0]
    for i, txt in enumerate(["Document", "ID Prefix", "Count"]):
        hdr4.cells[i].text = txt
    style_header_row(hdr4)

    for name, prefix, count in summary:
        row = table4.add_row()
        row.cells[0].text = name
        row.cells[1].text = prefix
        row.cells[2].text = count
        for c in row.cells:
            style_data_cell(c)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        if name == "TOTAL":
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs:
                        r.bold = True
                set_cell_shading(c, "1F3864")
                for p in c.paragraphs:
                    for r in p.runs:
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    add_zebra(table4)


def main():
    doc = Document()

    # Page setup: landscape, narrow margins
    for section in doc.sections:
        section.top_margin = Cm(1.27)
        section.bottom_margin = Cm(1.27)
        section.left_margin = Cm(1.27)
        section.right_margin = Cm(1.27)
        w, h = section.page_width, section.page_height
        section.page_width = max(w, h)
        section.page_height = min(w, h)

    # Title
    title = doc.add_heading("ReColor - Comprehensive Test Cases", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("All test case documents merged with coverage assessment\nMark [check] under Pass or Fail. Add remarks for failures.")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    run.italic = True

    # Table of Contents summary
    doc.add_paragraph()
    toc_h = doc.add_heading("Document Contents", level=1)
    for r in toc_h.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    for _, part_title, desc in FILES:
        p = doc.add_paragraph()
        run_b = p.add_run(f"{part_title}: ")
        run_b.bold = True
        run_b.font.size = Pt(10)
        run_b.font.name = "Calibri"
        run_d = p.add_run(desc)
        run_d.font.size = Pt(10)
        run_d.font.name = "Calibri"

    p = doc.add_paragraph()
    run_b = p.add_run("Part E: Coverage Assessment: ")
    run_b.bold = True
    run_b.font.size = Pt(10)
    run_b.font.name = "Calibri"
    run_d = p.add_run("Screen-level, component-level, and quality-dimension coverage analysis with identified gaps and overall verdict.")
    run_d.font.size = Pt(10)
    run_d.font.name = "Calibri"

    # Add each part
    for path, part_title, desc in FILES:
        doc.add_page_break()
        h = doc.add_heading(part_title, level=1)
        for r in h.runs:
            r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

        p = doc.add_paragraph()
        run = p.add_run(desc)
        run.font.size = Pt(10)
        run.font.name = "Calibri"
        run.italic = True
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        doc.add_paragraph()

        sections = parse_md(path)
        add_sections_to_doc(doc, sections)

    # Coverage Assessment
    add_coverage_assessment(doc)

    doc.save(OUT)
    print(f"[OK] Saved: {OUT}")


if __name__ == "__main__":
    main()
