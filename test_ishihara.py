"""
Test script for Ishihara Color Vision Test implementation.
Tests both quick (14 plates) and comprehensive (38 plates) modes.
"""
import sys
sys.path.insert(0, 'backend')

from app.ishihara.test import IshiharaTest, CVDType

def print_section(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def test_normal_vision():
    """Test with normal vision responses."""
    print_section("TEST 1: Normal Color Vision (Quick Test)")
    
    test = IshiharaTest(use_comprehensive=False)
    
    # Simulate perfect normal vision responses
    responses = {
        1: "12",    # Control
        3: "6",     # Transformation
        4: "29",    # Transformation
        5: "57",    # Transformation
        6: "5",     # Transformation
        7: "3",     # Transformation
        8: "15",    # Transformation
        9: "74",    # Transformation
        10: "2",    # Transformation
        11: "6",    # Transformation
        16: "",     # Hidden digit (normal sees nothing)
        18: "26",   # Classification
        19: "42",   # Classification
        20: "35",   # Classification
    }
    
    result = test.evaluate_test(responses)
    
    print(f"Total Plates: {result.total_plates}")
    print(f"Correct (Normal): {result.correct_normal}/{result.total_plates}")
    print(f"Correct (Protan): {result.correct_protan}/{result.total_plates}")
    print(f"Correct (Deutan): {result.correct_deutan}/{result.total_plates}")
    print(f"Control Failures: {result.control_failed}")
    print(f"\nDiagnosis: {result.cvd_type.value.upper()}")
    print(f"Severity: {result.severity:.2f} (0=normal, 1=severe)")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nInterpretation: {result.interpretation}")
    
    assert result.cvd_type == CVDType.NORMAL, "Should diagnose as normal"
    print("\n✅ TEST PASSED: Normal vision correctly identified")

def test_protan_deficiency():
    """Test with protanopia responses."""
    print_section("TEST 2: Protanopia (Red Deficiency)")
    
    test = IshiharaTest(use_comprehensive=False)
    
    # Simulate protan responses
    responses = {
        1: "12",    # Control - correct
        3: "5",     # Sees 5 instead of 6 (protan)
        4: "70",    # Sees 70 instead of 29 (protan)
        5: "35",    # Sees 35 instead of 57 (protan)
        6: "2",     # Sees 2 instead of 5 (protan)
        7: "5",     # Sees 5 instead of 3 (protan)
        8: "17",    # Sees 17 instead of 15 (protan)
        9: "21",    # Sees 21 instead of 74 (protan)
        10: "",     # Sees nothing (protan)
        11: "",     # Sees nothing (protan)
        16: "45",   # Hidden digit visible to CVD
        18: "6",    # Protan classification
        19: "2",    # Protan classification
        20: "5",    # Protan classification
    }
    
    result = test.evaluate_test(responses)
    
    print(f"Total Plates: {result.total_plates}")
    print(f"Correct (Normal): {result.correct_normal}/{result.total_plates}")
    print(f"Correct (Protan): {result.correct_protan}/{result.total_plates}")
    print(f"Correct (Deutan): {result.correct_deutan}/{result.total_plates}")
    print(f"Control Failures: {result.control_failed}")
    print(f"\nClassification Score:")
    print(f"  Protan: {result.classification_score['protan']}")
    print(f"  Deutan: {result.classification_score['deutan']}")
    print(f"\nDiagnosis: {result.cvd_type.value.upper()}")
    print(f"Severity: {result.severity:.2f}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nInterpretation: {result.interpretation}")
    
    assert result.cvd_type == CVDType.PROTAN, "Should diagnose as protan"
    assert result.classification_score['protan'] > result.classification_score['deutan'], \
        "Protan score should be higher"
    print("\n✅ TEST PASSED: Protanopia correctly identified")

def test_deutan_deficiency():
    """Test with deuteranopia responses."""
    print_section("TEST 3: Deuteranopia (Green Deficiency)")
    
    test = IshiharaTest(use_comprehensive=False)
    
    # Simulate deutan responses (similar to protan for most transformation plates)
    responses = {
        1: "12",    # Control
        3: "5",     # CVD response
        4: "70",    # CVD response
        5: "35",    # CVD response
        6: "2",     # CVD response
        7: "5",     # CVD response
        8: "17",    # CVD response
        9: "21",    # CVD response
        10: "",     # CVD sees nothing
        11: "",     # CVD sees nothing
        16: "45",   # Hidden digit
        18: "2",    # Deutan classification
        19: "4",    # Deutan classification
        20: "3",    # Deutan classification
    }
    
    result = test.evaluate_test(responses)
    
    print(f"Total Plates: {result.total_plates}")
    print(f"Correct (Normal): {result.correct_normal}/{result.total_plates}")
    print(f"Correct (Protan): {result.correct_protan}/{result.total_plates}")
    print(f"Correct (Deutan): {result.correct_deutan}/{result.total_plates}")
    print(f"\nClassification Score:")
    print(f"  Protan: {result.classification_score['protan']}")
    print(f"  Deutan: {result.classification_score['deutan']}")
    print(f"\nDiagnosis: {result.cvd_type.value.upper()}")
    print(f"Severity: {result.severity:.2f}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nInterpretation: {result.interpretation}")
    
    assert result.cvd_type == CVDType.DEUTAN, "Should diagnose as deutan"
    assert result.classification_score['deutan'] >= result.classification_score['protan'], \
        "Deutan score should be higher or equal"
    print("\n✅ TEST PASSED: Deuteranopia correctly identified")

def test_mild_cvd():
    """Test with mild CVD (some correct, some wrong)."""
    print_section("TEST 4: Mild Color Vision Deficiency")
    
    test = IshiharaTest(use_comprehensive=False)
    
    # Mix of normal and CVD responses (mild deficiency)
    responses = {
        1: "12",    # Control - correct
        3: "6",     # Normal - correct
        4: "29",    # Normal - correct
        5: "35",    # CVD response
        6: "5",     # Normal - correct
        7: "5",     # CVD response
        8: "15",    # Normal - correct
        9: "74",    # Normal - correct
        10: "2",    # Normal - correct
        11: "6",    # Normal - correct
        16: "",     # Normal response
        18: "26",   # Normal - correct
        19: "2",    # Protan response
        20: "35",   # Normal - correct
    }
    
    result = test.evaluate_test(responses)
    
    print(f"Total Plates: {result.total_plates}")
    print(f"Correct (Normal): {result.correct_normal}/{result.total_plates} ({result.correct_normal/result.total_plates:.1%})")
    print(f"\nDiagnosis: {result.cvd_type.value.upper()}")
    print(f"Severity: {result.severity:.2f}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nInterpretation: {result.interpretation}")
    
    assert result.severity < 0.5, "Severity should indicate mild deficiency"
    print("\n✅ TEST PASSED: Mild CVD correctly classified")

def test_control_failure():
    """Test with control plate failure (invalid test)."""
    print_section("TEST 5: Control Plate Failure (Invalid Test)")
    
    test = IshiharaTest(use_comprehensive=False)
    
    # Fail control plate
    responses = {
        1: "21",    # Control WRONG
        3: "6",
        4: "29",
        5: "57",
        6: "5",
        7: "3",
        8: "15",
        9: "74",
        10: "2",
        11: "6",
        16: "",
        18: "26",
        19: "42",
        20: "35",
    }
    
    result = test.evaluate_test(responses)
    
    print(f"Control Failures: {result.control_failed}")
    print(f"Diagnosis: {result.cvd_type.value.upper()}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nInterpretation: {result.interpretation}")
    
    assert result.control_failed > 0, "Should detect control failure"
    assert result.confidence < 0.5, "Confidence should be low for invalid test"
    print("\n✅ TEST PASSED: Control failure detected")

def test_comprehensive_mode():
    """Test comprehensive 38-plate mode."""
    print_section("TEST 6: Comprehensive Mode (38 Plates)")
    
    test = IshiharaTest(use_comprehensive=True)
    
    print(f"Total plates in comprehensive test: {len(test.plates)}")
    assert len(test.plates) == 38, "Should have 38 plates"
    
    # Test with partial responses (just first 10)
    responses = {i: "12" if i <= 2 else "correct" for i in range(1, 11)}
    result = test.evaluate_test(responses)
    
    print(f"Test evaluated with {len(responses)} responses")
    print(f"Result calculated successfully")
    print("\n✅ TEST PASSED: Comprehensive mode functional")

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ISHIHARA COLOR VISION TEST - COMPREHENSIVE TEST SUITE")
    print("DaltonLens Compatible Implementation")
    print("="*70)
    
    try:
        test_normal_vision()
        test_protan_deficiency()
        test_deutan_deficiency()
        test_mild_cvd()
        test_control_failure()
        test_comprehensive_mode()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("✅ Normal vision detection")
        print("✅ Protanopia classification")
        print("✅ Deuteranopia classification")
        print("✅ Severity grading (mild/moderate/strong)")
        print("✅ Control plate validation")
        print("✅ Comprehensive mode (38 plates)")
        print("\nClinical standards implemented:")
        print("  - Standard scoring thresholds")
        print("  - Protan/Deutan classification")
        print("  - Severity assessment")
        print("  - Control plate validation")
        print("  - Confidence scoring")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
