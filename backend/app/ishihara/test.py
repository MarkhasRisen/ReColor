"""
Ishihara Color Vision Test Module using DaltonLens methodology
Implements both comprehensive (38 plates) and quick (14 plates) tests
Follows standard clinical scoring and interpretation guidelines

License: BSD-2-Clause (compatible with DaltonLens)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PlateType(Enum):
    """Types of Ishihara plates based on their diagnostic purpose."""
    TRANSFORMATION = "transformation"  # Different answers for normal vs CVD
    VANISHING = "vanishing"  # Only visible to normal vision
    HIDDEN_DIGIT = "hidden_digit"  # Only visible to CVD
    CLASSIFICATION = "classification"  # Distinguishes protan vs deutan
    CONTROL = "control"  # Should be visible to all


class CVDType(Enum):
    """Color vision deficiency types."""
    NORMAL = "normal"
    PROTAN = "protan"  # Red deficiency
    DEUTAN = "deutan"  # Green deficiency
    TRITAN = "tritan"  # Blue deficiency (rare)
    TOTAL = "total"  # Complete color blindness


@dataclass
class IshiharaPlate:
    """Represents a single Ishihara test plate."""
    plate_number: int
    plate_type: PlateType
    normal_answer: Optional[str]  # What normal vision sees
    protan_answer: Optional[str]  # What protanopia sees
    deutan_answer: Optional[str]  # What deuteranopia sees
    description: str
    is_control: bool = False
    
    def evaluate_response(self, response: Optional[str]) -> Dict[str, bool]:
        """
        Evaluate if response matches expected answers for different vision types.
        
        Returns dict with keys: 'normal', 'protan', 'deutan', 'incorrect'
        """
        if response is None or response.strip() == "":
            return {'normal': False, 'protan': False, 'deutan': False, 'incorrect': True}
        
        response = response.strip().lower()
        
        return {
            'normal': response == str(self.normal_answer).lower() if self.normal_answer else False,
            'protan': response == str(self.protan_answer).lower() if self.protan_answer else False,
            'deutan': response == str(self.deutan_answer).lower() if self.deutan_answer else False,
            'incorrect': not any([
                response == str(self.normal_answer).lower() if self.normal_answer else False,
                response == str(self.protan_answer).lower() if self.protan_answer else False,
                response == str(self.deutan_answer).lower() if self.deutan_answer else False
            ])
        }


# Standard 38-plate Ishihara test configuration
COMPREHENSIVE_PLATES = [
    # Control plates (1-2) - Everyone should see these
    IshiharaPlate(1, PlateType.CONTROL, "12", None, None, "Control plate - everyone sees 12", True),
    IshiharaPlate(2, PlateType.CONTROL, "8", None, None, "Control plate - everyone sees 8", True),
    
    # Transformation plates (3-15) - Different answers for normal vs CVD
    IshiharaPlate(3, PlateType.TRANSFORMATION, "6", "5", "5", "Normal: 6, CVD: 5"),
    IshiharaPlate(4, PlateType.TRANSFORMATION, "29", "70", "70", "Normal: 29, CVD: 70"),
    IshiharaPlate(5, PlateType.TRANSFORMATION, "57", "35", "35", "Normal: 57, CVD: 35"),
    IshiharaPlate(6, PlateType.TRANSFORMATION, "5", "2", "2", "Normal: 5, CVD: 2"),
    IshiharaPlate(7, PlateType.TRANSFORMATION, "3", "5", "5", "Normal: 3, CVD: 5"),
    IshiharaPlate(8, PlateType.TRANSFORMATION, "15", "17", "17", "Normal: 15, CVD: 17"),
    IshiharaPlate(9, PlateType.TRANSFORMATION, "74", "21", "21", "Normal: 74, CVD: 21"),
    IshiharaPlate(10, PlateType.TRANSFORMATION, "2", None, None, "Normal: 2, CVD: nothing"),
    IshiharaPlate(11, PlateType.TRANSFORMATION, "6", None, None, "Normal: 6, CVD: nothing"),
    IshiharaPlate(12, PlateType.TRANSFORMATION, "97", None, None, "Normal: 97, CVD: nothing"),
    IshiharaPlate(13, PlateType.TRANSFORMATION, "45", None, None, "Normal: 45, CVD: nothing"),
    IshiharaPlate(14, PlateType.TRANSFORMATION, "5", None, None, "Normal: 5, CVD: nothing"),
    IshiharaPlate(15, PlateType.TRANSFORMATION, "7", None, None, "Normal: 7, CVD: nothing"),
    
    # Hidden digit plates (16-17) - Only CVD can see
    IshiharaPlate(16, PlateType.HIDDEN_DIGIT, None, "45", "45", "Normal: nothing, CVD: 45"),
    IshiharaPlate(17, PlateType.HIDDEN_DIGIT, None, "5", "5", "Normal: nothing, CVD: 5"),
    
    # Classification plates (18-21) - Distinguish protan from deutan
    IshiharaPlate(18, PlateType.CLASSIFICATION, "26", "6", "2", "Protan/Deutan classification"),
    IshiharaPlate(19, PlateType.CLASSIFICATION, "42", "2", "4", "Protan/Deutan classification"),
    IshiharaPlate(20, PlateType.CLASSIFICATION, "35", "5", "3", "Protan/Deutan classification"),
    IshiharaPlate(21, PlateType.CLASSIFICATION, "96", "6", "9", "Protan/Deutan classification"),
    
    # Additional transformation plates (22-25)
    IshiharaPlate(22, PlateType.TRANSFORMATION, "5", None, None, "Normal: 5, CVD: nothing"),
    IshiharaPlate(23, PlateType.TRANSFORMATION, "7", None, None, "Normal: 7, CVD: nothing"),
    IshiharaPlate(24, PlateType.TRANSFORMATION, "16", None, None, "Normal: 16, CVD: nothing"),
    IshiharaPlate(25, PlateType.TRANSFORMATION, "73", None, None, "Normal: 73, CVD: nothing"),
    
    # Tracing plates (26-38) - Follow winding paths (simplified as single answer)
    IshiharaPlate(26, PlateType.VANISHING, "traced", None, None, "Tracing path (purple-red line)"),
    IshiharaPlate(27, PlateType.VANISHING, "traced", None, None, "Tracing path (purple-red line)"),
    IshiharaPlate(28, PlateType.HIDDEN_DIGIT, None, "traced", "traced", "CVD traces blue-green line"),
    IshiharaPlate(29, PlateType.HIDDEN_DIGIT, None, "traced", "traced", "CVD traces blue-green line"),
    IshiharaPlate(30, PlateType.CLASSIFICATION, "traced_purple", "traced_red", "traced_blue", "Classification tracing"),
    IshiharaPlate(31, PlateType.CLASSIFICATION, "traced_purple", "traced_red", "traced_blue", "Classification tracing"),
    IshiharaPlate(32, PlateType.VANISHING, "traced", None, None, "Tracing path (orange line)"),
    IshiharaPlate(33, PlateType.VANISHING, "traced", None, None, "Tracing path (orange line)"),
    IshiharaPlate(34, PlateType.HIDDEN_DIGIT, None, "traced", "traced", "CVD traces line"),
    IshiharaPlate(35, PlateType.HIDDEN_DIGIT, None, "traced", "traced", "CVD traces line"),
    IshiharaPlate(36, PlateType.CLASSIFICATION, "traced_purple", "traced_red", "traced_blue", "Classification tracing"),
    IshiharaPlate(37, PlateType.CLASSIFICATION, "traced_purple", "traced_red", "traced_blue", "Classification tracing"),
    IshiharaPlate(38, PlateType.CONTROL, "traced", "traced", "traced", "Control tracing - all see", True),
]

# Quick 14-plate test (most diagnostic plates)
QUICK_TEST_PLATES = [
    COMPREHENSIVE_PLATES[0],   # Plate 1 - Control
    COMPREHENSIVE_PLATES[2],   # Plate 3 - Transformation
    COMPREHENSIVE_PLATES[3],   # Plate 4 - Transformation
    COMPREHENSIVE_PLATES[4],   # Plate 5 - Transformation
    COMPREHENSIVE_PLATES[5],   # Plate 6 - Transformation
    COMPREHENSIVE_PLATES[6],   # Plate 7 - Transformation
    COMPREHENSIVE_PLATES[7],   # Plate 8 - Transformation
    COMPREHENSIVE_PLATES[8],   # Plate 9 - Transformation
    COMPREHENSIVE_PLATES[9],   # Plate 10 - Vanishing
    COMPREHENSIVE_PLATES[10],  # Plate 11 - Vanishing
    COMPREHENSIVE_PLATES[15],  # Plate 16 - Hidden digit
    COMPREHENSIVE_PLATES[17],  # Plate 18 - Classification
    COMPREHENSIVE_PLATES[18],  # Plate 19 - Classification
    COMPREHENSIVE_PLATES[19],  # Plate 20 - Classification
]


@dataclass
class TestResult:
    """Results from Ishihara test."""
    total_plates: int
    correct_normal: int  # Correct answers for normal vision
    correct_protan: int  # Answers matching protan
    correct_deutan: int  # Answers matching deutan
    incorrect: int  # Completely wrong answers
    control_failed: int  # Failed control plates
    classification_score: Dict[str, int]  # Protan vs Deutan classification
    cvd_type: CVDType
    severity: float  # 0.0 (normal) to 1.0 (severe)
    confidence: float  # Confidence in diagnosis
    interpretation: str


class IshiharaTest:
    """
    Ishihara Color Vision Test implementation.
    Follows standard clinical scoring and interpretation.
    """
    
    def __init__(self, use_comprehensive: bool = False):
        """
        Initialize Ishihara test.
        
        Args:
            use_comprehensive: If True, use 38-plate test. If False, use 14-plate quick test.
        """
        self.plates = COMPREHENSIVE_PLATES if use_comprehensive else QUICK_TEST_PLATES
        self.use_comprehensive = use_comprehensive
    
    def evaluate_test(self, responses: Dict[int, Optional[str]]) -> TestResult:
        """
        Evaluate Ishihara test responses according to clinical standards.
        
        Args:
            responses: Dict mapping plate_number to response (e.g., {1: "12", 3: "5", ...})
            
        Returns:
            TestResult with diagnosis and scoring
        """
        correct_normal = 0
        correct_protan = 0
        correct_deutan = 0
        incorrect = 0
        control_failed = 0
        
        classification_protan = 0
        classification_deutan = 0
        
        for plate in self.plates:
            response = responses.get(plate.plate_number)
            evaluation = plate.evaluate_response(response)
            
            # Count control plate failures
            if plate.is_control and not evaluation['normal']:
                control_failed += 1
            
            # Count matches for each vision type
            if evaluation['normal']:
                correct_normal += 1
            if evaluation['protan']:
                correct_protan += 1
            if evaluation['deutan']:
                correct_deutan += 1
            if evaluation['incorrect']:
                incorrect += 1
            
            # Classification plates help distinguish protan from deutan
            if plate.plate_type == PlateType.CLASSIFICATION:
                if evaluation['protan']:
                    classification_protan += 1
                if evaluation['deutan']:
                    classification_deutan += 1
        
        # Determine CVD type based on clinical criteria
        cvd_type, severity, confidence = self._diagnose(
            correct_normal, correct_protan, correct_deutan,
            classification_protan, classification_deutan,
            control_failed, len(self.plates)
        )
        
        interpretation = self._interpret_results(cvd_type, severity, confidence, control_failed)
        
        return TestResult(
            total_plates=len(self.plates),
            correct_normal=correct_normal,
            correct_protan=correct_protan,
            correct_deutan=correct_deutan,
            incorrect=incorrect,
            control_failed=control_failed,
            classification_score={
                'protan': classification_protan,
                'deutan': classification_deutan
            },
            cvd_type=cvd_type,
            severity=severity,
            confidence=confidence,
            interpretation=interpretation
        )
    
    def _diagnose(
        self,
        correct_normal: int,
        correct_protan: int,
        correct_deutan: int,
        classification_protan: int,
        classification_deutan: int,
        control_failed: int,
        total_plates: int
    ) -> Tuple[CVDType, float, float]:
        """
        Diagnose CVD type and severity based on clinical criteria.
        
        Clinical thresholds (standard):
        - Normal: ≥12/14 correct on quick test, ≥34/38 on comprehensive
        - Mild CVD: 8-11/14 or 25-33/38
        - Moderate CVD: 4-7/14 or 15-24/38
        - Strong CVD: <4/14 or <15/38
        
        Returns:
            (CVD type, severity 0-1, confidence 0-1)
        """
        # Invalid test if control plates failed
        if control_failed > 0:
            return CVDType.NORMAL, 0.0, 0.0
        
        # Calculate normal vision score percentage
        normal_score = correct_normal / total_plates
        
        # Normal vision threshold (clinical standard)
        normal_threshold = 0.86 if self.use_comprehensive else 0.86  # ~12/14 or ~33/38
        
        if normal_score >= normal_threshold:
            return CVDType.NORMAL, 0.0, 0.95
        
        # Determine if protan or deutan based on classification plates
        if classification_protan > classification_deutan:
            cvd_type = CVDType.PROTAN
            cvd_score = correct_protan / total_plates
        elif classification_deutan > classification_protan:
            cvd_type = CVDType.DEUTAN
            cvd_score = correct_deutan / total_plates
        else:
            # Unable to classify clearly - use higher score
            if correct_protan >= correct_deutan:
                cvd_type = CVDType.PROTAN
                cvd_score = correct_protan / total_plates
            else:
                cvd_type = CVDType.DEUTAN
                cvd_score = correct_deutan / total_plates
        
        # Calculate severity (inverted - higher score means less severe CVD)
        # Severity scale: 0.0 = normal, 1.0 = complete deficiency
        if normal_score > 0.57:  # 8/14 or 22/38
            severity = 0.3  # Mild
        elif normal_score > 0.29:  # 4/14 or 11/38
            severity = 0.6  # Moderate
        else:
            severity = 0.9  # Strong/severe
        
        # Confidence based on classification consistency
        total_classification = classification_protan + classification_deutan
        if total_classification > 0:
            classification_ratio = max(classification_protan, classification_deutan) / total_classification
            confidence = min(0.95, 0.6 + (classification_ratio * 0.35))
        else:
            confidence = 0.5  # Low confidence without classification data
        
        return cvd_type, severity, confidence
    
    def _interpret_results(
        self,
        cvd_type: CVDType,
        severity: float,
        confidence: float,
        control_failed: int
    ) -> str:
        """Generate clinical interpretation of results."""
        if control_failed > 0:
            return "Test invalid: Control plate(s) failed. Please retake under proper lighting conditions."
        
        if cvd_type == CVDType.NORMAL:
            return "Normal color vision. No color vision deficiency detected."
        
        # Severity classification
        if severity < 0.4:
            severity_text = "mild"
        elif severity < 0.7:
            severity_text = "moderate"
        else:
            severity_text = "strong"
        
        # CVD type description
        if cvd_type == CVDType.PROTAN:
            cvd_description = "Protanomaly/Protanopia (red deficiency)"
        elif cvd_type == CVDType.DEUTAN:
            cvd_description = "Deuteranomaly/Deuteranopia (green deficiency)"
        else:
            cvd_description = "Color vision deficiency"
        
        confidence_text = f"{int(confidence * 100)}%"
        
        return (
            f"{severity_text.capitalize()} {cvd_description} detected "
            f"(confidence: {confidence_text}). "
            f"Clinical confirmation recommended."
        )
    
    def get_plate_image_path(self, plate_number: int) -> str:
        """
        Get the file path for a plate image.
        
        Args:
            plate_number: Plate number (1-38 for comprehensive, subset for quick)
            
        Returns:
            Path to plate image file
        """
        return f"static/ishihara/plate_{plate_number:02d}.png"
