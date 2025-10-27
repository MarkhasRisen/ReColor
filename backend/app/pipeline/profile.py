"""Vision profile utilities derived from calibration modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

PlateResult = Literal["correct", "incorrect", "skipped"]


@dataclass
class VisionProfile:
    """Represent a user-specific color vision profile."""

    deficiency: Literal["protan", "deutan", "tritan", "normal"]
    severity: float
    confidence: float

    @classmethod
    def from_ishihara_results(cls, responses: Dict[str, PlateResult]) -> "VisionProfile":
        """Infer profile from Ishihara module responses."""
        categories = {
            "protan": ["p1", "p2", "p3"],
            "deutan": ["d1", "d2", "d3"],
            "tritan": ["t1", "t2"],
        }
        scores = {key: 0 for key in categories}
        total_items = {key: len(value) for key, value in categories.items()}

        for category, plates in categories.items():
            for plate_id in plates:
                result = responses.get(plate_id, "skipped")
                if result == "incorrect":
                    scores[category] += 1
                elif result == "skipped":
                    scores[category] += 0.5

        normalized = {key: scores[key] / total_items[key] for key in scores}
        deficiency = max(normalized, key=normalized.get)
        severity = min(1.0, normalized[deficiency])
        confidence = 1.0 - abs(normalized[deficiency] - severity)

        if severity < 0.2:
            return cls(deficiency="normal", severity=0.0, confidence=confidence)
        return cls(deficiency=deficiency, severity=severity, confidence=confidence)
