"""Schemas for calibration endpoints."""
from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field, validator

from ..pipeline.profile import PlateResult, VisionProfile


class CalibrationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    responses: Dict[str, Literal["correct", "incorrect", "skipped"]]

    @validator("responses")
    def ensure_non_empty(cls, value: Dict[str, PlateResult]) -> Dict[str, PlateResult]:
        if not value:
            raise ValueError("responses cannot be empty")
        return value

    def to_profile(self) -> VisionProfile:
        return VisionProfile.from_ishihara_results(self.responses)


class CalibrationResponse(BaseModel):
    deficiency: str
    severity: float
    confidence: float
