"""Schemas for image processing endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    profile_version: Optional[str] = Field(None, description="Hint to select a specific TFLite model")
    image_base64: str = Field(..., description="Base64-encoded RGB image")


class ProcessResponse(BaseModel):
    content_type: str
    data: str
