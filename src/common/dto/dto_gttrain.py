# common/dto_gttrain.py
from __future__ import annotations
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class GTTrainRequest(BaseModel):
    job_id: str = Field(..., description="GT train job id")
    gt_version: str = Field("current", description="GT version or 'current'")
    epochs: int = 30
    imgsz: int = 640
    batch: int = 8

    # 모델별로 필요하면 사용
    extra: Dict[str, Any] = Field(default_factory=dict)


class GTTrainResponse(BaseModel):
    job_id: str
    model: str
    status: str  # STARTED | DONE | FAILED
    detail: Optional[str] = None
