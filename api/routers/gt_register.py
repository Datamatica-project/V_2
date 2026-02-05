from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.common.utils.register_gt import register_gt  # ✅ SSOT: src만 import

router = APIRouter(prefix="/api/v2/gt", tags=["GT"])


class RegisterGTRequest(BaseModel):
    ingest_id: str = Field(..., description="GT ingest id (e.g. gt_20260109_101200)", examples=["gt_20260109_101200"])
    extracted_dir: str = Field(..., description="path to extracted GT directory", examples=["/workspace/_tmp/gt_unpack/gt_20260109_101200"])
    copy_mode: Literal["copy", "symlink"] = Field(
        "symlink",
        description="how to place GT files into versioned dir (symlink is fast / copy is fully materialized)",
        examples=["symlink"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ingest_id": "gt_20260109_101200",
                    "extracted_dir": "/workspace/_tmp/gt_unpack/gt_20260109_101200",
                    "copy_mode": "symlink",
                }
            ]
        }
    }


class RegisterGTResponse(BaseModel):
    status: str = Field(..., description="OK if succeeded", examples=["OK"])
    ingest_id: str = Field(..., description="ingest id echoed back", examples=["gt_20260109_101200"])
    gt_version_dir: str = Field(..., description="created GT version directory path", examples=["/workspace/storage/datasets/gt/versions/gt_20260109_101200"])
    current_gt: str = Field(..., description="current GT pointer path (symlink/dir)", examples=["/workspace/storage/datasets/gt/GT"])


@router.post("/register", response_model=RegisterGTResponse)
def register_gt_api(req: RegisterGTRequest):
    """
    Register GT package and update current GT link immediately.
    """
    try:
        gt_ver_dir = register_gt(
            ingest_id=req.ingest_id,
            extracted_dir=Path(req.extracted_dir),
            copy_mode=req.copy_mode,
        )
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RegisterGTResponse(
        status="OK",
        ingest_id=req.ingest_id,
        gt_version_dir=str(gt_ver_dir),
        current_gt=str(Path(gt_ver_dir).parent.parent / "GT"),
    )
