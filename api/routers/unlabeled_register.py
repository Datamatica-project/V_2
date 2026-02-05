from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.common.utils.unlabeled_register import register_unlabeled

router = APIRouter(prefix="/api/v2/unlabeled", tags=["Unlabeled"])


class RegisterUnlabeledRequest(BaseModel):
    unlabeled_id: str = Field(
        ...,
        description="unlabeled ingest id (e.g. unlabeled_20260205_150000)",
        examples=["unlabeled_20260205_150000"],
    )
    source_dir: str = Field(
        ...,
        description="path to directory containing unlabeled images",
        examples=["/workspace/storage/incoming/unlabeled/images"],
    )
    copy_mode: Literal["symlink", "copy"] = Field(
        "symlink",
        description="how to register unlabeled images (symlink recommended)",
        examples=["symlink"],
    )


class RegisterUnlabeledResponse(BaseModel):
    status: str
    unlabeled_id: str
    unlabeled_version_dir: str
    current_unlabeled: str


@router.post("/register", response_model=RegisterUnlabeledResponse)
def register_unlabeled_api(req: RegisterUnlabeledRequest):
    """
    Register unlabeled image directory and update current UNLABELED pointer.
    """
    try:
        version_dir = register_unlabeled(
            unlabeled_id=req.unlabeled_id,
            source_dir=Path(req.source_dir),
            target_root=Path("/workspace/storage/unlabeled"),
            copy_mode=req.copy_mode,
        )
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RegisterUnlabeledResponse(
        status="OK",
        unlabeled_id=req.unlabeled_id,
        unlabeled_version_dir=str(version_dir),
        current_unlabeled=str(Path("/workspace/storage/unlabeled/UNLABELED")),
    )
