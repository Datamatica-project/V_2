from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Literal, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.common.utils.unlabeled_register import register_unlabeled

router = APIRouter(prefix="/api/v2/unlabeled", tags=["Unlabeled"])

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
UNLABELED_UNPACK_ROOT = Path(os.getenv("UNLABELED_UNPACK_ROOT", "/workspace/_tmp/unlabeled_unpack")).resolve()
UNLABELED_UNPACK_ROOT.mkdir(parents=True, exist_ok=True)

UNLABELED_TARGET_ROOT = Path(os.getenv("UNLABELED_TARGET_ROOT", "/workspace/storage/unlabeled")).resolve()
UNLABELED_TARGET_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------------------------------------------------------
# ZIP helpers (safe unzip)
# -----------------------------------------------------------------------------
def _safe_extract_zip(zip_path: Path, dst_dir: Path) -> None:
    """
    ZipSlip 방지용 안전 압축해제.
    - 절대경로/상위경로(..) 포함 엔트리 차단
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename

            # 디렉토리 엔트리
            if name.endswith("/"):
                continue

            p = Path(name)

            # 절대경로/상위경로 차단
            if p.is_absolute() or ".." in p.parts:
                raise HTTPException(status_code=400, detail=f"unsafe zip entry path: {name}")

            out_path = (dst_dir / p).resolve()
            if not str(out_path).startswith(str(dst_dir.resolve())):
                raise HTTPException(status_code=400, detail=f"unsafe zip entry path: {name}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            n += 1
    return n


def _resolve_images_dir(extracted_dir: Path) -> Path:
    """
    unlabeled zip은 보통
      1) images/ 아래에 이미지가 있거나
      2) 루트에 바로 이미지가 있을 수 있음
    둘 다 허용하되, register_unlabeled에는 '이미지들이 들어있는 디렉토리'를 넘긴다.
    """
    images_dir = extracted_dir / "images"
    if images_dir.exists():
        return images_dir
    return extracted_dir


def _validate_unlabeled_layout(extracted_dir: Path) -> Tuple[Path, int]:
    """
    최소 구조 검증:
      - images/ 또는 루트에 이미지가 1장 이상 있어야 함
    반환: (images_dir, n_images)
    """
    images_dir = _resolve_images_dir(extracted_dir)
    n_images = _count_images(images_dir)
    if n_images <= 0:
        raise HTTPException(status_code=400, detail="no images found in unlabeled zip (expected images/ or image files at root)")
    return images_dir, n_images


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------
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


class UploadUnlabeledResponse(BaseModel):
    status: str = Field(..., examples=["OK"])
    unlabeled_id: str = Field(..., examples=["unlabeled_20260210_090000"])
    extracted_dir: str = Field(..., examples=["/workspace/_tmp/unlabeled_unpack/unlabeled_20260210_090000"])
    images_dir: str = Field(..., examples=["/workspace/_tmp/unlabeled_unpack/unlabeled_20260210_090000/images"])
    n_images: int = Field(..., examples=[842])


class UploadAndRegisterUnlabeledResponse(BaseModel):
    status: str = Field(..., examples=["OK"])
    unlabeled_id: str = Field(..., examples=["unlabeled_20260210_090000"])
    extracted_dir: str = Field(..., examples=["/workspace/_tmp/unlabeled_unpack/unlabeled_20260210_090000"])
    images_dir: str = Field(..., examples=["/workspace/_tmp/unlabeled_unpack/unlabeled_20260210_090000/images"])
    n_images: int = Field(..., examples=[842])
    unlabeled_version_dir: str = Field(..., examples=["/workspace/storage/unlabeled/versions/unlabeled_20260210_090000"])
    current_unlabeled: str = Field(..., examples=["/workspace/storage/unlabeled/UNLABELED"])


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.post("/upload", response_model=UploadUnlabeledResponse)
async def upload_unlabeled_zip(
    file: UploadFile = File(..., description="unlabeled zip (images only; may contain images/ or images at root)"),
    unlabeled_id: Optional[str] = Form(None, description="optional unlabeled id; if omitted, use filename stem"),
):
    """
    Upload unlabeled zip -> unzip to /workspace/_tmp/unlabeled_unpack/{unlabeled_id} -> return extracted_dir.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    safe_stem = Path(file.filename).stem.replace(" ", "_")
    uid = unlabeled_id or f"unlabeled_{safe_stem}"

    dst_dir = (UNLABELED_UNPACK_ROOT / uid).resolve()
    if dst_dir.exists():
        raise HTTPException(status_code=409, detail=f"unlabeled_id already exists under unpack root: {dst_dir}")

    tmp_zip = (UNLABELED_UNPACK_ROOT / f"_{uid}.zip").resolve()
    tmp_zip.parent.mkdir(parents=True, exist_ok=True)

    try:
        # save zip
        with tmp_zip.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        # unzip safely
        _safe_extract_zip(tmp_zip, dst_dir)

        # validate layout
        images_dir, n_images = _validate_unlabeled_layout(dst_dir)

        return UploadUnlabeledResponse(
            status="OK",
            unlabeled_id=uid,
            extracted_dir=str(dst_dir),
            images_dir=str(images_dir),
            n_images=n_images,
        )

    finally:
        tmp_zip.unlink(missing_ok=True)


@router.post("/upload_and_register", response_model=UploadAndRegisterUnlabeledResponse)
async def upload_and_register_unlabeled(
    file: UploadFile = File(..., description="unlabeled zip"),
    unlabeled_id: Optional[str] = Form(None, description="optional unlabeled id"),
    copy_mode: Literal["symlink", "copy"] = Form("symlink", description="symlink recommended"),
):
    """
    Convenience endpoint:
      upload zip -> unzip -> register_unlabeled -> update current UNLABELED pointer.
    """
    up = await upload_unlabeled_zip(file=file, unlabeled_id=unlabeled_id)

    try:
        version_dir = register_unlabeled(
            unlabeled_id=up.unlabeled_id,
            source_dir=Path(up.images_dir),
            target_root=UNLABELED_TARGET_ROOT,
            copy_mode=copy_mode,
        )
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadAndRegisterUnlabeledResponse(
        status="OK",
        unlabeled_id=up.unlabeled_id,
        extracted_dir=up.extracted_dir,
        images_dir=up.images_dir,
        n_images=up.n_images,
        unlabeled_version_dir=str(version_dir),
        current_unlabeled=str(UNLABELED_TARGET_ROOT / "UNLABELED"),
    )


@router.post("/register", response_model=RegisterUnlabeledResponse)
def register_unlabeled_api(req: RegisterUnlabeledRequest):
    """
    Register unlabeled image directory and update current UNLABELED pointer.
    """
    try:
        version_dir = register_unlabeled(
            unlabeled_id=req.unlabeled_id,
            source_dir=Path(req.source_dir),
            target_root=UNLABELED_TARGET_ROOT,
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
        current_unlabeled=str(UNLABELED_TARGET_ROOT / "UNLABELED"),
    )
