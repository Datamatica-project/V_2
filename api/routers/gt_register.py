from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Literal, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.common.utils.register_gt import register_gt

router = APIRouter(prefix="/api/v2/gt", tags=["GT"])

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
GT_UNPACK_ROOT = Path(os.getenv("GT_UNPACK_ROOT", "/workspace/_tmp/gt_unpack")).resolve()
GT_UNPACK_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------------------------------------------------------
# Post-process GT for RTM (annotation only)
# -----------------------------------------------------------------------------
def _normalize_annotation_dir(gt_root: Path) -> Path | None:
    ann = gt_root / "annotation"
    anns = gt_root / "annotations"

    if ann.exists() and ann.is_dir():
        return ann

    if anns.exists() and anns.is_dir():
        anns.rename(ann)
        return ann

    return None


def _ensure_rtm_train_val_json(ann_dir: Path, copy_mode: Literal["copy", "symlink"]) -> None:
    if not ann_dir.exists():
        return

    train_path = ann_dir / "train.json"
    val_path = ann_dir / "val.json"

    if train_path.exists() and val_path.exists():
        return

    candidates = sorted([p for p in ann_dir.glob("*.json") if p.is_file() and p.name not in ("train.json", "val.json")])
    if not candidates:
        return

    src = next((p for p in candidates if p.name.startswith("instances")), candidates[0])

    def _make(dst: Path) -> None:
        if dst.exists():
            return
        if copy_mode == "symlink":
            dst.symlink_to(src.name)
        else:
            shutil.copy2(src, dst)

    _make(train_path)
    _make(val_path)


def _postprocess_gt_for_rtm(gt_root: Path, copy_mode: Literal["copy", "symlink"]) -> None:
    ann_dir = _normalize_annotation_dir(gt_root)
    if ann_dir is None:
        return
    _ensure_rtm_train_val_json(ann_dir, copy_mode=copy_mode)


# -----------------------------------------------------------------------------
# ZIP helpers (safe unzip)
# -----------------------------------------------------------------------------
def _safe_extract_zip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue

            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise HTTPException(status_code=400, detail=f"unsafe zip entry path: {name}")

            out_path = (dst_dir / p).resolve()
            if not str(out_path).startswith(str(dst_dir.resolve())):
                raise HTTPException(status_code=400, detail=f"unsafe zip entry path: {name}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and out_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"zip entry conflicts with directory: {name}",
                )

            with zf.open(info, "r") as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _count_images(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    n = 0
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            n += 1
    return n


def _list_top_dirs(p: Path) -> list[Path]:
    if not p.exists():
        return []
    return [x for x in p.iterdir() if x.is_dir() and x.name != "__MACOSX"]


def _maybe_descend_single_root(extracted_dir: Path) -> Path:
    dirs = _list_top_dirs(extracted_dir)
    if len(dirs) == 1 and (dirs[0] / "images").exists():
        return dirs[0]
    return extracted_dir


def _validate_gt_layout(extracted_dir: Path) -> Tuple[bool, bool, int]:
    images_dir = extracted_dir / "images"
    if not images_dir.exists():
        raise HTTPException(status_code=400, detail="GT zip must contain 'images/' directory")

    n_images = _count_images(images_dir)
    if n_images <= 0:
        raise HTTPException(status_code=400, detail="no images found under 'images/'")

    has_labels = (extracted_dir / "labels").exists()
    has_ann = (extracted_dir / "annotation").exists() or (extracted_dir / "annotations").exists()

    if not (has_labels or has_ann):
        raise HTTPException(status_code=400, detail="GT zip must contain 'labels/' or 'annotation(s)/' directory")

    return has_labels, has_ann, n_images


# -----------------------------------------------------------------------------
# API Models
# -----------------------------------------------------------------------------
class RegisterGTRequest(BaseModel):
    ingest_id: str = Field(..., description="GT ingest id", examples=["gt_20260109_101200"])
    extracted_dir: str = Field(..., description="path to extracted GT directory", examples=["/workspace/_tmp/gt_unpack/gt_20260109_101200"])
    copy_mode: Literal["copy", "symlink"] = Field("symlink", description="copy or symlink", examples=["symlink"])
    model_config = {"extra": "ignore"}


class RegisterGTResponse(BaseModel):
    status: str = Field(..., examples=["OK"])
    ingest_id: str = Field(..., examples=["gt_20260109_101200"])
    gt_version_dir: str = Field(..., examples=["/workspace/storage/GT_versions/GT_gt_20260109_101200"])
    current_gt: str = Field(..., examples=["/workspace/storage/GT/current"])


class UploadGTResponse(BaseModel):
    status: str = Field(..., examples=["OK"])
    ingest_id: str = Field(..., examples=["gt_20260209_153012"])
    extracted_dir: str = Field(..., examples=["/workspace/_tmp/gt_unpack/gt_20260209_153012"])
    n_images: int = Field(..., examples=[842])
    has_labels: bool = Field(..., examples=[True])
    has_annotation: bool = Field(..., examples=[True])


class UploadAndRegisterResponse(BaseModel):
    status: str = Field(..., examples=["OK"])
    ingest_id: str = Field(..., examples=["gt_20260209_153012"])
    extracted_dir: str = Field(..., examples=["/workspace/_tmp/gt_unpack/gt_20260209_153012"])
    n_images: int = Field(..., examples=[842])
    has_labels: bool = Field(..., examples=[True])
    has_annotation: bool = Field(..., examples=[True])
    gt_version_dir: str = Field(..., examples=["/workspace/storage/GT_versions/GT_gt_20260209_153012"])
    current_gt: str = Field(..., examples=["/workspace/storage/GT/current"])


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.post("/upload", response_model=UploadGTResponse)
async def upload_gt_zip(
    file: UploadFile = File(..., description="GT zip (must include images/ and labels/ or annotation(s)/)"),
    ingest_id: Optional[str] = Form(None, description="optional ingest id; if omitted, use filename stem"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    safe_stem = Path(file.filename).stem.replace(" ", "_")
    ing = ingest_id or f"gt_{safe_stem}"

    dst_dir = (GT_UNPACK_ROOT / ing).resolve()
    if dst_dir.exists():
        raise HTTPException(status_code=409, detail=f"ingest_id already exists under unpack root: {dst_dir} (delete it or use a new ingest_id)")

    tmp_zip = (GT_UNPACK_ROOT / f"_{ing}.zip").resolve()
    tmp_zip.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tmp_zip.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        _safe_extract_zip(tmp_zip, dst_dir)
        real_root = _maybe_descend_single_root(dst_dir)
        has_labels, has_ann, n_images = _validate_gt_layout(real_root)

        return UploadGTResponse(
            status="OK",
            ingest_id=ing,
            extracted_dir=str(real_root),
            n_images=n_images,
            has_labels=has_labels,
            has_annotation=has_ann,
        )
    finally:
        tmp_zip.unlink(missing_ok=True)


@router.post("/upload_and_register", response_model=UploadAndRegisterResponse)
async def upload_and_register_gt(
    file: UploadFile = File(..., description="GT zip"),
    ingest_id: Optional[str] = Form(None, description="optional ingest id"),
    copy_mode: Literal["copy", "symlink"] = Form("symlink", description="copy or symlink into versioned GT dir"),
):
    up = await upload_gt_zip(file=file, ingest_id=ingest_id)
    try:
        gt_ver_dir, current_gt = register_gt(
            ingest_id=up.ingest_id,
            extracted_dir=Path(up.extracted_dir),
            copy_mode=copy_mode,
        )
        _postprocess_gt_for_rtm(Path(gt_ver_dir), copy_mode=copy_mode)

    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadAndRegisterResponse(
        status="OK",
        ingest_id=up.ingest_id,
        extracted_dir=up.extracted_dir,
        n_images=up.n_images,
        has_labels=up.has_labels,
        has_annotation=up.has_annotation,
        gt_version_dir=str(gt_ver_dir),
        current_gt=str(current_gt),
    )


@router.post("/register", response_model=RegisterGTResponse)
def register_gt_api(req: RegisterGTRequest):
    try:
        gt_ver_dir, current_gt = register_gt(
            ingest_id=req.ingest_id,
            extracted_dir=Path(req.extracted_dir),
            copy_mode=req.copy_mode,
        )
        _postprocess_gt_for_rtm(Path(gt_ver_dir), copy_mode=req.copy_mode)

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
        current_gt=str(current_gt),
    )
