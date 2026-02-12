"""
register_gt.py (V2) - RTM view safe version

- extracted_dir 를 GT_versions/GT_<ingest_id> 로 "등록"(기본: copy)
- storage/GT/current -> 해당 버전으로 current 심볼릭 링크 갱신 (상대 심링크 권장)
- + RTM view 생성: rtm/images/{train,val}, rtm/annotations/instances_{train,val}.json

전제(권장):
gt_root/
  images/          (또는 images/train, images/val)
  labels/          (YOLO txt; 선택적)
  annotations/     (COCO json 최소 1개)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Literal, Tuple, List


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace")).resolve()
STORAGE = (PROJECT_ROOT / "storage").resolve()

GT_VERSIONS = (STORAGE / "GT_versions").resolve()
GT_DIR = (STORAGE / "GT").resolve()
GT_CURRENT = (GT_DIR / "current").resolve()

CopyMode = Literal["copy", "symlink"]  # NOTE: symlink는 extracted_dir 수명/마운트에 민감함


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rm_any(p: Path) -> None:
    if not p.exists() and not p.is_symlink():
        return
    if p.is_symlink() or p.is_file():
        p.unlink()
        return
    if p.is_dir():
        shutil.rmtree(p)
        return


def _list_top_dirs(p: Path) -> list[Path]:
    if not p.exists():
        return []
    return [x for x in p.iterdir() if x.is_dir() and x.name != "__MACOSX"]


def _maybe_descend_single_root(extracted_dir: Path) -> Path:
    """
    ZIP 안에 단일 루트 폴더가 있고 그 아래 images/ 가 있는 경우,
    그 루트로 한 단계 내려가서 반환.
    """
    dirs = _list_top_dirs(extracted_dir)
    if len(dirs) == 1 and (dirs[0] / "images").exists():
        return dirs[0]
    return extracted_dir


def _copytree_or_symlink(src: Path, dst: Path, copy_mode: CopyMode) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing path: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"target already exists: {dst}")

    if copy_mode == "copy":
        shutil.copytree(src, dst)
    else:
        os.symlink(str(src), str(dst), target_is_directory=True)


def _symlink_dir_relative(link_path: Path, target_dir: Path) -> None:
    """
    link_path 위치에서 target_dir로 가는 상대 심링크를 생성.
    """
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    rel = os.path.relpath(str(target_dir), start=str(link_path.parent))
    os.symlink(rel, str(link_path), target_is_directory=True)


def _pick_coco_json(ann_dir: Path) -> Path:
    if not ann_dir.exists():
        raise FileNotFoundError(f"missing annotations dir: {ann_dir}")

    cands = sorted([p for p in ann_dir.glob("*.json") if p.is_file()])
    if not cands:
        raise FileNotFoundError(f"no *.json in: {ann_dir}")

    def score(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        return (1 if "train" in name else 0, 1 if "instances" in name else 0)

    cands.sort(key=score, reverse=True)
    return cands[0]


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _dump_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _normalize_coco_filenames_for_dir(coco: dict, image_dir: Path, sample_k: int = 50) -> dict:
    """
    RTM view json에서만 file_name을 '실제 존재하는 파일'에 맞게 보정.
    원본 json은 건드리지 않기 위해 복사본 dict를 받아 수정/반환.
    """
    if "images" not in coco or not isinstance(coco["images"], list):
        return coco

    imgs: List[dict] = coco["images"]
    for im in imgs[: max(1, min(sample_k, len(imgs)))]:
        fn = im.get("file_name")
        if not isinstance(fn, str) or not fn:
            continue

        p1 = image_dir / fn
        if p1.exists():
            continue

        base = Path(fn).name
        p2 = image_dir / base
        if p2.exists():
            im["file_name"] = base
            continue

    return coco


def _make_rtm_view(
    gt_root: Path,
    use_train_as_val: bool = True,
    copy_mode: CopyMode = "symlink",
) -> None:
    """
    gt_root 아래에 rtm view 생성:
      - rtm/images/train, rtm/images/val
      - rtm/annotations/instances_train.json, instances_val.json
    """
    images_dir = gt_root / "images"
    ann_dir = gt_root / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"missing images dir: {images_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"missing annotations dir: {ann_dir}")

    train_src = images_dir / "train" if (images_dir / "train").exists() else images_dir
    val_src = images_dir / "val" if (images_dir / "val").exists() else (train_src if use_train_as_val else images_dir)

    rtm_dir = gt_root / "rtm"
    rtm_images = rtm_dir / "images"
    rtm_train = rtm_images / "train"
    rtm_val = rtm_images / "val"
    rtm_ann = rtm_dir / "annotations"

    _ensure_dir(rtm_images)
    _ensure_dir(rtm_ann)

    for p in (rtm_train, rtm_val):
        _rm_any(p)

    if copy_mode == "copy":
        shutil.copytree(train_src, rtm_train)
        shutil.copytree(train_src if use_train_as_val else val_src, rtm_val)
    else:
        os.symlink(str(train_src), str(rtm_train), target_is_directory=True)
        os.symlink(str(train_src if use_train_as_val else val_src), str(rtm_val), target_is_directory=True)

    src_json = _pick_coco_json(ann_dir)
    coco = _load_json(src_json)

    train_coco = json.loads(json.dumps(coco, ensure_ascii=False))
    train_coco = _normalize_coco_filenames_for_dir(train_coco, rtm_train)

    val_coco = json.loads(json.dumps(train_coco, ensure_ascii=False))
    val_coco = _normalize_coco_filenames_for_dir(val_coco, rtm_val)

    _dump_json(rtm_ann / "instances_train.json", train_coco)
    _dump_json(rtm_ann / "instances_val.json", val_coco)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def register_gt(
    ingest_id: str,
    extracted_dir: Path,
    copy_mode: CopyMode = "copy",
    strict: bool = True,
    make_rtm_view: bool = True,
    use_train_as_val: bool = True,
) -> tuple[Path, Path]:
    """
    returns:
      (gt_version_dir, current_gt_path)

    - gt_version_dir: storage/GT_versions/GT_<ingest_id>
    - current_gt_path: storage/GT/current (심링크)
    """
    if not ingest_id:
        raise ValueError("ingest_id is empty")

    extracted_dir = extracted_dir.resolve()
    if not extracted_dir.exists():
        raise FileNotFoundError(f"extracted_dir not found: {extracted_dir}")

    extracted_dir = _maybe_descend_single_root(extracted_dir)

    images = extracted_dir / "images"
    ann = extracted_dir / "annotations"
    labels = extracted_dir / "labels"

    missing_required = [p.name for p in (images, ann) if not p.exists()]
    if missing_required and strict:
        raise FileNotFoundError(f"missing required dirs in extracted_dir: {missing_required}")

    if strict and (not labels.exists()):
        raise FileNotFoundError("missing labels dir in extracted_dir (strict=True)")

    _ensure_dir(GT_VERSIONS)
    _ensure_dir(GT_DIR)

    gt_version_dir = (GT_VERSIONS / f"GT_{ingest_id}").resolve()

    if gt_version_dir.exists() or gt_version_dir.is_symlink():
        raise FileExistsError(f"GT version already exists: {gt_version_dir}")

    # 1) register version
    _copytree_or_symlink(extracted_dir, gt_version_dir, copy_mode=copy_mode)

    # 2) update current symlink: storage/GT/current -> GT_versions/GT_<id>
    _symlink_dir_relative(GT_CURRENT, gt_version_dir)

    # 3) make RTM view (always symlink recommended for speed)
    if make_rtm_view:
        try:
            _make_rtm_view(gt_version_dir, use_train_as_val=use_train_as_val, copy_mode="symlink")
        except Exception:
            if strict:
                raise

    return gt_version_dir, GT_CURRENT
