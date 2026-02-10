#!/usr/bin/env python3
"""
register_gt.py (V2) - RTM view 버전

- extracted_dir 를 GT_versions/GT_<ingest_id> 로 등록
- storage/GT -> 해당 버전으로 current 심볼릭 링크 갱신
- + RTM view 생성: rtm/images/{train,val}, rtm/annotations/instances_{train,val}.json

전제(권장):
extracted_dir/
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


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
STORAGE = PROJECT_ROOT / "storage"

GT_VERSIONS = STORAGE / "GT_versions"
GT_CURRENT = STORAGE / "GT"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_or_symlink_tree(src: Path, dst: Path, copy_mode: Literal["copy", "symlink"]) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing path: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"target already exists: {dst}")

    if copy_mode == "copy":
        shutil.copytree(src, dst)
    else:
        os.symlink(str(src), str(dst))


def _pick_coco_json(ann_dir: Path) -> Path:
    if not ann_dir.exists():
        raise FileNotFoundError(f"missing annotations dir: {ann_dir}")

    cands = sorted([p for p in ann_dir.glob("*.json") if p.is_file()])
    if not cands:
        raise FileNotFoundError(f"no *.json in: {ann_dir}")

    # 우선순위: train 포함 > instances 포함 > 첫 번째
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
    - 원본 json은 건드리지 않기 위해, 복사본 coco dict를 받아 수정해서 반환.
    - 전략:
      1) file_name 그대로 image_dir/file_name 존재하면 OK
      2) 안 되면 basename으로 바꿔서 image_dir/basename 존재하면 수정
      3) 그래도 안 되면 그대로 둠
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
    copy_mode: Literal["copy", "symlink"] = "symlink",
) -> None:
    """
    gt_root (GT_versions/GT_<id>) 아래에 RTM view 생성
      - rtm/images/train, rtm/images/val
      - rtm/annotations/instances_train.json, instances_val.json
    """
    images_dir = gt_root / "images"
    ann_dir = gt_root / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"missing images dir: {images_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"missing annotations dir: {ann_dir}")

    # images layout: images/train, images/val 이 있으면 그것을 우선 사용
    train_src = images_dir / "train" if (images_dir / "train").exists() else images_dir
    if (images_dir / "val").exists():
        val_src = images_dir / "val"
    else:
        val_src = train_src if use_train_as_val else images_dir

    rtm_dir = gt_root / "rtm"
    rtm_images = rtm_dir / "images"
    rtm_train = rtm_images / "train"
    rtm_val = rtm_images / "val"
    rtm_ann = rtm_dir / "annotations"

    _ensure_dir(rtm_images)
    _ensure_dir(rtm_ann)

    # 기존 링크/폴더 정리(있으면 삭제)
    for p in (rtm_train, rtm_val):
        if p.exists() or p.is_symlink():
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()

    # train/val link (or copy)
    if copy_mode == "copy":
        shutil.copytree(train_src, rtm_train)
        if use_train_as_val:
            shutil.copytree(train_src, rtm_val)
        else:
            shutil.copytree(val_src, rtm_val)
    else:
        os.symlink(str(train_src), str(rtm_train))
        os.symlink(str(train_src if use_train_as_val else val_src), str(rtm_val))

    # annotations json 준비: train json 하나를 골라서 2개로 만든다(파일명만 + file_name 정규화)
    src_json = _pick_coco_json(ann_dir)
    coco = _load_json(src_json)

    train_coco = json.loads(json.dumps(coco, ensure_ascii=False))
    train_coco = _normalize_coco_filenames_for_dir(train_coco, rtm_train)

    val_coco = json.loads(json.dumps(train_coco, ensure_ascii=False))
    val_coco = _normalize_coco_filenames_for_dir(val_coco, rtm_val)

    _dump_json(rtm_ann / "instances_train.json", train_coco)
    _dump_json(rtm_ann / "instances_val.json", val_coco)


def register_gt(
    ingest_id: str,
    extracted_dir: Path,
    copy_mode: Literal["copy", "symlink"] = "symlink",
    strict: bool = True,
    make_rtm_view: bool = True,
    use_train_as_val: bool = True,
) -> Path:
    """
    returns: gt_version_dir (e.g., storage/GT_versions/GT_<ingest_id>)
    """
    if not ingest_id:
        raise ValueError("ingest_id is empty")
    if not extracted_dir.exists():
        raise FileNotFoundError(f"extracted_dir not found: {extracted_dir}")

    # basic validation
    images = extracted_dir / "images"
    ann = extracted_dir / "annotations"
    # labels는 YOLO용이므로 "권장"이지만, COCO만 들어오는 케이스도 있어서 hard-required는 상황에 따라
    labels = extracted_dir / "labels"

    missing_required = [p.name for p in (images, ann) if not p.exists()]
    if missing_required and strict:
        raise FileNotFoundError(f"missing required dirs in extracted_dir: {missing_required}")

    if strict and (not labels.exists()):
        # strict에서 labels까지 강제하고 싶으면 이 라인을 유지
        # YOLO 라벨이 필수가 아니라면 아래 2줄을 주석 처리하면 됨.
        raise FileNotFoundError("missing labels dir in extracted_dir (strict=True)")

    GT_VERSIONS.mkdir(parents=True, exist_ok=True)
    gt_version_dir = GT_VERSIONS / f"GT_{ingest_id}"

    # 1) register version
    _copy_or_symlink_tree(extracted_dir, gt_version_dir, copy_mode=copy_mode)

    # 2) update current GT symlink
    if GT_CURRENT.exists() or GT_CURRENT.is_symlink():
        GT_CURRENT.unlink()
    os.symlink(str(gt_version_dir), str(GT_CURRENT))

    # 3) make RTM view
    if make_rtm_view:
        try:
            _make_rtm_view(gt_version_dir, use_train_as_val=use_train_as_val, copy_mode="symlink")
        except Exception:
            if strict:
                raise
    return gt_version_dir
