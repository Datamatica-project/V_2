#!/usr/bin/env python3
"""
register_gt.py (V2)
- extracted_dir 를 GT_versions/GT_<ingest_id> 로 등록
- storage/GT -> 해당 버전으로 current 심볼릭 링크 갱신
- + YOLOX view 생성: yolox/{train2017,val2017,annotations/instances_*.json}

전제(권장):
extracted_dir/
  images/          (또는 images/train, images/val)
  labels/          (YOLO용)
  annotations/     (COCO json 최소 1개)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Literal, Optional, Tuple, List


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
STORAGE = PROJECT_ROOT / "storage"

GT_VERSIONS = STORAGE / "GT_versions"
GT_CURRENT = STORAGE / "GT"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), str(dst))


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
    YOLOX view json에서만 file_name을 '실제 존재하는 파일'에 맞게 보정.
    - 원본 json은 건드리지 않기 위해, 복사본 coco dict를 받아 수정해서 반환.
    - 전략:
      1) file_name 그대로 image_dir/file_name 존재하면 OK
      2) 안 되면 basename으로 바꿔서 image_dir/basename 존재하면 수정
      3) 그래도 안 되면 그대로 둠(나중에 학습에서 터지면 strict 처리 가능)
    """
    if "images" not in coco or not isinstance(coco["images"], list):
        return coco

    imgs: List[dict] = coco["images"]
    # 샘플링은 앞에서부터 sample_k개만 체크
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

        # 추가로 "train/xxx.jpg" 같은 경우도 basename으로 정리되는데
        # 파일이 없다면 그대로 둔다.
        # (여기서 억지로 바꾸면 오히려 틀릴 수 있음)
    return coco


def _make_yolox_view(
    gt_root: Path,
    use_train_as_val: bool = True,
    copy_mode: Literal["copy", "symlink"] = "symlink",
) -> None:
    """
    gt_root (GT_versions/GT_<id>) 아래에 yolox view 생성
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
        val_src = train_src if use_train_as_val else images_dir  # fallback

    yolox_dir = gt_root / "yolox"
    train2017 = yolox_dir / "train2017"
    val2017 = yolox_dir / "val2017"
    yolox_ann = yolox_dir / "annotations"

    # train2017/val2017 link (or copy)
    if train2017.exists() or train2017.is_symlink():
        train2017.unlink()
    if val2017.exists() or val2017.is_symlink():
        val2017.unlink()

    if copy_mode == "copy":
        # copy는 tree 복사가 커서 비추천이지만 옵션 지원은 함
        shutil.copytree(train_src, train2017)
        if use_train_as_val:
            # val은 train과 동일하게 복제
            shutil.copytree(train_src, val2017)
        else:
            shutil.copytree(val_src, val2017)
    else:
        os.symlink(str(train_src), str(train2017))
        os.symlink(str(train_src if use_train_as_val else val_src), str(val2017))

    # annotations json 준비: train json 하나를 골라서 2개로 만든다(이름만 바꾸기 + file_name 정규화)
    src_json = _pick_coco_json(ann_dir)
    coco = _load_json(src_json)

    # train용: train2017 기준으로 file_name 정리
    train_coco = json.loads(json.dumps(coco, ensure_ascii=False))
    train_coco = _normalize_coco_filenames_for_dir(train_coco, train2017)

    # val용: train을 그대로 복제 (use_train_as_val=True)
    val_coco = json.loads(json.dumps(train_coco, ensure_ascii=False))

    _ensure_dir(yolox_ann)
    _dump_json(yolox_ann / "instances_train2017.json", train_coco)
    _dump_json(yolox_ann / "instances_val2017.json", val_coco)


def register_gt(
    ingest_id: str,
    extracted_dir: Path,
    copy_mode: Literal["copy", "symlink"] = "symlink",
    strict: bool = True,
    make_yolox_view: bool = True,
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
    labels = extracted_dir / "labels"
    ann = extracted_dir / "annotations"

    missing = [p.name for p in (images, labels, ann) if not p.exists()]
    if missing and strict:
        raise FileNotFoundError(f"missing required dirs in extracted_dir: {missing}")

    GT_VERSIONS.mkdir(parents=True, exist_ok=True)
    gt_version_dir = GT_VERSIONS / f"GT_{ingest_id}"

    # 1) register version
    _copy_or_symlink_tree(extracted_dir, gt_version_dir, copy_mode=copy_mode)

    # 2) update current GT symlink
    if GT_CURRENT.exists() or GT_CURRENT.is_symlink():
        GT_CURRENT.unlink()
    os.symlink(str(gt_version_dir), str(GT_CURRENT))

    # 3) make YOLOX view
    if make_yolox_view:
        try:
            _make_yolox_view(gt_version_dir, use_train_as_val=use_train_as_val, copy_mode="symlink")
            # (yolox view는 symlink로 고정 추천: 빨라야 함)
        except Exception:
            if strict:
                raise
            # strict=False면 yolox view 실패해도 등록은 유지

    return gt_version_dir
