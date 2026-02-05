# src/common/utils/unlabeled_index.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Set, Tuple

import cv2  # ✅ opencv 강제 사용 (컨테이너에 설치 전제)

# ✅ DTO 단일화
from ..dto.dto import InferItem

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ImageIdMode = Literal["path", "content"]


def _hash_bytes(b: bytes, n: int = 16) -> str:
    return hashlib.blake2b(b, digest_size=n).hexdigest()


def _image_id_from_path(root: Path, abs_path: Path) -> str:
    rel = abs_path.relative_to(root).as_posix().encode("utf-8")
    return _hash_bytes(rel)


def _image_id_from_content(abs_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.blake2b(digest_size=16)
    with abs_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_wh(abs_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    opencv 기반 이미지 크기 로드
    - 실패 시 (None, None)
    """
    try:
        img = cv2.imread(str(abs_path))
        if img is None:
            return None, None
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        return None, None


def scan_unlabeled(
    unlabeled_dir: str | Path,
    *,
    id_mode: ImageIdMode = "path",
    read_size: bool = False,
    recursive: bool = True,
) -> List[InferItem]:
    root = Path(unlabeled_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"unlabeled_dir not found or not a dir: {root}")

    it: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")

    items: List[InferItem] = []
    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue

        abs_path = p.resolve()

        if id_mode == "path":
            image_id = _image_id_from_path(root, abs_path)
        elif id_mode == "content":
            image_id = _image_id_from_content(abs_path)
        else:
            raise ValueError(f"unknown id_mode: {id_mode}")

        w, h = (None, None)
        if read_size:
            w, h = _read_wh(abs_path)

        items.append(
            InferItem(
                image_id=image_id,
                image_path=str(abs_path),
                width=w,
                height=h,
            )
        )

    items.sort(key=lambda x: x.image_path)
    return items
