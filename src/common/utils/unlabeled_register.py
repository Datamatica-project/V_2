from __future__ import annotations

from pathlib import Path
from typing import Literal

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def register_unlabeled(
    *,
    unlabeled_id: str,
    source_dir: Path,
    target_root: Path,
    link_name: str = "UNLABELED",
    copy_mode: Literal["symlink", "copy"] = "symlink",
) -> Path:
    """
    Register unlabeled image directory and update current UNLABELED pointer.

    Returns:
        Path to the registered unlabeled directory
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    images = [p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError(f"no image files found in {source_dir}")

    target_root.mkdir(parents=True, exist_ok=True)
    version_dir = target_root / unlabeled_id

    if version_dir.exists():
        raise FileExistsError(f"unlabeled_id already exists: {unlabeled_id}")

    if copy_mode == "copy":
        version_dir.mkdir()
        for img in images:
            (version_dir / img.name).write_bytes(img.read_bytes())
    else:
        version_dir.symlink_to(source_dir.resolve(), target_is_directory=True)

    # current pointer
    current_link = target_root / link_name
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(version_dir, target_is_directory=True)

    return version_dir
