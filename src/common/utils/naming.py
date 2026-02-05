# src/common/utils/naming.py
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal


# -----------------------------------------------------------------------------
# User key
# -----------------------------------------------------------------------------
def make_user_key(
    email: str,
    *,
    mode: Literal["slug", "hash"] = "slug",
    hash_len: int = 10,
) -> str:
    """
    사용자 이메일을 파일/경로 안전한 user_key로 변환

    mode="slug":
      - kim.haram@gmail.com -> kim_haram_at_gmail_com
    mode="hash":
      - sha1(email) 앞 hash_len 자리

    ⚠️ 운영/보안 환경에서는 hash 권장
    """
    email = email.strip().lower()

    if mode == "hash":
        h = hashlib.sha1(email.encode("utf-8")).hexdigest()
        return h[:hash_len]

    # slug 모드
    s = email
    s = s.replace("@", "_at_")
    s = s.replace(".", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# -----------------------------------------------------------------------------
# Datetime helpers
# -----------------------------------------------------------------------------
def now_ts(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    현재 시각 문자열
    - 기본: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime(fmt)


# -----------------------------------------------------------------------------
# Version helpers
# -----------------------------------------------------------------------------
_VERSION_RE = re.compile(r"_v(\d+)\.(pt|pth)$", re.IGNORECASE)


def extract_version(path: Path) -> Optional[int]:
    """
    파일명에서 vN 추출
    """
    m = _VERSION_RE.search(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def next_version(
    weights_dir: Path,
) -> int:
    """
    주어진 디렉토리에서 vN 최대값을 찾아 +1 반환
    - 파일이 없으면 1
    """
    if not weights_dir.exists():
        return 1

    max_v = 0
    for p in weights_dir.iterdir():
        if not p.is_file():
            continue
        v = extract_version(p)
        if v is not None:
            max_v = max(max_v, v)

    return max_v + 1


# -----------------------------------------------------------------------------
# Weight filename / path
# -----------------------------------------------------------------------------
def make_weight_filename(
    *,
    user_key: str,
    project_id: str,
    ts: Optional[str],
    model: str,
    version: int,
    ext: Literal["pt", "pth"],
) -> str:
    """
    정책 기반 weight 파일명 생성

    형식:
      {UserKey}_{ProjectId}_{YYYYMMDD_HHMMSS}_{Model}_v{N}.pt|pth
    """
    ts = ts or now_ts()
    return f"{user_key}_{project_id}_{ts}_{model}_v{version}.{ext}"


def weight_dir(
    *,
    weights_root: Path,
    user_key: str,
    project_id: str,
    model: str,
) -> Path:
    """
    사용자 GT 학습 weight 디렉토리

    /weights/gt/{user_key}/{project_id}/{model}/
    """
    return weights_root / "gt" / user_key / project_id / model


def baseline_weight_path(
    *,
    weights_root: Path,
    model: str,
    ext: Optional[str] = None,
) -> Path:
    """
    baseline weight 경로

    /weights/baseline/baseline_{model}.*
    """
    base = weights_root / "baseline"
    if ext:
        return base / f"baseline_{model}.{ext}"

    # 확장자 미지정 시 pt/pth 중 존재하는 것 선택
    for e in ("pt", "pth"):
        p = base / f"baseline_{model}.{e}"
        if p.exists():
            return p

    # 기본값(존재 안 해도 반환)
    return base / f"baseline_{model}.pt"


# -----------------------------------------------------------------------------
# Export filename
# -----------------------------------------------------------------------------
def make_export_filename(
    *,
    user_key: str,
    project_id: str,
    ts: Optional[str],
    export_version: int,
) -> str:
    """
    export zip 파일명

    형식:
      {UserKey}_{ProjectId}_{YYYYMMDD_HHMMSS}_export_v{K}.zip
    """
    ts = ts or now_ts()
    return f"{user_key}_{project_id}_{ts}_export_v{export_version}.zip"


# -----------------------------------------------------------------------------
# Convenience: full path creators
# -----------------------------------------------------------------------------
def prepare_gt_weight_path(
    *,
    weights_root: Path,
    user_key: str,
    project_id: str,
    model: str,
    ext: Literal["pt", "pth"],
    ts: Optional[str] = None,
) -> Path:
    """
    다음 GT 학습 weight의 '저장 경로'를 계산
    - 디렉토리 생성까지 수행
    """
    d = weight_dir(
        weights_root=weights_root,
        user_key=user_key,
        project_id=project_id,
        model=model,
    )
    d.mkdir(parents=True, exist_ok=True)

    v = next_version(d)
    fname = make_weight_filename(
        user_key=user_key,
        project_id=project_id,
        ts=ts,
        model=model,
        version=v,
        ext=ext,
    )
    return d / fname
