# src/rtdetr/model.py
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ultralytics import RTDETR  # type: ignore
except Exception:  # pragma: no cover
    RTDETR = None  # type: ignore


@dataclass(frozen=True)
class Det:
    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).resolve()


WEIGHTS_DIR = _env_path("WEIGHTS_DIR", "/weights")
BASE_DIR = WEIGHTS_DIR / "_base" / "rtdetr"

DEFAULT_DEVICE = os.getenv("DEVICE", "1").strip()
DEFAULT_IMG_SZ = int(os.getenv("DEFAULT_IMGSZ", "640"))
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))


def pick_latest_weight(weights_dir: Path, exts: Tuple[str, ...] = (".pt",)) -> Optional[Path]:
    if not weights_dir.exists():
        return None
    cands = [p for p in weights_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_dir(base_dir: Path, requested_dir: Optional[str]) -> Optional[Path]:
    """
    requested_dir:
      - 절대경로 or WEIGHTS_DIR 기준 상대경로
      - 예) "projects/userA/proj1/rtdetr" 또는 "/weights/projects/userA/proj1/rtdetr"
    """
    if not requested_dir:
        return None
    p = Path(str(requested_dir)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p if p.exists() and p.is_dir() else None


def _normalize_device(device: Any) -> str:
    d = "" if device is None else str(device).strip()
    if not d:
        d = DEFAULT_DEVICE

    low = d.lower()
    if low == "cpu":
        return "cpu"
    if low == "cuda":
        return "0"
    if low.startswith("cuda:"):
        return d
    if d.isdigit():
        return d
    return d


class RTDETRRunner:
    """
    Ultralytics RT-DETR runner
    - weight 변경 시 hot reload
    - weight 선택 우선순위:
        1) params["weight"|"weight_path"]
        2) params["weight_dir"|"weights_dir"] (그 dir 안에서 latest)
        3) fallback: weights/_base/rtdetr 최신
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._weight_path: Optional[Path] = None

    def _resolve_weight(self, params: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
        # 1) explicit weight
        requested = params.get("weight") or params.get("weight_path")
        if requested:
            p = Path(str(requested)).expanduser()
            if not p.is_absolute():
                p = (WEIGHTS_DIR / p).resolve()
            return (p if p.exists() and p.is_file() else None), None

        # 2) scoped latest (if provided)
        requested_wdir = params.get("weight_dir") or params.get("weights_dir")
        wdir = _resolve_dir(WEIGHTS_DIR, requested_wdir)

        if requested_wdir:
            # weight_dir를 받았는데 디렉토리가 없으면 base fallback
            if wdir is None:
                base_latest = pick_latest_weight(BASE_DIR, exts=(".pt",))
                return base_latest, str(BASE_DIR)

            latest = pick_latest_weight(wdir, exts=(".pt",))
            if latest is not None:
                return latest, str(wdir)

            # dir는 있는데 비어있으면 base fallback
            base_latest = pick_latest_weight(BASE_DIR, exts=(".pt",))
            return base_latest, str(BASE_DIR)

        # 3) no scope -> base fallback
        base_latest = pick_latest_weight(BASE_DIR, exts=(".pt",))
        return base_latest, str(BASE_DIR)

    def _load_if_needed(self, weight_path: Optional[Path]) -> None:
        if RTDETR is None or weight_path is None:
            self._model = None
            self._weight_path = None
            return

        if self._model is not None and self._weight_path == weight_path:
            return

        self._model = RTDETR(str(weight_path))
        self._weight_path = weight_path

    def predict_batch(
        self,
        image_paths: List[str],
        params: Dict[str, Any],
    ) -> Tuple[List[List[Det]], Dict[str, Any]]:
        t0 = time.time()

        imgsz = int(params.get("imgsz", DEFAULT_IMG_SZ))
        conf = float(params.get("conf", DEFAULT_CONF))
        device = _normalize_device(params.get("device", DEFAULT_DEVICE))

        weight_path, weight_dir_used = self._resolve_weight(params)

        with self._lock:
            self._load_if_needed(weight_path)

            if self._model is None:
                return (
                    [[] for _ in image_paths],
                    {
                        "weight_used": str(weight_path) if weight_path else None,
                        "weight_dir_used": weight_dir_used,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "device": device,
                        "imgsz": imgsz,
                        "conf": conf,
                    },
                )

            results = self._model.predict(
                source=image_paths,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )

        outputs: List[List[Det]] = []
        n = min(len(image_paths), len(results))

        for r in results[:n]:
            dets: List[Det] = []
            try:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    outputs.append(dets)
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()

                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    dets.append(
                        Det(
                            cls=int(k),
                            conf=float(c),
                            xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        )
                    )
            except Exception:
                dets = []
            outputs.append(dets)

        # 결과 길이 보정
        for _ in range(len(image_paths) - len(outputs)):
            outputs.append([])

        return (
            outputs,
            {
                "weight_used": str(weight_path) if weight_path else None,
                "weight_dir_used": weight_dir_used,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "device": device,
                "imgsz": imgsz,
                "conf": conf,
            },
        )
