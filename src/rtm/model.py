# src/rtm/model.py
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from mmdet.apis import init_detector, inference_detector  # type: ignore
except Exception:  # pragma: no cover
    init_detector = None  # type: ignore
    inference_detector = None  # type: ignore


@dataclass(frozen=True)
class Det:
    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).resolve()


WEIGHTS_DIR = _env_path("WEIGHTS_DIR", "/weights")
BASE_DIR = WEIGHTS_DIR / "_base" / "rtm"
DEFAULT_DEVICE = os.getenv("DEVICE", "cuda:0").strip()
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))
DEFAULT_FP16 = os.getenv("RTM_FP16", "0").strip().lower() in ("1", "true", "yes", "y")
DEFAULT_RTM_CONFIG = os.getenv("RTM_CONFIG", "").strip()


def pick_latest_weight(weights_dir: Path, exts: Tuple[str, ...] = (".pth", ".pt")) -> Optional[Path]:
    if not weights_dir.exists():
        return None
    cands = [p for p in weights_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_dir(base_dir: Path, requested_dir: Optional[str]) -> Optional[Path]:
    if not requested_dir:
        return None
    p = Path(str(requested_dir)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p if p.exists() and p.is_dir() else None


def _normalize_device(device: Optional[str]) -> str:
    d = (device or "").strip()
    if not d:
        d = DEFAULT_DEVICE
    if d.lower() == "cpu":
        return "cpu"
    if d.isdigit():
        return f"cuda:{d}"
    if d.lower() == "cuda":
        return "cuda:0"
    return d


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


class RTMRunner:
    """
    RTMDet runner (mmdetection)
    weight 선택 우선순위:
      1) params.weight / weight_path
      2) params.weight_dir (scoped latest)
      3) fallback: WEIGHTS_DIR 전체 latest
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None

        self._weight_path: Optional[Path] = None
        self._weight_dir_used: Optional[str] = None
        self._config_path: Optional[str] = None
        self._device: str = _normalize_device(DEFAULT_DEVICE)
        self._fp16: bool = DEFAULT_FP16
        self._conf: float = DEFAULT_CONF

    def _can_run(self) -> bool:
        return torch is not None and init_detector is not None and inference_detector is not None

    def _resolve_weight(self, params: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
        # 1) explicit weight
        requested = params.get("weight") or params.get("weights") or params.get("weight_path")
        if requested:
            p = Path(str(requested))
            if not p.is_absolute():
                p = (WEIGHTS_DIR / p).resolve()
            return (p if p.exists() else None), None

        requested_wdir = params.get("weight_dir") or params.get("weights_dir")
        wdir = _resolve_dir(WEIGHTS_DIR, requested_wdir)

        if requested_wdir:
            # weight_dir를 받았는데 디렉토리가 없으면 base fallback
            if wdir is None:
                base_latest = pick_latest_weight(BASE_DIR)
                return base_latest, str(BASE_DIR)

            latest = pick_latest_weight(wdir)
            if latest is not None:
                return latest, str(wdir)

            # dir는 있는데 비어있으면 base fallback
            base_latest = pick_latest_weight(BASE_DIR)
            return base_latest, str(BASE_DIR)

        # scope가 없으면 base fallback
        base_latest = pick_latest_weight(BASE_DIR)
        return base_latest, str(BASE_DIR)

    def _resolve_config(self, params: Dict[str, Any]) -> Optional[str]:
        cfg = str(params.get("config") or params.get("config_file") or DEFAULT_RTM_CONFIG).strip()
        return cfg or None

    def _load_if_needed(self, weight_path: Optional[Path], weight_dir_used: Optional[str], params: Dict[str, Any]) -> None:
        if not self._can_run() or weight_path is None:
            self._model = None
            self._weight_path = None
            return

        config_path = self._resolve_config(params)
        if not config_path:
            self._model = None
            return

        device = _normalize_device(params.get("device", self._device))
        fp16 = _truthy(params.get("fp16", self._fp16))
        conf = float(params.get("conf", params.get("score_thr", self._conf)))

        if (
            self._model is not None
            and self._weight_path == weight_path
            and self._config_path == config_path
            and self._device == device
            and self._fp16 == fp16
            and abs(self._conf - conf) < 1e-9
        ):
            return

        model = init_detector(config_path, str(weight_path), device=device)

        if fp16 and device != "cpu" and torch is not None:
            try:
                model = model.half()
            except Exception:
                pass

        self._model = model
        self._weight_path = weight_path
        self._weight_dir_used = weight_dir_used
        self._config_path = config_path
        self._device = device
        self._fp16 = fp16
        self._conf = conf

    def predict_batch(self, image_paths: List[str], params: Dict[str, Any]) -> Tuple[List[List[Det]], Dict[str, Any]]:
        t0 = time.time()

        conf_thre = float(params.get("conf", params.get("score_thr", DEFAULT_CONF)))
        weight_path, weight_dir_used = self._resolve_weight(params)

        with self._lock:
            self._load_if_needed(weight_path, weight_dir_used, params)

            if self._model is None or not self._can_run():
                return (
                    [[] for _ in image_paths],
                    {
                        "weight_used": str(weight_path) if weight_path else None,
                        "weight_dir_used": weight_dir_used,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "device": _normalize_device(params.get("device", DEFAULT_DEVICE)),
                        "conf": conf_thre,
                        "config": self._resolve_config(params),
                        "fp16": _truthy(params.get("fp16", DEFAULT_FP16)),
                    },
                )

            model = self._model

        outputs: List[List[Det]] = []
        for p in image_paths:
            try:
                out = inference_detector(model, p)
                dets = []
                # (기존 decode 로직 유지)
                for cls_id, boxes in enumerate(out if isinstance(out, list) else []):
                    if boxes is None:
                        continue
                    for row in boxes:
                        if row is None or len(row) < 5:
                            continue
                        x1, y1, x2, y2, score = map(float, row[:5])
                        if score < conf_thre:
                            continue
                        dets.append(Det(cls=cls_id, conf=score, xyxy=(x1, y1, x2, y2)))
                outputs.append(dets)
            except Exception:
                outputs.append([])

        return (
            outputs,
            {
                "weight_used": str(weight_path) if weight_path else None,
                "weight_dir_used": weight_dir_used,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "device": self._device,
                "conf": conf_thre,
                "config": self._config_path,
                "fp16": self._fp16,
            },
        )
