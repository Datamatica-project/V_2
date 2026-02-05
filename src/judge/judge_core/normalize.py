"""
judge_core.normalize

Convert per-model raw outputs to a common detection format.

Design goals:
- Be permissive: accept minor schema differences.
- Be deterministic: stable sort of detections.
- Never crash the entire batch: return empty list on malformed parts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...common.dto.dto import ModelName, ModelPredResponse
from .types import Det


def _to_xyxy(v: Any) -> Tuple[float, float, float, float] | None:
    """Accept xyxy, xywh, dict formats. Returns xyxy or None."""
    if v is None:
        return None

    if isinstance(v, (list, tuple)) and len(v) == 4:
        a, b, c, d = v
        try:
            a = float(a)
            b = float(b)
            c = float(c)
            d = float(d)
        except Exception:
            return None
        # heuristic: if c,d look like x2,y2 (>= a,b), treat as xyxy else xywh
        if c >= a and d >= b:
            return (a, b, c, d)
        return (a, b, a + c, b + d)

    if isinstance(v, dict):
        if all(k in v for k in ("x1", "y1", "x2", "y2")):
            try:
                return (float(v["x1"]), float(v["y1"]), float(v["x2"]), float(v["y2"]))
            except Exception:
                return None
        if all(k in v for k in ("x", "y", "w", "h")):
            try:
                x = float(v["x"])
                y = float(v["y"])
                w = float(v["w"])
                h = float(v["h"])
            except Exception:
                return None
            return (x, y, x + w, y + h)

    return None


def parse_model_raw_to_image_dets(resp: ModelPredResponse) -> Dict[str, List[Det]]:
    """Parse ModelPredResponse.raw into {image_id: [Det,...]}.

    Supported raw schemas (best-effort):
    1) {"results": [{"image_id": str, "detections": [{cls, conf, xyxy}, ...]}, ...]}
    2) {"preds":   [{"image_id": str, "detections": [...]}, ...]}
    3) {"image_id": str, "detections": [...]}

    A detection item may use keys:
    - cls / class_id / class
    - conf / score / confidence
    - xyxy / bbox / box
    """
    raw: Dict[str, Any] = resp.raw or {}
    model: ModelName = resp.model

    def parse_one_det(d: Dict[str, Any]) -> Det | None:
        if not isinstance(d, dict):
            return None
        cls = d.get("cls", d.get("class_id", d.get("class", None)))
        conf = d.get("conf", d.get("score", d.get("confidence", 0.0)))
        xyxy = d.get("xyxy", d.get("bbox", d.get("box", None)))
        xyxy2 = _to_xyxy(xyxy)
        if cls is None or xyxy2 is None:
            return None
        try:
            cls_i = int(cls)
            conf_f = float(conf)
        except Exception:
            return None
        return Det(model=model, cls=cls_i, conf=conf_f, xyxy=xyxy2)

    def _parse_det_list(dets_any: Any) -> List[Det]:
        dets: List[Det] = []
        if not isinstance(dets_any, list):
            return dets
        for d in dets_any:
            det = parse_one_det(d)
            if det is not None:
                dets.append(det)
        dets.sort(key=lambda x: (-x.conf, x.cls, x.xyxy[0], x.xyxy[1]))
        return dets

    def parse_result_list(lst: Any) -> Dict[str, List[Det]]:
        out: Dict[str, List[Det]] = {}
        if not isinstance(lst, list):
            return out
        for r in lst:
            if not isinstance(r, dict):
                continue
            image_id = r.get("image_id") or r.get("id") or r.get("image")
            dets = r.get("detections") or r.get("predictions") or r.get("dets") or []
            if not image_id:
                continue
            out[str(image_id)] = _parse_det_list(dets)
        return out

    if isinstance(raw, dict):
        if "results" in raw:
            return parse_result_list(raw.get("results"))
        if "preds" in raw:
            return parse_result_list(raw.get("preds"))
        if "image_id" in raw and ("detections" in raw or "predictions" in raw):
            image_id = raw.get("image_id")
            dets_any = raw.get("detections") or raw.get("predictions")
            return {str(image_id): _parse_det_list(dets_any)}

    return {}
