# src/gateway/clients/rtdetr.py
from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

from ...common.dto.dto import (
    Detection,
    InferBatchRequest,
    InferItemResult,
    ModelPredResponse,
)
from ..http_client import post_json


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {"raw": x}


def _coerce_results(raw: Any) -> List[InferItemResult]:
    """
    모델 원본 응답(raw)에서 표준 results를 최대한 안전하게 추출.
    기대 형태:
      raw["results"] = [
        {"image_id": "...", "detections": [{"cls":0,"conf":0.9,"xyxy":[...]} , ...]},
        ...
      ]
    """
    if not isinstance(raw, dict):
        return []

    r = raw.get("results")
    if not isinstance(r, list):
        return []

    out: List[InferItemResult] = []
    for item in r:
        if not isinstance(item, dict):
            continue

        image_id = item.get("image_id")
        dets = item.get("detections", [])
        if not isinstance(image_id, str) or not isinstance(dets, list):
            continue

        detections: List[Detection] = []
        for d in dets:
            if not isinstance(d, dict):
                continue

            cls = d.get("cls")
            conf = d.get("conf")
            xyxy = d.get("xyxy")

            if not isinstance(cls, int):
                continue
            if not isinstance(conf, (int, float)):
                continue
            if not (
                isinstance(xyxy, list)
                and len(xyxy) == 4
                and all(isinstance(x, (int, float)) for x in xyxy)
            ):
                continue

            detections.append(
                Detection(cls=int(cls), conf=float(conf), xyxy=[float(x) for x in xyxy])
            )

        out.append(InferItemResult(image_id=image_id, detections=detections))

    return out


def _get_meta(raw: Any) -> Dict[str, Any]:
    """
    모델 서버 응답이:
      raw["raw"]["meta"] 형태로 오거나,
      raw["meta"] 형태로 올 수 있음.
    """
    if not isinstance(raw, dict):
        return {}
    # 1) infer 서버의 표준: raw={"meta": {...}}
    m = raw.get("meta")
    if isinstance(m, dict):
        return m
    # 2) gateway client는 raw=_as_dict(raw)로 감싸기도 함
    r2 = raw.get("raw")
    if isinstance(r2, dict):
        m2 = r2.get("meta")
        if isinstance(m2, dict):
            return m2
    return {}


def _coerce_weight_used(raw: Any) -> Optional[str]:
    """
    우선순위:
      1) top-level weight_used
      2) meta.weight_used
    """
    if isinstance(raw, dict):
        w = raw.get("weight_used")
        if w:
            return str(w)
        meta = _get_meta(raw)
        w2 = meta.get("weight_used")
        return str(w2) if w2 else None
    return None


def _coerce_ok(raw: Any) -> bool:
    if isinstance(raw, dict) and "ok" in raw:
        return bool(raw.get("ok"))
    return True


def _coerce_error(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        e = raw.get("error")
        return str(e) if e else None
    return None


def _fill_missing_items(req: InferBatchRequest, results: List[InferItemResult]) -> List[InferItemResult]:
    """
    결과가 일부 누락되었을 때 req.items 기준으로 빈 결과를 채움.
    """
    items = list(req.items or [])
    seen = {r.image_id for r in results}
    fixed = list(results)
    for it in items:
        if it.image_id not in seen:
            fixed.append(InferItemResult(image_id=it.image_id, detections=[]))
    return fixed


def infer_rtdetr(
    req: InferBatchRequest,
    *,
    url: str,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> ModelPredResponse:
    t0 = perf_counter()
    try:
        payload = req.model_dump(by_alias=True)

        raw = post_json(
            url=url,
            payload=payload,
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
        )

        ok = _coerce_ok(raw)
        err = _coerce_error(raw)

        results = _coerce_results(raw)
        results = _fill_missing_items(req, results)

        # ✅ meta를 raw에 유지 (gateway에서 weight_dir_used 로그 가능)
        raw_dict = _as_dict(raw)
        meta = _get_meta(raw_dict)
        if meta:
            raw_dict.setdefault("meta", meta)

        return ModelPredResponse(
            model="rtdetr",
            batch_id=req.batch_id,
            ok=ok,
            error=err if not ok else None,
            elapsed_ms=int((perf_counter() - t0) * 1000),
            weight_used=_coerce_weight_used(raw),
            results=results,
            raw=raw_dict,
        )

    except Exception as e:
        return ModelPredResponse(
            model="rtdetr",
            batch_id=req.batch_id,
            ok=False,
            error=str(e),
            elapsed_ms=int((perf_counter() - t0) * 1000),
            weight_used=None,
            results=_fill_missing_items(req, []),
            raw={},
        )
