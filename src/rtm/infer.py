from __future__ import annotations

import time
from typing import Any, Dict, List

from src.common.dto.dto import (
    InferBatchRequest,
    ModelName,
    Detection,
    InferItemResult,
    ModelPredResponse,
)

from .model import RTMRunner  # Det는 여기서 직접 타입으로 안 씀

MODEL_NAME: ModelName = "rtm"
_runner = RTMRunner()


def _ensure_len(results: List[InferItemResult], items: List[Any]) -> List[InferItemResult]:
    """요청 items 길이만큼 결과 보장 (누락 방지)"""
    seen = {r.image_id for r in results}
    for it in items:
        if it.image_id not in seen:
            results.append(InferItemResult(image_id=it.image_id, detections=[]))
    return results


def infer_batch(req: InferBatchRequest) -> ModelPredResponse:
    """
    ✅ SSOT 표준화된 RTM infer entry
    - input : InferBatchRequest
    - output: ModelPredResponse
    - 모델 로직은 .model.RTMRunner 에 위임
    """
    t0 = time.time()

    # ✅ 원본 params 공유 방지
    params: Dict[str, Any] = dict(req.params or {})
    items = list(req.items or [])
    image_paths = [it.image_path for it in items]

    try:
        per_img, meta = _runner.predict_batch(image_paths, params)

        results: List[InferItemResult] = []
        n = min(len(items), len(per_img))

        for it, dets in zip(items[:n], per_img[:n]):
            detections: List[Detection] = []
            for d in (dets or []):
                try:
                    xyxy = list(d.xyxy)  # tuple/list/numpy -> list
                except Exception:
                    continue
                if len(xyxy) != 4:
                    continue

                detections.append(
                    Detection(
                        cls=int(d.cls),
                        conf=float(d.conf),
                        xyxy=[float(x) for x in xyxy],
                    )
                )

            results.append(InferItemResult(image_id=it.image_id, detections=detections))

        results = _ensure_len(results, items)

        meta_dict = meta if isinstance(meta, dict) else {"note": "meta is not dict"}
        elapsed_ms = int(meta_dict.get("elapsed_ms", int((time.time() - t0) * 1000)))

        weight_used = meta_dict.get("weight_used")
        weight_used = str(weight_used) if weight_used is not None else None

        return ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=True,
            error=None,
            elapsed_ms=elapsed_ms,
            weight_used=weight_used,
            results=results,
            raw={
                "meta": meta_dict,          # ✅ 여기 안에 weight_dir_used도 들어옴
                "n_items": len(items),
            },
        )

    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        empty_results = [InferItemResult(image_id=it.image_id, detections=[]) for it in items]
        return ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=elapsed_ms,
            weight_used=None,
            results=empty_results,
            raw={},
        )
