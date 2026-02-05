from __future__ import annotations

import time
from typing import Any, Dict, List

from src.common.dto.dto import (
    InferBatchRequest,
    Detection,
    InferItemResult,
    ModelPredResponse,
    ModelName,
)

from .model import RTDETRRunner


MODEL_NAME: ModelName = "rtdetr"
_runner = RTDETRRunner()


def infer_batch(req: InferBatchRequest) -> ModelPredResponse:
    """
    SSOT 표준:
    - input:  InferBatchRequest (items[*].image_path, alias=path 지원)
    - output: ModelPredResponse (results: List[InferItemResult])
    """
    t0 = time.time()

    # ✅ 원본 params를 그대로 쓰지 말고 복사(안전)
    params: Dict[str, Any] = dict(req.params or {})
    items = list(req.items or [])
    paths = [it.image_path for it in items]

    try:
        per_img, meta = _runner.predict_batch(paths, params)

        results: List[InferItemResult] = []
        n = min(len(items), len(per_img))

        for it, dets in zip(items[:n], per_img[:n]):
            results.append(
                InferItemResult(
                    image_id=it.image_id,
                    detections=[
                        Detection(
                            cls=int(d.cls),
                            conf=float(d.conf),
                            xyxy=[float(x) for x in list(d.xyxy)],
                        )
                        for d in (dets or [])
                    ],
                )
            )

        # 길이 mismatch 방어: 부족분은 빈 detections로 채움
        if len(results) < len(items):
            done = {r.image_id for r in results}
            for it in items:
                if it.image_id not in done:
                    results.append(InferItemResult(image_id=it.image_id, detections=[]))

        meta_dict = meta if isinstance(meta, dict) else {"note": "meta is not dict"}
        elapsed_ms = int(meta_dict.get("elapsed_ms", int((time.time() - t0) * 1000)))

        weight_used = meta_dict.get("weight_used")
        weight_used = str(weight_used) if weight_used is not None else None

        # ✅ weight_dir_used(스코프)도 전달 (우리 변경의 핵심)
        # (DTO에 field가 없으니 raw에 넣는 방식 유지)
        raw = {
            "meta": meta_dict,
            "n_items": len(items),
        }

        return ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=True,
            error=None,
            elapsed_ms=elapsed_ms,
            weight_used=weight_used,
            results=results,
            raw=raw,
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
