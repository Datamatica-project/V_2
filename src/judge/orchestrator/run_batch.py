# src/judge/orchestrator/run_batch.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

# ✅ logging 실제 위치: common/utils/logging.py
from ...common.utils.logging import setup_logger, LogContext, log_event

from ...common.utils.unlabeled_index import scan_unlabeled
from ...common.utils.batch import make_batches
from ...common.dto.dto import InferBatchRequest

from ...gateway.bundler import infer_bundle

from .status import init_status, update_status, mark_done, mark_failed

from ..registry.sink import append_verdict, append_attribution
from ...ensemble.core import ensemble_bundle  # <-- 핵심 변경


logger = setup_logger(
    service="judge",
    logger_name="judge.run_batch",  # ✅ 모듈별 로거 분리
    log_file="/workspace/logs/judge/run_batch.jsonl",
    level="INFO",
)


def _count_verdicts(verdicts: List[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in verdicts or []:
        if isinstance(v, dict):
            key = v.get("verdict") or v.get("status") or v.get("label")
        else:
            key = getattr(v, "verdict", None) or getattr(v, "status", None) or getattr(v, "label", None)
        key = str(key) if key else "UNKNOWN"
        out[key] = out.get(key, 0) + 1
    return out


def _build_ensemble_kwargs(ensemble_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    configs/ensemble.yaml -> ensemble_bundle(**kwargs) 인자로 변환
    """
    cfg = ensemble_cfg or {}
    ens = cfg.get("ensemble") or {}
    if not isinstance(ens, dict):
        ens = {}

    keys = [
        "iou_thr",
        "conf_thr",
        "min_agree",
        "strong_agree",
        "min_conf_p3",
        "min_conf_p2",
        "min_pair_iou_p3",
        "min_pair_iou_p2",
        "nms_lite_iou",
        "fail_single_top_k_per_model",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in ens:
            out[k] = ens[k]
    return out


def run_batch(
    *,
    run_id: str,
    unlabeled_dir: str,
    batch_size: int,
    model_cfg: Dict[str, Any],
    infer_params: Optional[Dict[str, Any]] = None,
    segment_size_batches: int = 100,
    ensemble_cfg: Optional[Dict[str, Any]] = None,  # ✅ server.py에서 넘어옴
) -> None:
    ctx = LogContext(run_id=run_id, stage="run_batch", model="judge")
    t_run0 = time.time()

    try:
        items = scan_unlabeled(unlabeled_dir)
        total_images = len(items)

        init_status(run_id, total_images)

        ensemble_kwargs = _build_ensemble_kwargs(ensemble_cfg)

        log_event(
            logger,
            "run start",
            ctx=ctx,
            unlabeled_dir=unlabeled_dir,
            total_images=total_images,
            batch_size=batch_size,
            segment_size_batches=segment_size_batches,
            infer_params_keys=list((infer_params or {}).keys()),
            ensemble_kwargs=ensemble_kwargs,
        )

        processed = 0
        progress_every_n_batch = int(os.getenv("PROGRESS_EVERY_N_BATCH", "10"))

        for batch_index, (batch_id, batch_items) in enumerate(make_batches(items, batch_size)):
            t_b0 = time.time()

            params = dict(infer_params or {})
            params["run_id"] = run_id

            req = InferBatchRequest(
                batch_id=batch_id,
                items=batch_items,
                params=params,
            )

            # 1) 모델 fan-out
            bundle = infer_bundle(req, model_cfg=model_cfg)

            # 2) 앙상블
            verdicts, attributions = ensemble_bundle(
                bundle,
                **ensemble_kwargs,
            )

            elapsed_b_ms = int((time.time() - t_b0) * 1000)
            verdict_counts = _count_verdicts(verdicts)

            log_event(
                logger,
                "batch done",
                ctx=ctx,
                batch_index=batch_index,
                batch_id=batch_id,
                n_items=len(batch_items),
                elapsed_ms=elapsed_b_ms,
                verdict_counts=verdict_counts,
                n_attributions=len(attributions) if attributions is not None else 0,
                has_model_errors=bool(getattr(bundle, "errors", None)),
            )

            # 3) 저장 (segment 단위)
            for v in verdicts or []:
                append_verdict(run_id, batch_index, v, segment_size_batches=segment_size_batches)

            for a in attributions or []:
                append_attribution(run_id, batch_index, a, segment_size_batches=segment_size_batches)

            processed += len(batch_items)
            update_status(run_id, processed, batch_id)

            if progress_every_n_batch > 0 and (batch_index % progress_every_n_batch) == 0:
                log_event(
                    logger,
                    "run progress",
                    ctx=ctx,
                    batch_index=batch_index,
                    processed=processed,
                    total_images=total_images,
                    last_batch_id=batch_id,
                )

        mark_done(run_id)
        total_ms = int((time.time() - t_run0) * 1000)

        log_event(
            logger,
            "run done",
            ctx=ctx,
            total_images=total_images,
            elapsed_ms=total_ms,
        )

    except Exception as e:
        mark_failed(run_id, reason=str(e))
        logger.exception(
            "run failed",
            extra=ctx.as_extra() | {"error": f"{type(e).__name__}: {e}"},
        )
        raise
