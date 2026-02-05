"""
src/judge/judge_core/judge_entry.py

✅ UPDATED: Judge entry now delegates ensemble logic to src/ensemble/core.py

Responsibilities (keep this file thin):
- load/merge ensemble.yaml knobs (ensemble_cfg)
- call src.ensemble.core.ensemble_bundle(bundle, **knobs)
- convert dict outputs -> DTO (JudgeVerdictRecord / JudgeAttributionRecord)

Notes
- This makes judge use the "real" ensemble engine in src/ensemble/*
- judge_core.match/consensus/thresholds remain for reference but are no longer in the main path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...common.dto.dto import (
    AttributionType,
    GatewayBundle,
    JudgeAttributionRecord,
    JudgeVerdictRecord,
)

# ✅ 핵심: src/ensemble 엔진을 사용
from ...ensemble.core import ensemble_bundle


def _safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        v = d.get(key, default)
    except Exception:
        v = default
    return default if v is None else v


def _build_ensemble_kwargs(ensemble_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    configs/ensemble.yaml -> ensemble_bundle(**kwargs) 변환

    ensemble_bundle signature:
      iou_thr, conf_thr, min_agree, strong_agree,
      min_conf_p3, min_conf_p2, min_pair_iou_p3, min_pair_iou_p2,
      nms_lite_iou, fail_single_top_k_per_model
    """
    if not ensemble_cfg or not isinstance(ensemble_cfg, dict):
        return {}

    ens = ensemble_cfg.get("ensemble") or {}
    if not isinstance(ens, dict):
        return {}

    out: Dict[str, Any] = {}

    # ✅ 1) consensus 기본
    if "iou_thr" in ens:
        out["iou_thr"] = float(ens["iou_thr"])
    if "conf_thr" in ens:
        out["conf_thr"] = float(ens["conf_thr"])
    if "min_agree" in ens:
        out["min_agree"] = int(ens["min_agree"])
    if "strong_agree" in ens:
        out["strong_agree"] = int(ens["strong_agree"])

    # ✅ 2) PASS 판정 기준(튜닝 포인트)
    if "min_conf_p3" in ens:
        out["min_conf_p3"] = float(ens["min_conf_p3"])
    if "min_conf_p2" in ens:
        out["min_conf_p2"] = float(ens["min_conf_p2"])
    if "min_pair_iou_p3" in ens:
        out["min_pair_iou_p3"] = float(ens["min_pair_iou_p3"])
    if "min_pair_iou_p2" in ens:
        out["min_pair_iou_p2"] = float(ens["min_pair_iou_p2"])

    # ✅ 3) FAIL_SINGLE 증거 정리
    if "nms_lite_iou" in ens:
        out["nms_lite_iou"] = float(ens["nms_lite_iou"])
    if "fail_single_top_k_per_model" in ens:
        out["fail_single_top_k_per_model"] = int(ens["fail_single_top_k_per_model"])

    return out


def judge_bundle(
    bundle: GatewayBundle,
    ensemble_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[JudgeVerdictRecord], List[JudgeAttributionRecord]]:
    """
    ✅ Delegating judge to src/ensemble/core.py

    Returns:
      - List[JudgeVerdictRecord] : includes objects/fail_singles/miss_evidence (DTO supports it)
      - List[JudgeAttributionRecord] : FAIL/MISS reasons
    """
    kwargs = _build_ensemble_kwargs(ensemble_cfg)

    verdict_dicts, attr_dicts = ensemble_bundle(
        bundle,
        **kwargs,
    )

    # image_id -> verdict (attribution에 verdict가 없어서 보강)
    verdict_map: Dict[str, str] = {}
    for v in verdict_dicts or []:
        if isinstance(v, dict):
            vid = str(v.get("image_id", ""))
            vv = str(v.get("verdict", ""))
            if vid:
                verdict_map[vid] = vv

    verdicts: List[JudgeVerdictRecord] = []
    for v in verdict_dicts or []:
        if not isinstance(v, dict):
            continue
        verdicts.append(JudgeVerdictRecord(**v))

    attributions: List[JudgeAttributionRecord] = []
    for a in attr_dicts or []:
        if not isinstance(a, dict):
            continue

        image_id = str(a.get("image_id", ""))
        inferred = verdict_map.get(image_id, "FAIL")

        # attribution DTO는 verdict가 반드시 FAIL/MISS여야 함
        if inferred not in ("FAIL", "MISS"):
            # PASS 계열이면 attribution이 생기면 안 되지만, 방어적으로 FAIL로 둠
            inferred = "FAIL"

        reasons_in = a.get("reasons", []) or []
        reasons: List[AttributionType] = []
        for r in reasons_in:
            if isinstance(r, dict):
                reasons.append(AttributionType(**r))

        attributions.append(
            JudgeAttributionRecord(
                image_id=image_id,
                batch_id=str(a.get("batch_id", getattr(bundle, "batch_id", ""))),
                verdict=inferred,
                reasons=reasons,
            )
        )

    return verdicts, attributions
