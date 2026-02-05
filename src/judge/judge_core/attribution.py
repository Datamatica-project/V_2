"""
judge_core.attribution

Break down FAIL reasons (heuristics).
Goal: actionable signals for fail-loops.
"""

from __future__ import annotations

from typing import Dict, List

from ...common.dto.dto import AttributionType, ModelName
from .types import ClusterDecision, Det


def build_fail_reasons(
    *,
    model_ok: Dict[ModelName, bool],
    model_errors: Dict[ModelName, str],
    dets_by_model: Dict[ModelName, List[Det]],
    decisions: List[ClusterDecision],
) -> List[AttributionType]:
    reasons: List[AttributionType] = []

    # 1) model call errors
    for m, ok in model_ok.items():
        if not ok:
            reasons.append(AttributionType(type="MODEL_ERROR", model=m, detail={"error": model_errors.get(m, "")}))

    # 2) structural anomaly
    if not dets_by_model and not reasons:
        reasons.append(AttributionType(type="STRUCTURAL_ANOMALY", model=None, detail={"error": "no dets dict"}))
        return reasons

    # If infra errors exist, don't add noisy heuristics
    if reasons:
        return reasons

    total = sum(len(v) for v in dets_by_model.values())
    if total == 0:
        reasons.append(AttributionType(type="ALL_MISS", model=None, detail={"note": "no detections from any model"}))
        return reasons

    any_weak = any(d.level == "WEAK" for d in decisions)
    any_strong = any(d.level == "STRONG" for d in decisions)

    if not any_weak and not any_strong:
        low_conf_votes = 0
        for dets in dets_by_model.values():
            for d in dets:
                if d.conf < 0.25:
                    low_conf_votes += 1
        if low_conf_votes >= max(1, total // 2):
            reasons.append(
                AttributionType(type="MODEL_LOW_CONF", model=None, detail={"total": total, "low_conf": low_conf_votes})
            )
        else:
            reasons.append(
                AttributionType(type="LOC_DISAGREE", model=None, detail={"total": total, "note": "no cluster consensus"})
            )

    if any_weak and not any_strong:
        reasons.append(AttributionType(type="MODEL_MISS", model=None, detail={"note": "only weak single-model dets"}))

    return reasons
