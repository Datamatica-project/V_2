"""
judge_core.consensus

Convert clusters into decisions:
- PASS_3 / PASS_2 / FAIL
- representative box selection

Representative box is picked from *existing* model outputs
(to avoid synthetic box distribution shift).
"""

from __future__ import annotations

from typing import Dict, List

from .thresholds import JudgePolicy
from .types import Cluster, ClusterDecision, Det
from .utils_iou import iou_xyxy


def _pick_rep_mean_iou(members: List[Det]) -> Det:
    if len(members) == 1:
        return members[0]
    best = members[0]
    best_score = -1.0
    for i, di in enumerate(members):
        s = 0.0
        c = 0
        for j, dj in enumerate(members):
            if i == j:
                continue
            s += iou_xyxy(di.xyxy, dj.xyxy)
            c += 1
        mean_iou = s / max(1, c)
        if mean_iou > best_score:
            best_score = mean_iou
            best = di
    return best


def _pick_rep_priority(members: List[Det], priority_model: str) -> Det:
    for d in members:
        if d.model == priority_model:
            return d
    return _pick_rep_mean_iou(members)


def _pick_rep_highest_conf(members: List[Det]) -> Det:
    return sorted(members, key=lambda d: d.conf, reverse=True)[0]


def decide_clusters(
    clusters: List[Cluster],
    *,
    policy: JudgePolicy,
) -> List[ClusterDecision]:
    """
    Decide cluster levels using policy knobs.

    - PASS_3 if k >= policy.strong_k
    - PASS_2 if k >= policy.weak_k (fallback 2 if missing)
    - FAIL   otherwise

    NOTE:
    현재 JudgePolicy에 weak_k가 없을 수도 있어,
    getattr(policy, "weak_k", 2) 로 안전 처리함.
    """
    decisions: List[ClusterDecision] = []

    # ✅ configurable thresholds
    strong_k = int(getattr(policy, "strong_k", 3))
    weak_k = int(getattr(policy, "weak_k", 2))

    # guardrail: weak_k는 strong_k보다 클 수 없음
    if weak_k > strong_k:
        weak_k = strong_k

    for cl in clusters:
        models = cl.models_in()
        k = len(models)

        # representative box
        if policy.rep_strategy == "priority_model":
            rep = _pick_rep_priority(cl.members, policy.priority_model)
        elif policy.rep_strategy == "highest_conf":
            rep = _pick_rep_highest_conf(cl.members)
        else:
            rep = _pick_rep_mean_iou(cl.members)

        # 🔑 최종 level 결정 (policy 기반)
        if k >= strong_k:
            level = "PASS_3"
        elif k >= weak_k:
            level = "PASS_2"
        else:
            level = "FAIL"

        decisions.append(
            ClusterDecision(
                level=level,
                rep=rep,
                meta={
                    "cls": cl.cls,
                    "k": k,
                    "models": sorted(list(models)),
                    "weak_k": weak_k,
                    "strong_k": strong_k,
                },
            )
        )

    return decisions


def summarize_decisions(decisions: List[ClusterDecision]) -> Dict[str, object]:
    return {
        "n_clusters": len(decisions),
        "n_pass3": sum(1 for d in decisions if d.level == "PASS_3"),
        "n_pass2": sum(1 for d in decisions if d.level == "PASS_2"),
        "n_fail":  sum(1 for d in decisions if d.level == "FAIL"),
    }
