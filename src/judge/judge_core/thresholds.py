"""
judge_core.thresholds

Centralized thresholds/policy knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class JudgePolicy:
    # cluster IoU threshold
    default_iou_th: float = 0.55
    iou_th_by_class: Dict[int, float] = None  # type: ignore

    # consensus k thresholds
    weak_k: int = 2          # ✅ PASS_2 기준 (ensemble.min_agree)
    strong_k: int = 3        # ✅ PASS_3 기준 (ensemble.strong_agree)

    # confidence threshold for considering dets in clustering
    default_weak_conf_th: float = 0.50
    weak_conf_th_by_class: Dict[int, float] = None  # type: ignore

    # representative selection strategy
    rep_strategy: str = "mean_iou"  # mean_iou | priority_model | highest_conf
    priority_model: str = "yolov11" # ✅ V2: yolox 제거


def _safe_dict(d):
    return d if isinstance(d, dict) else {}


def get_iou_th(policy: JudgePolicy, cls: int) -> float:
    d = _safe_dict(policy.iou_th_by_class)
    return float(d.get(cls, policy.default_iou_th))


def get_weak_conf_th(policy: JudgePolicy, cls: int) -> float:
    d = _safe_dict(policy.weak_conf_th_by_class)
    return float(d.get(cls, policy.default_weak_conf_th))
