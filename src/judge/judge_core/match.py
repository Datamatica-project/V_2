"""
judge_core.match

Make per-image clusters (object hypotheses) from multi-model detections.

Default strategy:
- Anchor + greedy 1:1 matching (per other model) within each class.
- Second pass: leftover detections (from non-anchor models) can become seeds
  to avoid recall loss when the anchor misses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Set

from ...common.dto.dto import ModelName
from .thresholds import JudgePolicy, get_iou_th, get_weak_conf_th  # ✅ conf 필터용 추가
from .types import Cluster, Det
from .utils_iou import iou_xyxy


def _group_by_class(dets: List[Det]) -> Dict[int, List[Det]]:
    out: DefaultDict[int, List[Det]] = defaultdict(list)
    for d in dets:
        out[d.cls].append(d)
    return dict(out)


def _filter_by_conf(dets: List[Det], policy: JudgePolicy) -> List[Det]:
    """
    클러스터링 후보에서 저신뢰 박스 제거.
    - ensemble.yaml의 conf_thr를 policy.default_weak_conf_th로 매핑해두면 여기서 적용됨
    - class별 threshold(weak_conf_th_by_class)도 있으면 자동 반영
    """
    kept: List[Det] = []
    for d in dets:
        th = get_weak_conf_th(policy, d.cls)
        if float(getattr(d, "conf", 0.0)) >= float(th):
            kept.append(d)
    return kept


def build_clusters_anchor_greedy(
    *,
    dets_by_model: Dict[ModelName, List[Det]],
    policy: JudgePolicy,
    anchor_model: ModelName,
) -> List[Cluster]:
    """Cluster detections for a single image."""

    if not dets_by_model:
        return []

    # 0) conf 필터링(노이즈 억제)
    dets_by_model = {m: _filter_by_conf(ds, policy) for m, ds in dets_by_model.items()}

    # 1) anchor 결정: (1) 전달 anchor -> (2) policy.priority_model -> (3) 첫 모델
    if anchor_model not in dets_by_model:
        pm = getattr(policy, "priority_model", None)
        if isinstance(pm, str) and pm in dets_by_model:
            anchor_model = pm  # type: ignore
        else:
            anchor_model = next(iter(dets_by_model.keys()))

    anchor = dets_by_model.get(anchor_model, [])
    others: Dict[ModelName, List[Det]] = {m: ds for m, ds in dets_by_model.items() if m != anchor_model}

    used: Dict[ModelName, Set[int]] = {m: set() for m in others.keys()}
    clusters: List[Cluster] = []

    # 2) seed from anchor
    anchor_by_cls = _group_by_class(anchor)

    for cls, a_list in anchor_by_cls.items():
        iou_th = get_iou_th(policy, cls)
        for a in a_list:
            members = [a]
            for m, ds in others.items():
                best_j = None
                best_iou = 0.0
                for j, d in enumerate(ds):
                    if j in used[m]:
                        continue
                    if d.cls != cls:
                        continue
                    v = iou_xyxy(a.xyxy, d.xyxy)
                    if v >= iou_th and v > best_iou:
                        best_iou = v
                        best_j = j
                if best_j is not None:
                    used[m].add(best_j)
                    members.append(ds[best_j])
            clusters.append(Cluster(cls=cls, members=members))

    # 3) leftover seeds (anchor miss 보정)
    for m, ds in others.items():
        leftovers = [d for j, d in enumerate(ds) if j not in used[m]]
        for d in leftovers:
            iou_th = get_iou_th(policy, d.cls)
            best_k = None
            best_iou = 0.0
            for k, cl in enumerate(clusters):
                if cl.cls != d.cls:
                    continue
                v = 0.0
                for md in cl.members:
                    v = max(v, iou_xyxy(md.xyxy, d.xyxy))
                if v >= iou_th and v > best_iou:
                    best_iou = v
                    best_k = k
            if best_k is not None:
                clusters[best_k].members.append(d)
            else:
                clusters.append(Cluster(cls=d.cls, members=[d]))

    # 4) dedupe
    for cl in clusters:
        seen = set()
        uniq = []
        for d in cl.members:
            key = (d.model, d.cls, d.xyxy)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        cl.members = uniq

    return clusters
