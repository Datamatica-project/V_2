"""
judge_core.types
Internal types for judge_core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Literal

from ...common.dto.dto import ModelName

XYXY = Tuple[float, float, float, float]

# 클러스터(객체) 단위 등급
# - MISS는 "클러스터가 없는 이미지" 상태라 여기에 포함하지 않음
ClusterLevel = Literal["PASS_3", "PASS_2", "FAIL"]


@dataclass(frozen=True)
class Det:
    model: ModelName
    cls: int
    conf: float
    xyxy: XYXY

    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Cluster:
    cls: int
    members: List[Det]

    def models_in(self) -> Set[ModelName]:
        return {d.model for d in self.members}

    def k(self) -> int:
        return len(self.models_in())


@dataclass
class ClusterDecision:
    # PASS_3: k==3, PASS_2: k==2, FAIL: otherwise (e.g. k==1, conflict, unstable)
    level: ClusterLevel
    rep: Optional[Det]              # representative box (chosen from members)
    meta: Dict[str, object]         # debug/trace (k, models, thresholds, etc.)
