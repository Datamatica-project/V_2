# ensemble/match.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: List[float]) -> float:
    # population std (MVP용)
    if not xs:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return float(v ** 0.5)


@dataclass
class Box:
    model: str
    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]
    raw: Any = None  # 원본 보관(옵션)


@dataclass
class Cluster:
    cls: int
    members: Dict[str, Box] = field(default_factory=dict)  # model -> Box

    # -----------------------------
    # Basic
    # -----------------------------
    def agree_count(self) -> int:
        return len(self.members)

    def best_box(self) -> Box:
        # export 등에서 대표가 필요하면 "conf 최대"로 선택
        return max(self.members.values(), key=lambda b: b.conf)

    def conf_list(self) -> List[float]:
        return [float(b.conf) for b in self.members.values()]

    def conf_stats(self) -> Dict[str, float]:
        """
        신뢰도 분류(feature)용 통계
        - conf는 모델간 calibration이 다를 수 있으므로 "정답 확률"로 해석하지 말고,
          합의 품질을 보는 feature로만 사용.
        """
        cs = self.conf_list()
        return {
            "min": float(min(cs)) if cs else 0.0,
            "mean": _mean(cs),
            "max": float(max(cs)) if cs else 0.0,
            "std": _std(cs),
        }

    # -----------------------------
    # Fusion (Draft 대표 박스 생성)
    # -----------------------------
    def fuse_xyxy(self, *, method: str = "conf_weighted_avg", eps: float = 1e-9) -> Tuple[float, float, float, float]:
        """
        클러스터 내부 박스들을 하나의 대표 박스로 합친다 (Draft용).
        기본: conf-weighted average.
        - NOTE: WBF-lite (클러스터가 이미 "같은 객체"라고 합의된 상태에서만 호출 권장)
        """
        boxes = list(self.members.values())
        if not boxes:
            return (0.0, 0.0, 0.0, 0.0)
        if len(boxes) == 1:
            return boxes[0].xyxy

        if method not in ("conf_weighted_avg", "avg"):
            raise ValueError(f"unknown fuse method: {method}")

        if method == "avg":
            xs = [b.xyxy for b in boxes]
            return (
                _mean([x[0] for x in xs]),
                _mean([x[1] for x in xs]),
                _mean([x[2] for x in xs]),
                _mean([x[3] for x in xs]),
            )

        # conf_weighted_avg
        ws = [max(0.0, float(b.conf)) for b in boxes]
        denom = sum(ws)
        if denom <= eps:
            # conf가 전부 0이면 평균으로 fallback
            return self.fuse_xyxy(method="avg")

        x1 = sum(w * b.xyxy[0] for w, b in zip(ws, boxes)) / denom
        y1 = sum(w * b.xyxy[1] for w, b in zip(ws, boxes)) / denom
        x2 = sum(w * b.xyxy[2] for w, b in zip(ws, boxes)) / denom
        y2 = sum(w * b.xyxy[3] for w, b in zip(ws, boxes)) / denom
        return (float(x1), float(y1), float(x2), float(y2))

    def fused_conf(self, *, policy: str = "mean") -> float:
        """
        Fusion 대표 박스에 부여할 conf 값(표시/정렬용).
        - conf는 모델간 스케일 차가 있을 수 있어 "학습 신뢰도"로 쓰지 말 것.
        """
        cs = self.conf_list()
        if not cs:
            return 0.0
        if policy == "mean":
            return _mean(cs)
        if policy == "min":
            return float(min(cs))
        if policy == "max":
            return float(max(cs))
        raise ValueError(f"unknown fused_conf policy: {policy}")

    # -----------------------------
    # Spread / Stability (분류에 쓰는 지표)
    # -----------------------------
    def pairwise_ious(self) -> List[float]:
        bs = list(self.members.values())
        n = len(bs)
        if n < 2:
            return []
        out: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                out.append(iou_xyxy(bs[i].xyxy, bs[j].xyxy))
        return out

    def iou_spread(self) -> Dict[str, float]:
        """
        클러스터 내부 위치 일관성 지표:
        - min_iou가 낮으면 모델들 박스 위치가 크게 어긋남
        """
        ious = self.pairwise_ious()
        if not ious:
            return {"min": 1.0, "mean": 1.0, "std": 0.0}
        return {
            "min": float(min(ious)),
            "mean": _mean(ious),
            "std": _std(ious),
        }

    def l1_spread(self, *, fused_xyxy: Optional[Tuple[float, float, float, float]] = None) -> float:
        """
        fused 박스 대비 멤버 박스들의 평균 L1 거리(좌표 기준).
        - 값이 클수록 위치/크기 불안정
        """
        bs = list(self.members.values())
        if not bs:
            return 0.0
        f = fused_xyxy or self.fuse_xyxy()
        total = 0.0
        for b in bs:
            x1, y1, x2, y2 = b.xyxy
            total += abs(x1 - f[0]) + abs(y1 - f[1]) + abs(x2 - f[2]) + abs(y2 - f[3])
        return float(total / len(bs))


def cluster_consensus(
    boxes: List[Box],
    *,
    iou_thr: float,
    min_agree: int,
) -> List[Cluster]:
    """
    같은 클래스 + IoU>=iou_thr인 박스들을 클러스터로 묶고,
    agree_count>=min_agree(기본 2)인 클러스터만 반환.
    - 대표 박스(합치기)는 여기서 만들지 않음. (Fusion은 Cluster.fuse_xyxy에서)
    """
    # 클래스별로만 클러스터링
    boxes_by_cls: Dict[int, List[Box]] = {}
    for b in boxes:
        boxes_by_cls.setdefault(b.cls, []).append(b)

    out: List[Cluster] = []

    for cls, cls_boxes in boxes_by_cls.items():
        clusters: List[Cluster] = []

        for b in cls_boxes:
            # 같은 모델이 한 클러스터에 2개 들어가면 안 되므로,
            # "가장 잘 맞는 클러스터"를 하나만 선택
            best_idx: Optional[int] = None
            best_iou: float = 0.0

            for ci, c in enumerate(clusters):
                if b.model in c.members:
                    continue  # 동일 모델 중복 금지

                # 클러스터 내 어떤 멤버와든 IoU가 충분하면 매칭 후보
                # (여기선 max IoU 기준으로 best cluster 선택)
                mx = 0.0
                for mb in c.members.values():
                    mx = max(mx, iou_xyxy(b.xyxy, mb.xyxy))
                if mx >= iou_thr and mx > best_iou:
                    best_iou = mx
                    best_idx = ci

            if best_idx is None:
                c = Cluster(cls=cls)
                c.members[b.model] = b
                clusters.append(c)
            else:
                clusters[best_idx].members[b.model] = b

        # 합의된 클러스터만 남김
        for c in clusters:
            if c.agree_count() >= min_agree:
                out.append(c)

    return out
