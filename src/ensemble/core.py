# ensemble/core.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from .match import Box, Cluster, cluster_consensus


def _get_attr(obj: Any, name: str, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _as_xyxy(det: Any) -> Tuple[float, float, float, float]:
    xyxy = _get_attr(det, "xyxy", None)
    if xyxy is None and isinstance(det, dict):
        xyxy = det.get("xyxy")
    if xyxy is None:
        raise ValueError("detection missing xyxy")
    if len(xyxy) != 4:
        raise ValueError(f"xyxy must be length 4 (got {xyxy})")
    return (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))


def _as_cls(det: Any) -> int:
    v = _get_attr(det, "cls", None)
    if v is None and isinstance(det, dict):
        v = det.get("cls")
    return int(v)


def _as_conf(det: Any) -> float:
    v = _get_attr(det, "conf", None)
    if v is None and isinstance(det, dict):
        v = det.get("conf")
    return float(v)


def _index_model_results(resp: Any) -> Dict[str, List[Any]]:
    """
    모델 응답에서 image_id -> detections 인덱싱
    지원 형태:
      1) resp.results: [{image_id, detections:[...]}]
      2) resp.raw.results: [{image_id, detections:[...]}]   # ✅ 현재 DTO 구조
      3) dict 형태도 동일하게 지원
    """
    out: Dict[str, List[Any]] = {}

    results = _get_attr(resp, "results", None)
    if results is None and isinstance(resp, dict):
        results = resp.get("results")

    # resp.raw.results
    if results is None:
        raw = _get_attr(resp, "raw", None)
        if raw is None and isinstance(resp, dict):
            raw = resp.get("raw")
        if isinstance(raw, dict):
            results = raw.get("results")

    if results is None:
        results = []

    for r in results or []:
        image_id = _get_attr(r, "image_id", None)
        if image_id is None and isinstance(r, dict):
            image_id = r.get("image_id")

        dets = _get_attr(r, "detections", None)
        if dets is None and isinstance(r, dict):
            dets = r.get("detections", [])

        if image_id is None:
            continue
        out[str(image_id)] = list(dets or [])

    return out


# -----------------------------
# IoU / NMS-lite for UI sanity
# -----------------------------
def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _nms_lite(boxes: List[Box], iou_thr: float = 0.7, top_k: Optional[int] = None) -> List[Box]:
    """
    표시/메타용 NMS-lite.
    - 같은 model 내에서 conf 내림차순
    - IoU가 높으면 제거
    """
    if not boxes:
        return []
    xs = sorted(boxes, key=lambda b: float(b.conf), reverse=True)
    kept: List[Box] = []
    for b in xs:
        drop = False
        for k in kept:
            if _iou_xyxy(b.xyxy, k.xyxy) >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append(b)
        if top_k is not None and len(kept) >= top_k:
            break
    return kept


def _box_key(b: Box) -> Tuple[str, int, float, float, float, float, float]:
    # used/unused 판별용(부동소수 안정 위해 round)
    x1, y1, x2, y2 = b.xyxy
    return (
        str(b.model),
        int(b.cls),
        round(float(b.conf), 6),
        round(float(x1), 3),
        round(float(y1), 3),
        round(float(x2), 3),
        round(float(y2), 3),
    )


def _union_xyxy(xyxys: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not xyxys:
        return None
    x1 = min(v[0] for v in xyxys)
    y1 = min(v[1] for v in xyxys)
    x2 = max(v[2] for v in xyxys)
    y2 = max(v[3] for v in xyxys)
    return (float(x1), float(y1), float(x2), float(y2))


def _reliability_level_for_cluster(
    c: Cluster,
    *,
    pass3_agree: int,
    min_conf_p3: float,
    min_conf_p2: float,
    min_pair_iou_p3: float,
    min_pair_iou_p2: float,
) -> Tuple[str, Dict[str, Any]]:
    """
    클러스터 단위 신뢰도 분류(MVP 룰셋)
    - PASS_3: 3모델 합의 + conf/위치 일관성 높음
    - PASS_2: 2모델 이상 합의 + conf/위치 일관성 중간 이상
    - FAIL  : (2모델 이상 합의이긴 하나) 품질 낮음  -> FAIL_WEAK 의미
    """
    agree = c.agree_count()
    conf_stats = c.conf_stats()
    iou_spread = c.iou_spread()
    fused_xyxy = c.fuse_xyxy()
    l1_spread = c.l1_spread(fused_xyxy=fused_xyxy)

    # PASS_3
    if (
        agree >= pass3_agree
        and conf_stats["min"] >= min_conf_p3
        and iou_spread["min"] >= min_pair_iou_p3
    ):
        return "PASS_3", {
            "agree_count": agree,
            "conf_stats": conf_stats,
            "iou_spread": iou_spread,
            "l1_spread": l1_spread,
        }

    # PASS_2
    if (
        agree >= 2
        and conf_stats["min"] >= min_conf_p2
        and iou_spread["min"] >= min_pair_iou_p2
    ):
        return "PASS_2", {
            "agree_count": agree,
            "conf_stats": conf_stats,
            "iou_spread": iou_spread,
            "l1_spread": l1_spread,
        }

    # FAIL_WEAK
    return "FAIL", {
        "agree_count": agree,
        "conf_stats": conf_stats,
        "iou_spread": iou_spread,
        "l1_spread": l1_spread,
    }


def ensemble_bundle(
    bundle: Any,
    *,
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
    min_agree: int = 2,
    strong_agree: int = 3,
    # --- MVP reliability thresholds (튜닝 가능) ---
    min_conf_p3: float = 0.55,
    min_conf_p2: float = 0.35,
    min_pair_iou_p3: float = 0.55,
    min_pair_iou_p2: float = 0.40,
    # --- NMS-lite for FAIL_SINGLE / evidence ---
    nms_lite_iou: float = 0.7,
    fail_single_top_k_per_model: Optional[int] = 50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    목적(개정):
      - 1) Consensus: IoU 기반 클러스터(기본 min_agree=2)
      - 2) Fusion: 클러스터 내부 conf-weighted avg로 fused box 생성 (Draft)
      - 3) Classification: PASS_3 / PASS_2 / FAIL / MISS
         FAIL은 다음을 포함:
           - FAIL_WEAK: (2+ 합의 클러스터 존재) 품질 기준 미달
           - FAIL_SINGLE: 합의 클러스터에 못 들어간 단일 박스(1모델 검출)

    Returns:
      - verdict_records: 이미지별 verdict (PASS_3/PASS_2/FAIL/MISS)
      - attribution_records: FAIL/MISS 원인 기록(간단)
    """
    if isinstance(bundle, dict):
        batch_id = bundle.get("batch_id", "")
        items = bundle.get("items", []) or []
        responses = bundle.get("responses", {}) or {}
        errors = bundle.get("errors", {}) or {}
    else:
        batch_id = _get_attr(bundle, "batch_id", "")
        items = _get_attr(bundle, "items", []) or []
        responses = _get_attr(bundle, "responses", {}) or {}
        errors = _get_attr(bundle, "errors", {}) or {}
    model_ok: Dict[str, bool] = {}
    any_err = bool(errors)

    for m, r in (responses.items() if isinstance(responses, dict) else []):
        ok = _get_attr(r, "ok", None)
        if ok is None and isinstance(r, dict):
            ok = r.get("ok", True)
        ok = bool(ok)
        model_ok[str(m)] = ok
        if not ok:
            any_err = True

    # 모델별 image_id -> dets
    model_dets: Dict[str, Dict[str, List[Any]]] = {}
    for m, r in (responses.items() if isinstance(responses, dict) else []):
        model_dets[str(m)] = _index_model_results(r)

    verdicts: List[Dict[str, Any]] = []
    attributions: List[Dict[str, Any]] = []

    # 모델 에러: 보수적으로 전부 FAIL
    if any_err:
        for it in items:
            image_id = _get_attr(it, "image_id", None) or (it.get("image_id") if isinstance(it, dict) else None)
            image_id = str(image_id)

            verdicts.append({
                "image_id": image_id,
                "batch_id": batch_id,
                "verdict": "FAIL",
                "model_ok": model_ok,
                "summary": {"note": "ensemble: model error => FAIL"},
                "objects": [],
                "fail_singles": [],
            })

            reasons = []
            for m, e in (errors.items() if isinstance(errors, dict) else []):
                reasons.append({"type": "MODEL_ERROR", "model": str(m), "detail": {"error": str(e)}})
            if not reasons:
                reasons.append({"type": "MODEL_ERROR", "model": None, "detail": {"error": "unknown"}})

            attributions.append({
                "image_id": image_id,
                "batch_id": batch_id,
                "reasons": reasons,
            })
        return verdicts, attributions

    # 정상 케이스: image별 consensus + fusion + 분류
    model_names = list(model_dets.keys())

    for it in items:
        image_id = _get_attr(it, "image_id", None) or (it.get("image_id") if isinstance(it, dict) else None)
        image_id = str(image_id)

        # evidence: 모델별 raw(필터 전)
        raw_by_model: Dict[str, List[Dict[str, Any]]] = {}
        # boxes_for_consensus: conf_thr 적용 후 Box
        boxes: List[Box] = []

        total_raw = 0
        kept_after_conf = 0

        for m in model_names:
            dets = model_dets[m].get(image_id, [])
            raw_list: List[Dict[str, Any]] = []
            for d in dets:
                total_raw += 1
                try:
                    cls_ = _as_cls(d)
                    conf_ = _as_conf(d)
                    xyxy_ = _as_xyxy(d)
                except Exception:
                    continue

                raw_list.append({
                    "cls": int(cls_),
                    "conf": float(conf_),
                    "xyxy": [float(xyxy_[0]), float(xyxy_[1]), float(xyxy_[2]), float(xyxy_[3])],
                })

                if float(conf_) < conf_thr:
                    continue

                kept_after_conf += 1
                boxes.append(Box(
                    model=m,
                    cls=int(cls_),
                    conf=float(conf_),
                    xyxy=xyxy_,
                    raw=d,
                ))

            raw_by_model[m] = raw_list

        # consensus(2+)
        consensus: List[Cluster] = cluster_consensus(
            boxes,
            iou_thr=iou_thr,
            min_agree=min_agree,
        )

        # ----------------------------------------------------------
        # 1) consensus가 0개인 경우: MISS vs FAIL(싱글) 분기
        # ----------------------------------------------------------
        if not consensus:
            all_miss = (total_raw == 0) or (kept_after_conf == 0)

            # conf_thr 이후 기준으로 모델별 "유효 박스가 있는지"
            has_kept_by_model = {m: False for m in model_names}
            for b in boxes:
                has_kept_by_model[str(b.model)] = True

            kept_models = [m for m, v in has_kept_by_model.items() if v]
            three_models_have = (len(model_names) >= 3 and all(has_kept_by_model.get(m, False) for m in model_names[:3])) \
                                if len(model_names) == 3 else (len(kept_models) == len(model_names) and len(model_names) >= 3)

            if all_miss:
                verdicts.append({
                    "image_id": image_id,
                    "batch_id": batch_id,
                    "verdict": "MISS",
                    "model_ok": model_ok,
                    "summary": {
                        "note": "ensemble: ALL_MISS (no valid boxes)",
                        "miss_type": "ALL_MISS",
                        "total_raw_boxes": total_raw,
                        "kept_after_conf": kept_after_conf,
                        "conf_thr": conf_thr,
                        "iou_thr": iou_thr,
                        "min_agree": min_agree,
                    },
                    "objects": [],
                    "fail_singles": [],
                    "miss_evidence": {
                        "raw_by_model": raw_by_model,
                        "hint_box": None,
                    },
                })
                attributions.append({
                    "image_id": image_id,
                    "batch_id": batch_id,
                    "reasons": [{"type": "ALL_MISS", "model": None, "detail": {}}],
                })
                continue

            # 박스는 있는데 2+ 합의가 0개
            if three_models_have:
                # DISAGREE_3MODELS: 힌트 union box 생성(inspection only)
                # union은 "각 모델의 최고 conf 박스 1개" 기준으로 만드는 것을 권장(너무 커지는 것을 완화)
                top_xyxys: List[Tuple[float, float, float, float]] = []
                for m in model_names:
                    # boxes에는 conf_thr 이상만 있음. 여기서 모델별 top1
                    bs = [b for b in boxes if str(b.model) == m]
                    if not bs:
                        continue
                    b0 = max(bs, key=lambda x: float(x.conf))
                    top_xyxys.append(b0.xyxy)

                uni = _union_xyxy(top_xyxys)
                hint_box = None
                if uni is not None:
                    hint_box = {
                        "type": "union_box",
                        "bbox": [float(uni[0]), float(uni[1]), float(uni[2]), float(uni[3])],
                        "inspection_only": True,
                        "source": "top1_per_model_union",
                    }

                verdicts.append({
                    "image_id": image_id,
                    "batch_id": batch_id,
                    "verdict": "MISS",
                    "model_ok": model_ok,
                    "summary": {
                        "note": "ensemble: DISAGREE_3MODELS (no 2+ consensus)",
                        "miss_type": "DISAGREE_3MODELS",
                        "total_raw_boxes": total_raw,
                        "kept_after_conf": kept_after_conf,
                        "conf_thr": conf_thr,
                        "iou_thr": iou_thr,
                        "min_agree": min_agree,
                    },
                    "objects": [],
                    "fail_singles": [],
                    "miss_evidence": {
                        "raw_by_model": raw_by_model,
                        "hint_box": hint_box,
                    },
                })
                attributions.append({
                    "image_id": image_id,
                    "batch_id": batch_id,
                    "reasons": [{"type": "LOC_DISAGREE", "model": None, "detail": {"hint": "3 models detected but cannot form IoU consensus"}}],
                })
                continue

            # 1~2모델만 박스가 존재하는데 합의가 없음 => FAIL(싱글 박스 검수 필요)
            # 여기서는 boxes(=conf_thr 통과) 자체가 싱글 근거가 됨. 모델별 NMS-lite로 정리
            fail_singles_by_model: Dict[str, List[Box]] = {}
            for m in model_names:
                bs = [b for b in boxes if str(b.model) == m]
                bs = _nms_lite(bs, iou_thr=nms_lite_iou, top_k=fail_single_top_k_per_model)
                if bs:
                    fail_singles_by_model[m] = bs

            fail_singles: List[Dict[str, Any]] = []
            oi = 0
            # object_index는 모델/좌표 기준으로 결정적으로 정렬
            flat = []
            for m, bs in fail_singles_by_model.items():
                for b in bs:
                    flat.append(b)
            flat.sort(key=lambda b: (float(b.xyxy[0]), float(b.xyxy[1]), -float(b.conf)))
            for b in flat:
                x1, y1, x2, y2 = b.xyxy
                fail_singles.append({
                    "object_index": oi,
                    "level": "FAIL",
                    "fail_subtype": "FAIL_SINGLE",
                    "source_model": str(b.model),
                    "cls": int(b.cls),
                    "conf": float(b.conf),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                })
                oi += 1

            verdicts.append({
                "image_id": image_id,
                "batch_id": batch_id,
                "verdict": "FAIL",
                "model_ok": model_ok,
                "summary": {
                    "note": "ensemble: no 2+ consensus, but single-model boxes exist => FAIL",
                    "total_raw_boxes": total_raw,
                    "kept_after_conf": kept_after_conf,
                    "conf_thr": conf_thr,
                    "iou_thr": iou_thr,
                    "min_agree": min_agree,
                    "n_fail_singles": len(fail_singles),
                },
                "objects": [],  # 합의 클러스터 없음
                "fail_singles": fail_singles,
                "evidence": {
                    "raw_by_model": raw_by_model,
                },
            })

            attributions.append({
                "image_id": image_id,
                "batch_id": batch_id,
                "reasons": [{"type": "MODEL_MISS", "model": None, "detail": {"hint": "only 1-2 models produced boxes, no 2+ consensus"}}],
            })
            continue

        # ----------------------------------------------------------
        # 2) consensus가 있는 경우: 클러스터 객체 생성(PASS/FAIL_WEAK)
        #    + 남은 싱글 박스 => FAIL_SINGLE
        # ----------------------------------------------------------
        objects: List[Dict[str, Any]] = []
        image_levels: List[str] = []

        # consensus에 사용된 박스 키 집합
        used_keys = set()
        for c in consensus:
            for bb in c.members.values():
                used_keys.add(_box_key(bb))

        # (A) 클러스터 객체
        for ci, c in enumerate(consensus):
            level, feats = _reliability_level_for_cluster(
                c,
                pass3_agree=strong_agree,
                min_conf_p3=min_conf_p3,
                min_conf_p2=min_conf_p2,
                min_pair_iou_p3=min_pair_iou_p3,
                min_pair_iou_p2=min_pair_iou_p2,
            )

            fused_xyxy = c.fuse_xyxy()
            fused_conf = c.fused_conf(policy="mean")

            # FAIL_WEAK 구분 태그
            fail_subtype = None
            if level == "FAIL":
                fail_subtype = "FAIL_WEAK"

            obj = {
                "object_index": ci,  # 임시. 아래에서 정렬 후 재부여
                "cls": int(c.cls),
                "agree_count": int(c.agree_count()),
                "level": level,  # PASS_3 / PASS_2 / FAIL(=WEAK)
                "fail_subtype": fail_subtype,
                "fused": {
                    "xyxy": [float(fused_xyxy[0]), float(fused_xyxy[1]), float(fused_xyxy[2]), float(fused_xyxy[3])],
                    "conf": float(fused_conf),
                    "method": "conf_weighted_avg",
                },
                "features": feats,
                "members": {
                    mm: {
                        "cls": int(bb.cls),
                        "conf": float(bb.conf),
                        "xyxy": [float(bb.xyxy[0]), float(bb.xyxy[1]), float(bb.xyxy[2]), float(bb.xyxy[3])],
                    }
                    for mm, bb in c.members.items()
                },
            }
            objects.append(obj)
            image_levels.append(level)

        # (B) 남은 박스(싱글) => FAIL_SINGLE
        # boxes는 conf_thr 통과한 전체 Box list
        singles_by_model: Dict[str, List[Box]] = {m: [] for m in model_names}
        for b in boxes:
            if _box_key(b) in used_keys:
                continue
            singles_by_model[str(b.model)].append(b)

        # 모델별 NMS-lite 적용 (표시/메타 과부하 방지)
        for m in list(singles_by_model.keys()):
            singles_by_model[m] = _nms_lite(
                singles_by_model[m],
                iou_thr=nms_lite_iou,
                top_k=fail_single_top_k_per_model,
            )

        fail_singles: List[Dict[str, Any]] = []
        flat_singles: List[Box] = []
        for m, bs in singles_by_model.items():
            for b in bs:
                flat_singles.append(b)

        # deterministic sort
        flat_singles.sort(key=lambda b: (float(b.xyxy[0]), float(b.xyxy[1]), -float(b.conf)))

        # (C) object_index 재정렬/재부여
        # 먼저 consensus objects를 fused.x1,y1 기준 정렬
        objects.sort(key=lambda o: (float(o["fused"]["xyxy"][0]), float(o["fused"]["xyxy"][1])))

        next_oi = 0
        for o in objects:
            o["object_index"] = next_oi
            next_oi += 1

        # singles는 이어서 번호 부여
        for b in flat_singles:
            x1, y1, x2, y2 = b.xyxy
            fail_singles.append({
                "object_index": next_oi,
                "level": "FAIL",
                "fail_subtype": "FAIL_SINGLE",
                "source_model": str(b.model),
                "cls": int(b.cls),
                "conf": float(b.conf),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })
            next_oi += 1

        # ----------------------------------------------------------
        # 3) 이미지 verdict 요약
        # 규칙(업데이트):
        #   - FAIL_WEAK(클러스터 FAIL) 또는 FAIL_SINGLE이 하나라도 있으면 이미지 FAIL
        #   - 아니면 PASS_2 있으면 PASS_2
        #   - 아니면 PASS_3
        # ----------------------------------------------------------
        has_fail_weak = any(lv == "FAIL" for lv in image_levels)
        has_fail_single = len(fail_singles) > 0

        if has_fail_weak or has_fail_single:
            verdict = "FAIL"
        elif any(lv == "PASS_2" for lv in image_levels):
            verdict = "PASS_2"
        else:
            verdict = "PASS_3"

        # attribution: FAIL 사유(WEAK + SINGLE)
        if verdict == "FAIL":
            reasons: List[Dict[str, Any]] = []

            # FAIL_WEAK: 클러스터 품질 미달
            for obj in objects:
                if obj.get("level") != "FAIL":
                    continue
                feats = obj.get("features", {}) or {}
                cs = (feats.get("conf_stats") or {})
                ios = (feats.get("iou_spread") or {})
                hint = {
                    "object_index": obj.get("object_index"),
                    "cls": obj.get("cls"),
                    "agree_count": obj.get("agree_count"),
                    "min_conf": cs.get("min"),
                    "min_pair_iou": ios.get("min"),
                    "l1_spread": feats.get("l1_spread"),
                }
                if cs.get("min", 0.0) < min_conf_p2:
                    reasons.append({"type": "MODEL_LOW_CONF", "model": None, "detail": hint})
                elif ios.get("min", 0.0) < min_pair_iou_p2:
                    reasons.append({"type": "LOC_DISAGREE", "model": None, "detail": hint})
                else:
                    reasons.append({"type": "STRUCTURAL_ANOMALY", "model": None, "detail": hint})

            # FAIL_SINGLE: 단일 모델 박스
            if has_fail_single:
                reasons.append({
                    "type": "SINGLE_MODEL_DETECTION",
                    "model": None,
                    "detail": {
                        "n_fail_singles": len(fail_singles),
                        "hint": "boxes not included in any 2+ consensus cluster",
                    },
                })

            attributions.append({
                "image_id": image_id,
                "batch_id": batch_id,
                "reasons": reasons or [{"type": "STRUCTURAL_ANOMALY", "model": None, "detail": {}}],
            })

        verdicts.append({
            "image_id": image_id,
            "batch_id": batch_id,
            "verdict": verdict,  # PASS_3 / PASS_2 / FAIL / MISS
            "model_ok": model_ok,
            "summary": {
                "note": "ensemble: consensus + fusion + reliability (+ singles)",
                "n_objects": len(objects),
                "n_fail_singles": len(fail_singles),
                "object_levels": {
                    "PASS_3": sum(1 for x in objects if x.get("level") == "PASS_3"),
                    "PASS_2": sum(1 for x in objects if x.get("level") == "PASS_2"),
                    "FAIL_WEAK": sum(1 for x in objects if x.get("level") == "FAIL"),
                    "FAIL_SINGLE": len(fail_singles),
                },
                "iou_thr": iou_thr,
                "conf_thr": conf_thr,
                "min_agree": min_agree,
                "strong_agree": strong_agree,
                "thr": {
                    "min_conf_p3": min_conf_p3,
                    "min_conf_p2": min_conf_p2,
                    "min_pair_iou_p3": min_pair_iou_p3,
                    "min_pair_iou_p2": min_pair_iou_p2,
                },
            },
            # 합의 기반 객체(PASS + FAIL_WEAK)
            "objects": objects,
            # 합의에 못 들어간 단일 박스(FAIL_SINGLE)
            "fail_singles": fail_singles,
            # (옵션) 디버깅/프론트 근거용
            "evidence": {
                "raw_by_model": raw_by_model,
            },
        })

    return verdicts, attributions
