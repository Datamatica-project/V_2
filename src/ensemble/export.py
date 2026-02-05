# ensemble/export.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _xyxy_to_yolo(
    xyxy: Tuple[float, float, float, float],
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (cx / w, cy / h, bw / w, bh / h)


def _xyxy_to_coco_bbox(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """COCO bbox = [x, y, w, h]"""
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (float(x1), float(y1), float(w), float(h))


def _clamp_xyxy(
    xyxy: Tuple[float, float, float, float],
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = min(max(x1, 0.0), float(w))
    x2 = min(max(x2, 0.0), float(w))
    y1 = min(max(y1, 0.0), float(h))
    y2 = min(max(y2, 0.0), float(h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (float(x1), float(y1), float(x2), float(y2))


def _pick_xyxy_from_object(
    obj: Dict[str, Any],
    *,
    pick_policy: str = "fused",
) -> Optional[Tuple[float, float, float, float]]:
    """
    object에서 export에 사용할 xyxy를 고른다.
    - "fused": obj["fused"]["xyxy"] (없으면 max_conf_member로 fallback)
    - "max_conf_member": obj["members"] 중 conf 최대 박스
    """
    def _from_fused(o: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        fused = o.get("fused") or {}
        fxyxy = fused.get("xyxy")
        if isinstance(fxyxy, (list, tuple)) and len(fxyxy) == 4:
            return (float(fxyxy[0]), float(fxyxy[1]), float(fxyxy[2]), float(fxyxy[3]))
        return None

    def _from_members(o: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        members = o.get("members") or {}
        best = None
        best_conf = -1.0
        if isinstance(members, dict):
            for _, mb in members.items():
                if not isinstance(mb, dict):
                    continue
                conf = float(mb.get("conf", 0.0))
                if conf > best_conf:
                    best_conf = conf
                    best = mb
        if best is None:
            return None
        bxyxy = best.get("xyxy")
        if isinstance(bxyxy, (list, tuple)) and len(bxyxy) == 4:
            return (float(bxyxy[0]), float(bxyxy[1]), float(bxyxy[2]), float(bxyxy[3]))
        return None

    if pick_policy == "fused":
        xyxy = _from_fused(obj)
        if xyxy is not None:
            return xyxy
        # ✅ fused 누락 시 fallback
        return _from_members(obj)

    if pick_policy == "max_conf_member":
        return _from_members(obj)

    raise ValueError(f"unknown pick_policy: {pick_policy}")


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_image_path(verdict_record: Dict[str, Any]) -> Optional[str]:
    v = verdict_record.get("image_path")
    if isinstance(v, str) and v.strip():
        return v.strip()
    # 다른 키가 있다면 확장 가능
    return None


def _get_file_name(verdict_record: Dict[str, Any]) -> Optional[str]:
    v = verdict_record.get("file_name")
    if isinstance(v, str) and v.strip():
        return Path(v.strip()).name
    return None


def _safe_image_file_name(verdict_record: Dict[str, Any]) -> str:
    """
    COCO images[].file_name에 넣을 실제 파일명.
    우선순위:
      1) verdict_record.image_path basename
      2) verdict_record.file_name
      3) image_id(+.jpg)
    """
    ip = _get_image_path(verdict_record)
    if ip:
        return Path(ip).name

    fn = _get_file_name(verdict_record)
    if fn:
        return fn

    image_id = str(verdict_record.get("image_id"))
    s = str(image_id)
    if "." in Path(s).name:
        return Path(s).name
    return f"{s}.jpg"


def _label_stem(verdict_record: Dict[str, Any]) -> str:
    """
    라벨 파일 stem(확장자 제외)을 정한다.
    우선순위:
      1) image_path의 stem  (권장: 이미지 파일명과 label 파일명 매칭)
      2) file_name의 stem
      3) image_id
    ⚠️ 서로 다른 폴더에 같은 파일명이 있으면 충돌할 수 있음.
       그런 경우 image_id 기반 정책으로 바꾸거나, 상대경로 기반으로 prefix를 붙이는 방식으로 확장 가능.
    """
    ip = _get_image_path(verdict_record)
    if ip:
        return Path(ip).stem

    fn = _get_file_name(verdict_record)
    if fn:
        return Path(fn).stem

    return str(verdict_record.get("image_id"))


# -----------------------------------------------------------------------------
# YOLO TXT export (PASS objects only)
# -----------------------------------------------------------------------------
def export_yolo_txt(
    verdict_record: Dict[str, Any],
    *,
    out_dir: str | Path,
    image_wh: Tuple[int, int],
    only_if_verdict_in: Tuple[str, ...] = ("PASS_3", "PASS_2"),
    pick_policy: str = "fused",
    include_fail_objects: bool = False,
    write_empty: bool = False,
) -> Optional[Path]:
    """
    Export (YOLO txt):
    - ✅ 이미지 verdict로 export를 막지 않는다.
    - objects 중 level이 PASS_3/PASS_2(기본)만 export.
    - FAIL/MISS는 label에 넣지 않는 정책이 기본(include_fail_objects=False).

    write_empty:
      - True면 export 대상이 0개여도 빈 파일 생성
      - False면 대상 0개면 None 반환
    """
    objects = verdict_record.get("objects") or []
    if not objects:
        return None if not write_empty else (_ensure_dir(out_dir) / f"{_label_stem(verdict_record)}.txt")

    w, h = image_wh
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image_wh: {image_wh}")

    out_dir = _ensure_dir(out_dir)
    stem = _label_stem(verdict_record)
    p = out_dir / f"{stem}.txt"

    lines: List[str] = []
    allowed_levels = set(only_if_verdict_in)

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        level = str(obj.get("level", ""))

        # ✅ 기본 정책: PASS만 export
        if not include_fail_objects:
            if level not in allowed_levels:
                continue
        else:
            # debug 옵션: FAIL도 넣고 싶으면 허용 (MISS는 여전히 제외)
            if level not in allowed_levels and level != "FAIL":
                continue

        cls = int(obj.get("cls", -1))
        if cls < 0:
            continue

        xyxy = _pick_xyxy_from_object(obj, pick_policy=pick_policy)
        if xyxy is None:
            continue

        xyxy = _clamp_xyxy(xyxy, w, h)
        cx, cy, bw, bh = _xyxy_to_yolo(xyxy, w, h)

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)

        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines and not write_empty:
        return None

    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


# -----------------------------------------------------------------------------
# COCO JSON export (PASS objects only; per-image json)
# -----------------------------------------------------------------------------
def export_coco_json(
    verdict_record: Dict[str, Any],
    *,
    out_dir: str | Path,
    image_wh: Tuple[int, int],
    only_if_verdict_in: Tuple[str, ...] = ("PASS_3", "PASS_2"),
    pick_policy: str = "fused",
    include_fail_objects: bool = False,
    image_numeric_id: int = 1,
    write_empty: bool = False,
) -> Optional[Path]:
    """
    Export (COCO json):
    - ✅ 이미지 verdict로 export를 막지 않는다.
    - objects 중 PASS_3/PASS_2(기본)만 annotations로 변환
    - per-image COCO JSON (images=1개)
    - categories는 여기서 만들지 않음(외부 classes.json 등으로 merge 권장)

    write_empty:
      - True면 export 대상 0개여도 images만 포함된 json 생성
      - False면 대상 0개면 None 반환
    """
    objects = verdict_record.get("objects") or []
    if not objects:
        return None if not write_empty else (_ensure_dir(out_dir) / f"{_label_stem(verdict_record)}.json")

    w, h = image_wh
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image_wh: {image_wh}")

    out_dir = _ensure_dir(out_dir)
    stem = _label_stem(verdict_record)
    p = out_dir / f"{stem}.json"

    image_id = str(verdict_record.get("image_id"))
    file_name = _safe_image_file_name(verdict_record)

    images = [
        {
            "id": int(image_numeric_id),
            "file_name": file_name,
            "width": int(w),
            "height": int(h),
        }
    ]

    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    allowed_levels = set(only_if_verdict_in)

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        level = str(obj.get("level", ""))

        if not include_fail_objects:
            if level not in allowed_levels:
                continue
        else:
            if level not in allowed_levels and level != "FAIL":
                continue

        cls = int(obj.get("cls", -1))
        if cls < 0:
            continue

        xyxy = _pick_xyxy_from_object(obj, pick_policy=pick_policy)
        if xyxy is None:
            continue

        xyxy = _clamp_xyxy(xyxy, w, h)
        x, y, bw, bh = _xyxy_to_coco_bbox(xyxy)
        area = float(bw * bh)

        score = None
        fused = obj.get("fused") or {}
        if isinstance(fused, dict) and "conf" in fused:
            try:
                score = float(fused.get("conf"))
            except Exception:
                score = None

        ann: Dict[str, Any] = {
            "id": ann_id,
            "image_id": int(image_numeric_id),
            "category_id": int(cls),
            "bbox": [float(x), float(y), float(bw), float(bh)],
            "area": float(area),
            "iscrowd": 0,
        }
        if score is not None:
            ann["score"] = score

        annotations.append(ann)
        ann_id += 1

    if not annotations and not write_empty:
        return None

    coco = {
        "info": {
            "description": "V2 auto-label draft (PASS objects only)",
            "source": "v2-judge/ensemble",
            "image_id": image_id,
            "batch_id": verdict_record.get("batch_id"),
            "export_policy": {
                "object_levels_in": list(only_if_verdict_in),
                "pick_policy": pick_policy,
                "include_fail_objects": include_fail_objects,
                "write_empty": write_empty,
            },
        },
        "images": images,
        "annotations": annotations,
        "categories": [],
    }

    p.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# -----------------------------------------------------------------------------
# META JSON export (FAIL/MISS evidence; PASS는 index-only 가능)
# -----------------------------------------------------------------------------
def export_meta_json(
    verdict_record: Dict[str, Any],
    *,
    out_dir: str | Path,
    image_wh: Optional[Tuple[int, int]] = None,
    include_pass_objects_index_only: bool = True,
    include_pass_fused_bbox: bool = False,
) -> Path:
    out_dir = _ensure_dir(out_dir)

    image_id = str(verdict_record.get("image_id"))
    stem = _label_stem(verdict_record)
    p = out_dir / f"{stem}.meta.json"

    w = h = None
    if image_wh is not None:
        w, h = image_wh

    objects = verdict_record.get("objects") or []
    fail_singles = verdict_record.get("fail_singles") or []

    miss_evidence = verdict_record.get("miss_evidence")
    evidence = verdict_record.get("evidence") or {}
    raw_by_model = None
    hint_box = None
    miss_type = None

    if isinstance(miss_evidence, dict):
        miss_type = miss_evidence.get("miss_type")
        raw_by_model = miss_evidence.get("raw_by_model")
        hint_box = miss_evidence.get("hint_box")
    else:
        raw_by_model = evidence.get("raw_by_model")

    def _count(level: str) -> int:
        return sum(1 for o in objects if str(getattr(o, "get", lambda *_: None)("level")) == level) if isinstance(objects, list) else 0

    summary = {
        "pass_3": sum(1 for o in objects if isinstance(o, dict) and str(o.get("level")) == "PASS_3"),
        "pass_2": sum(1 for o in objects if isinstance(o, dict) and str(o.get("level")) == "PASS_2"),
        "fail_weak": sum(1 for o in objects if isinstance(o, dict) and str(o.get("level")) == "FAIL"),
        "fail_single": len(fail_singles) if isinstance(fail_singles, list) else 0,
        "verdict": verdict_record.get("verdict"),
    }

    meta_objects: List[Dict[str, Any]] = []

    # PASS objects: index-only or bbox 포함
    if include_pass_objects_index_only and isinstance(objects, list):
        for o in objects:
            if not isinstance(o, dict):
                continue
            lv = str(o.get("level", ""))
            if lv not in ("PASS_3", "PASS_2"):
                continue

            rec: Dict[str, Any] = {
                "object_index": int(o.get("object_index", -1)),
                "verdict": lv,
                "note": "exported_in_label",
            }
            if include_pass_fused_bbox:
                fused = o.get("fused") or {}
                fxyxy = fused.get("xyxy") if isinstance(fused, dict) else None
                rec["bbox"] = [float(x) for x in fxyxy] if isinstance(fxyxy, (list, tuple)) and len(fxyxy) == 4 else None
            else:
                rec["bbox"] = None

            meta_objects.append(rec)

    # FAIL_WEAK objects: objects 중 level == FAIL
    if isinstance(objects, list):
        for o in objects:
            if not isinstance(o, dict):
                continue
            if str(o.get("level", "")) != "FAIL":
                continue
            fused = o.get("fused") or {}
            fxyxy = fused.get("xyxy") if isinstance(fused, dict) else None
            rec = {
                "object_index": int(o.get("object_index", -1)),
                "verdict": "FAIL",
                "fail_subtype": o.get("fail_subtype") or "FAIL_WEAK",
                "cls": int(o.get("cls", -1)),
                "confidence": float(fused.get("conf", 0.0)) if isinstance(fused, dict) else 0.0,
                "bbox": [float(x) for x in fxyxy] if isinstance(fxyxy, (list, tuple)) and len(fxyxy) == 4 else None,
                "members": o.get("members") or {},
            }
            meta_objects.append(rec)

    # FAIL_SINGLE objects
    if isinstance(fail_singles, list):
        for s in fail_singles:
            if not isinstance(s, dict):
                continue
            rec = {
                "object_index": int(s.get("object_index", -1)),
                "verdict": "FAIL",
                "fail_subtype": "FAIL_SINGLE",
                "source_model": s.get("source_model"),
                "cls": int(s.get("cls", -1)),
                "confidence": float(s.get("conf", 0.0)),
                "bbox": s.get("xyxy"),
            }
            meta_objects.append(rec)

    meta: Dict[str, Any] = {
        "image_id": image_id,
        "batch_id": verdict_record.get("batch_id"),
        "file_name": _safe_image_file_name(verdict_record),
        "image_path": _get_image_path(verdict_record),
        "image_size": {"width": w, "height": h} if (w is not None and h is not None) else {},
        "summary": summary,
        "objects": meta_objects,
        "miss": None,
        "policy": {
            "label_export": {"include": ["PASS_3", "PASS_2"], "exclude": ["FAIL", "MISS"]},
            "object_index_scope": "per_image",
        },
    }

    if verdict_record.get("verdict") == "MISS":
        meta["miss"] = {
            "miss_type": miss_type or (verdict_record.get("summary", {}) or {}).get("miss_type"),
            "hint_box": hint_box,
            "raw_by_model": raw_by_model,
        }

    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# -----------------------------------------------------------------------------
# Convenience: export both label formats + meta in one call
# -----------------------------------------------------------------------------
def export_all_for_image(
    verdict_record: Dict[str, Any],
    *,
    yolo_dir: str | Path,
    coco_dir: str | Path,
    meta_dir: str | Path,
    image_wh: Tuple[int, int],
    only_if_verdict_in: Tuple[str, ...] = ("PASS_3", "PASS_2"),
    pick_policy: str = "fused",
    include_fail_objects_in_labels: bool = False,
    meta_include_pass_index_only: bool = True,
    meta_include_pass_fused_bbox: bool = False,
    write_empty_labels: bool = False,
) -> Dict[str, Optional[str]]:
    """
    한 이미지에 대해:
      - YOLO txt (PASS objects only)
      - COCO json (PASS objects only)
      - meta json (FAIL/MISS evidence)
    """
    y = export_yolo_txt(
        verdict_record,
        out_dir=yolo_dir,
        image_wh=image_wh,
        only_if_verdict_in=only_if_verdict_in,
        pick_policy=pick_policy,
        include_fail_objects=include_fail_objects_in_labels,
        write_empty=write_empty_labels,
    )
    c = export_coco_json(
        verdict_record,
        out_dir=coco_dir,
        image_wh=image_wh,
        only_if_verdict_in=only_if_verdict_in,
        pick_policy=pick_policy,
        include_fail_objects=include_fail_objects_in_labels,
        write_empty=write_empty_labels,
    )
    m = export_meta_json(
        verdict_record,
        out_dir=meta_dir,
        image_wh=image_wh,
        include_pass_objects_index_only=meta_include_pass_index_only,
        include_pass_fused_bbox=meta_include_pass_fused_bbox,
    )

    return {
        "yolo_txt": str(y) if y else None,
        "coco_json": str(c) if c else None,
        "meta_json": str(m) if m else None,
    }
