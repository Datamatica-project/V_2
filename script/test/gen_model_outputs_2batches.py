#!/usr/bin/env python3
"""
Generate realistic per-model raw outputs (as if returned by each model /infer),
split into 2 batches (b00/b01), each has N items (default 64).

Outputs:
  - infer_req_b00.json / infer_req_b01.json
  - yolov11_raw_b00.json / rtm_raw_b00.json / rtdetr_raw_b00.json
  - yolov11_raw_b01.json / rtm_raw_b01.json / rtdetr_raw_b01.json

These raw outputs intentionally make 3 models "see the same object differently":
- jittered boxes per model
- occasional model miss
- occasional 3-model disagree (far boxes)
- occasional low-conf (still >= conf_thr 0.25) to trigger FAIL_WEAK
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rand_box(rng: random.Random, w: int, h: int) -> List[float]:
    bw = rng.uniform(40, 320)
    bh = rng.uniform(40, 320)
    x1 = rng.uniform(0, w - bw - 1)
    y1 = rng.uniform(0, h - bh - 1)
    x2 = x1 + bw
    y2 = y1 + bh
    return [float(x1), float(y1), float(x2), float(y2)]


def jitter_box(xyxy: List[float], rng: random.Random, w: int, h: int, amp: float) -> List[float]:
    x1, y1, x2, y2 = xyxy
    out = [
        clamp(x1 + rng.uniform(-amp, amp), 0, w - 2),
        clamp(y1 + rng.uniform(-amp, amp), 0, h - 2),
        clamp(x2 + rng.uniform(-amp, amp), 1, w - 1),
        clamp(y2 + rng.uniform(-amp, amp), 1, h - 1),
    ]
    if out[2] <= out[0] + 1:
        out[2] = min(w - 1, out[0] + 2)
    if out[3] <= out[1] + 1:
        out[3] = min(h - 1, out[1] + 2)
    return [float(out[0]), float(out[1]), float(out[2]), float(out[3])]


def make_infer_req(batch_id: str, prefix: str, n: int, w: int, h: int, run_id: str, user_key: str, project_id: str) -> Dict[str, Any]:
    items = []
    for i in range(n):
        items.append(
            {
                "image_id": f"{prefix}_{i:04d}",
                "image_path": f"/data/unlabeled/{prefix}_{i:04d}.jpg",
                "width": int(w),
                "height": int(h),
            }
        )
    return {
        "batch_id": batch_id,
        "items": items,
        "params": {"run_id": run_id, "user_key": user_key, "project_id": project_id},
    }


def per_image_object_count(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.10:
        return 0
    if r < 0.35:
        return 1
    if r < 0.65:
        return 2
    if r < 0.85:
        return 3
    if r < 0.95:
        return 4
    return 5


def build_raw_for_batch(
    *,
    model: str,
    batch_id: str,
    infer_req: Dict[str, Any],
    seed: int,
    role: str,
) -> Dict[str, Any]:
    """
    role in {"y","r","d"} used to apply model-specific jitter/conf patterns.
    Output mimics model server /infer raw response:
      { model, batch_id, ok, elapsed_ms, weight_used, results:[{image_id, detections:[{cls,conf,xyxy}]}] }
    """
    rng = random.Random(seed)
    w = int(infer_req["items"][0]["width"])
    h = int(infer_req["items"][0]["height"])

    results: List[Dict[str, Any]] = []

    for idx, it in enumerate(infer_req["items"]):
        image_id = str(it["image_id"])
        n_obj = per_image_object_count(rng)

        dets_y: List[Dict[str, Any]] = []
        dets_r: List[Dict[str, Any]] = []
        dets_d: List[Dict[str, Any]] = []

        for _ in range(n_obj):
            cls_id = rng.choice([0, 1, 2, 3, 4])
            base = rand_box(rng, w, h)
            t = rng.random()

            # 0.00~0.60: PASS_3 (3모델 모두, 위치 유사/신뢰도 높음)
            if t < 0.60:
                dets_y.append({"cls": cls_id, "conf": round(rng.uniform(0.70, 0.95), 3), "xyxy": jitter_box(base, rng, w, h, 4.0)})
                dets_r.append({"cls": cls_id, "conf": round(rng.uniform(0.65, 0.93), 3), "xyxy": jitter_box(base, rng, w, h, 5.0)})
                dets_d.append({"cls": cls_id, "conf": round(rng.uniform(0.60, 0.90), 3), "xyxy": jitter_box(base, rng, w, h, 6.0)})

            # 0.60~0.78: PASS_2 (2모델만 검출)
            elif t < 0.78:
                miss = rng.choice(["y", "r", "d"])
                if miss != "y":
                    dets_y.append({"cls": cls_id, "conf": round(rng.uniform(0.55, 0.90), 3), "xyxy": jitter_box(base, rng, w, h, 5.0)})
                if miss != "r":
                    dets_r.append({"cls": cls_id, "conf": round(rng.uniform(0.50, 0.88), 3), "xyxy": jitter_box(base, rng, w, h, 6.0)})
                if miss != "d":
                    dets_d.append({"cls": cls_id, "conf": round(rng.uniform(0.50, 0.85), 3), "xyxy": jitter_box(base, rng, w, h, 7.0)})

            # 0.78~0.88: FAIL_SINGLE (1모델만 검출)
            elif t < 0.88:
                only = rng.choice(["y", "r", "d"])
                det = {"cls": cls_id, "conf": round(rng.uniform(0.40, 0.85), 3), "xyxy": jitter_box(base, rng, w, h, 6.0)}
                if only == "y":
                    dets_y.append(det)
                elif only == "r":
                    dets_r.append(det)
                else:
                    dets_d.append(det)

            # 0.88~0.94: DISAGREE_3MODELS (3모델 모두 있으나 위치가 많이 다름 -> MISS 유도)
            elif t < 0.94:
                far1 = rand_box(rng, w, h)
                far2 = rand_box(rng, w, h)
                dets_y.append({"cls": cls_id, "conf": round(rng.uniform(0.55, 0.90), 3), "xyxy": jitter_box(base, rng, w, h, 4.0)})
                dets_r.append({"cls": cls_id, "conf": round(rng.uniform(0.55, 0.90), 3), "xyxy": jitter_box(far1, rng, w, h, 4.0)})
                dets_d.append({"cls": cls_id, "conf": round(rng.uniform(0.55, 0.90), 3), "xyxy": jitter_box(far2, rng, w, h, 4.0)})

            # 0.94~1.00: LOW_CONF 2모델 합의(conf_thr=0.25는 넘지만 min_conf_p2=0.35 못 넘게) -> FAIL_WEAK 유도
            else:
                dets_y.append({"cls": cls_id, "conf": round(rng.uniform(0.26, 0.34), 3), "xyxy": jitter_box(base, rng, w, h, 4.0)})
                dets_r.append({"cls": cls_id, "conf": round(rng.uniform(0.26, 0.34), 3), "xyxy": jitter_box(base, rng, w, h, 5.0)})
                # d는 미검출

        # 모델별로 "자기가 본 것"만 고르기
        if role == "y":
            dets = dets_y
        elif role == "r":
            dets = dets_r
        else:
            dets = dets_d

        results.append({"image_id": image_id, "detections": dets})

    return {
        "model": model,
        "batch_id": batch_id,
        "ok": True,
        "error": None,
        "elapsed_ms": 25,
        "weight_used": f"/weights/{model}/u1_p1_{model}_v001.pt",
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="./_test_runs/model_raw_64x2")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--run-id", type=str, default=f"run_{int(time.time())}")
    ap.add_argument("--user-key", type=str, default="u1")
    ap.add_argument("--project-id", type=str, default="p1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # batch 0/1
    b00_id = f"batch_{args.run_id}_b00"
    b01_id = f"batch_{args.run_id}_b01"

    infer_b00 = make_infer_req(b00_id, "img_b00", int(args.n), int(args.w), int(args.h), args.run_id, args.user_key, args.project_id)
    infer_b01 = make_infer_req(b01_id, "img_b01", int(args.n), int(args.w), int(args.h), args.run_id, args.user_key, args.project_id)

    _write(out_dir / "infer_req_b00.json", infer_b00)
    _write(out_dir / "infer_req_b01.json", infer_b01)

    # seed는 모델별/배치별로 달리
    # (중요) 같은 이미지에서도 모델별 raw가 다르게 나오게 role/y-r-d로 분기
    for tag, infer_req, base_seed in [("b00", infer_b00, 1337), ("b01", infer_b01, 2026)]:
        y = build_raw_for_batch(model="yolov11", batch_id=infer_req["batch_id"], infer_req=infer_req, seed=base_seed + 11, role="y")
        r = build_raw_for_batch(model="rtm",    batch_id=infer_req["batch_id"], infer_req=infer_req, seed=base_seed + 22, role="r")
        d = build_raw_for_batch(model="rtdetr", batch_id=infer_req["batch_id"], infer_req=infer_req, seed=base_seed + 33, role="d")

        _write(out_dir / f"yolov11_raw_{tag}.json", y)
        _write(out_dir / f"rtm_raw_{tag}.json", r)
        _write(out_dir / f"rtdetr_raw_{tag}.json", d)

    print(f"[ok] wrote files under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
