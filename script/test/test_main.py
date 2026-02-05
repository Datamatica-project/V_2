#!/usr/bin/env python3
"""
V2 end-to-end-ish smoke test (no GPU, no HTTP).

What it does
- Build a synthetic GatewayBundle (3 models) for multiple scenarios
- Run Judge -> (delegates) Ensemble
- Export YOLO TXT / COCO JSON (PASS only) + META JSON (FAIL/MISS evidence)
- Print structured logs per step and optionally write JSONL

Run
  python script/test/test_main.py --out-dir /tmp/v2_test_run

Notes
- Requires your repo root on PYTHONPATH so "src.*" imports work.
  Example:
    PYTHONPATH=. python script/test/test_main.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- V2 imports (SSOT) ---
from src.common.dto.dto import (
    Detection,
    GatewayBundle,
    InferItem,
    InferItemResult,
    ModelPredResponse,
)

from src.judge.judge_core.judge_entry import judge_bundle
from src.ensemble.export import export_coco_json, export_meta_json, export_yolo_txt


# -------------------------
# Logging helpers
# -------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log_line(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def log_jsonl(fp: Optional[Path], event: Dict[str, Any]) -> None:
    if fp is None:
        return
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# -------------------------
# Scenario builders
# -------------------------
def det(cls: int, conf: float, xyxy: Tuple[float, float, float, float]) -> Detection:
    return Detection(cls=int(cls), conf=float(conf), xyxy=[float(x) for x in xyxy])


def item_result(image_id: str, dets: List[Detection]) -> InferItemResult:
    return InferItemResult(image_id=image_id, detections=dets)


def model_resp(model: str, batch_id: str, results: List[InferItemResult], *, weight_used: str) -> ModelPredResponse:
    # raw.meta는 "weight_used/weight_dir_used" 흔들릴 때를 대비해서 같이 넣어둠
    return ModelPredResponse(
        model=model,  # type: ignore[arg-type]
        batch_id=batch_id,
        ok=True,
        error=None,
        elapsed_ms=5,
        weight_used=weight_used,
        results=results,
        raw={"meta": {"weight_used": weight_used, "weight_dir_used": str(Path(weight_used).parent)}},
    )


def build_bundle(scenario: str, *, batch_id: str, image_id: str, image_path: str, wh: Tuple[int, int]) -> GatewayBundle:
    w, h = wh
    it = InferItem(image_id=image_id, image_path=image_path, width=w, height=h)

    # Base weights (fake paths)
    w_yolo = f"/weights/projects/u1/p1/yolov11/v001/best.pt"
    w_rtm = f"/weights/projects/u1/p1/rtm/v001/best.pth"
    w_det = f"/weights/projects/u1/p1/rtdetr/v001/best.pt"

    # Boxes
    good = (100, 100, 300, 300)          # strong overlap candidate
    good2 = (110, 110, 310, 310)         # near overlap
    far1 = (50, 50, 200, 200)            # disagree
    far2 = (900, 50, 1100, 250)          # disagree
    far3 = (400, 500, 600, 700)          # disagree

    # Default empty
    yolo_dets: List[Detection] = []
    rtm_dets: List[Detection] = []
    det_dets: List[Detection] = []

    if scenario == "pass3":
        yolo_dets = [det(0, 0.92, good)]
        rtm_dets  = [det(0, 0.88, good2)]
        det_dets  = [det(0, 0.90, good)]
    elif scenario == "pass2":
        yolo_dets = [det(0, 0.80, good)]
        rtm_dets  = [det(0, 0.78, good2)]
        det_dets  = []  # third misses
    elif scenario == "fail_single":
        yolo_dets = [det(0, 0.80, good)]
        rtm_dets  = []
        det_dets  = []
    elif scenario == "miss_all":
        yolo_dets = []
        rtm_dets  = []
        det_dets  = []
    elif scenario == "miss_disagree3":
        # 3 models have confident boxes but far apart => DISAGREE_3MODELS -> MISS + hint_box
        yolo_dets = [det(0, 0.85, far1)]
        rtm_dets  = [det(0, 0.86, far2)]
        det_dets  = [det(0, 0.87, far3)]
    elif scenario == "fail_weak_lowconf":
        # 2+ consensus exists (conf_thr=0.25) but below min_conf_p2(0.35) => FAIL_WEAK
        yolo_dets = [det(0, 0.28, good)]
        rtm_dets  = [det(0, 0.30, good2)]
        det_dets  = []
    elif scenario == "fail_pass2_plus_single":
        # consensus PASS_2 + extra single from third => image verdict FAIL (due to FAIL_SINGLE)
        yolo_dets = [det(0, 0.82, good)]
        rtm_dets  = [det(0, 0.79, good2)]
        det_dets  = [det(0, 0.70, far3)]
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    yolo = model_resp("yolov11", batch_id, [item_result(image_id, yolo_dets)], weight_used=w_yolo)
    rtm  = model_resp("rtm",     batch_id, [item_result(image_id, rtm_dets)],  weight_used=w_rtm)
    rtd  = model_resp("rtdetr",  batch_id, [item_result(image_id, det_dets)],  weight_used=w_det)

    return GatewayBundle(
        batch_id=batch_id,
        items=[it],
        responses={"yolov11": yolo, "rtm": rtm, "rtdetr": rtd},  # type: ignore[arg-type]
        errors={},
    )


# -------------------------
# Runner
# -------------------------
def run_one(
    scenario: str,
    *,
    out_dir: Path,
    wh: Tuple[int, int],
    log_fp: Optional[Path],
) -> None:
    batch_id = f"batch_{scenario}"
    image_id = f"{scenario}_img1"
    image_path = f"/dummy/{scenario}.jpg"

    log_line(f"=== SCENARIO: {scenario} ===")
    log_jsonl(log_fp, {"ts": _now(), "stage": "scenario_start", "scenario": scenario})

    # 1) Build bundle
    bundle = build_bundle(scenario, batch_id=batch_id, image_id=image_id, image_path=image_path, wh=wh)
    log_line(f"[1] bundle ready: models={list(bundle.responses.keys())}, items={len(bundle.items)}")
    log_jsonl(log_fp, {"ts": _now(), "stage": "bundle_ready", "scenario": scenario, "batch_id": batch_id})

    # 2) Judge -> Ensemble
    t0 = time.time()
    verdicts, attributions = judge_bundle(bundle, ensemble_cfg=None)
    dt_ms = int((time.time() - t0) * 1000)
    log_line(f"[2] judge done: verdicts={len(verdicts)} attributions={len(attributions)} elapsed_ms={dt_ms}")
    log_jsonl(
        log_fp,
        {
            "ts": _now(),
            "stage": "judge_done",
            "scenario": scenario,
            "elapsed_ms": dt_ms,
            "verdicts": [v.model_dump() for v in verdicts],
            "attributions": [a.model_dump() for a in attributions],
        },
    )

    if not verdicts:
        log_line("[!] No verdicts returned. Stop.")
        return

    v0 = verdicts[0].model_dump()
    verdict = v0.get("verdict")
    log_line(f"[2.1] image verdict = {verdict}")

    # 3) Export
    exp_dir = out_dir / scenario
    labels_dir = exp_dir / "labels"
    ann_dir = exp_dir / "annotations"
    meta_dir = exp_dir / "meta"

    yolo_p = export_yolo_txt(v0, out_dir=labels_dir, image_wh=wh)
    coco_p = export_coco_json(v0, out_dir=ann_dir, image_wh=wh)
    meta_p = export_meta_json(v0, out_dir=meta_dir, image_wh=wh)

    log_line(f"[3] export:")
    log_line(f"    - yolo_txt: {str(yolo_p) if yolo_p else None}")
    log_line(f"    - coco_json: {str(coco_p) if coco_p else None}")
    log_line(f"    - meta_json: {str(meta_p)}")
    log_jsonl(
        log_fp,
        {
            "ts": _now(),
            "stage": "export_done",
            "scenario": scenario,
            "yolo_txt": str(yolo_p) if yolo_p else None,
            "coco_json": str(coco_p) if coco_p else None,
            "meta_json": str(meta_p),
        },
    )

    # 4) Sanity assertions (soft; prints only)
    if verdict in ("PASS_2", "PASS_3"):
        if yolo_p is None or coco_p is None:
            log_line("[!] ERROR: PASS verdict but label export missing.")
        else:
            log_line("[4] PASS export OK.")
    else:
        if yolo_p is not None or coco_p is not None:
            log_line("[!] ERROR: FAIL/MISS verdict but PASS exports were generated.")
        else:
            log_line("[4] FAIL/MISS export policy OK (no yolo/coco).")

    log_line(f"=== DONE: {scenario} ===\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="./_test_runs/v2_test_main", help="output directory")
    ap.add_argument("--w", type=int, default=1280, help="image width")
    ap.add_argument("--h", type=int, default=720, help="image height")
    ap.add_argument(
        "--scenarios",
        type=str,
        default="pass3,pass2,fail_single,miss_all,miss_disagree3,fail_weak_lowconf,fail_pass2_plus_single",
        help="comma separated scenario list",
    )
    ap.add_argument("--jsonl", action="store_true", help="write jsonl log file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wh = (int(args.w), int(args.h))

    log_fp: Optional[Path] = (out_dir / "test_main.log.jsonl") if args.jsonl else None

    log_line(f"V2 test_main start: out_dir={out_dir}")
    log_line(f"scenarios={args.scenarios}")

    for sc in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        run_one(sc, out_dir=out_dir, wh=wh, log_fp=log_fp)

    log_line("V2 test_main finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
