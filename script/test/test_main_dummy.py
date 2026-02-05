#!/usr/bin/env python3
"""
V2 Dummy Ensemble Smoke Test (NO HTTP, NO IMAGE FILE).

- Build dummy GatewayBundle with standardized ModelPredResponse.results (multi-image batch)
- Run src.ensemble.core.ensemble_bundle()  ✅ (real ensemble)
- Export YOLO/COCO/META using src.ensemble.export

Key: dump-level to avoid "file hell"
  --dump-level summary (default): save only per-batch summary + merged + run_summary
  --dump-level full: save per-batch infer_req/bundle/verdicts/attributions too
  --dump-level none: save only merged + run_summary (no per-batch files)

Examples:
  # Two batches of 64, mixed, merged, minimal files (recommended)
  PYTHONPATH=. python script/test/test_main_dummy.py \
    --batches 2 --n-per-batch 64 --mix --merge \
    --dump-level summary \
    --out-dir ./_test_runs/dummy64x2 --jsonl

  # Same but ultra-minimal (no per-batch files at all)
  PYTHONPATH=. python script/test/test_main_dummy.py \
    --batches 2 --n-per-batch 64 --mix --merge \
    --dump-level none \
    --out-dir ./_test_runs/dummy64x2_min --jsonl

  # Full debug dump (only when you need to inspect)
  PYTHONPATH=. python script/test/test_main_dummy.py \
    --batches 2 --n-per-batch 64 --mix --merge \
    --dump-level full \
    --out-dir ./_test_runs/dummy64x2_full --jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ensemble.core import ensemble_bundle
from src.ensemble.export import export_coco_json, export_meta_json, export_yolo_txt


# -------------------------
# logging
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# build request (same shape as HTTP test)
# -------------------------
def build_infer_request(
    *,
    batch_id: str,
    base_image_id: str,
    base_image_path: str,
    wh: Tuple[int, int],
    run_id: str,
    user_key: str,
    project_id: str,
    weight_dir: Optional[str],
    n: int,
) -> Dict[str, Any]:
    w, h = wh
    params: Dict[str, Any] = {
        "run_id": run_id,
        "user_key": user_key,
        "project_id": project_id,
    }
    if weight_dir:
        params["weight_dir"] = weight_dir

    items: List[Dict[str, Any]] = []
    for i in range(int(n)):
        image_id = f"{base_image_id}_{i:04d}" if n > 1 else base_image_id

        # NOTE: avoid rstrip('.jpg') because it strips any of chars {'.','j','p','g'}.
        if n > 1:
            if base_image_path.lower().endswith(".jpg"):
                stem = base_image_path[:-4]
                image_path = f"{stem}_{i:04d}.jpg"
            else:
                image_path = f"{base_image_path}_{i:04d}.jpg"
        else:
            image_path = base_image_path

        items.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "width": int(w),
                "height": int(h),
            }
        )

    return {
        "batch_id": batch_id,
        "items": items,
        "params": params,
    }


def _model_resp(model: str, batch_id: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "model": model,
        "batch_id": batch_id,
        "ok": True,
        "error": None,
        "elapsed_ms": 1,
        "weight_used": f"/weights/{model}/dummy.pt",
        "results": results,
        "raw": {},
    }


def _dets_for_case(case: str, *, shift: float = 0.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    box_a = [100.0 + shift, 120.0 + shift, 260.0 + shift, 300.0 + shift]
    box_b = [102.0 + shift, 118.0 + shift, 262.0 + shift, 302.0 + shift]
    box_c = [98.0 + shift, 123.0 + shift, 255.0 + shift, 298.0 + shift]

    if case == "pass3":
        return (
            [{"cls": 0, "conf": 0.92, "xyxy": box_a}],
            [{"cls": 0, "conf": 0.90, "xyxy": box_b}],
            [{"cls": 0, "conf": 0.88, "xyxy": box_c}],
        )
    if case == "pass2":
        return (
            [{"cls": 0, "conf": 0.90, "xyxy": box_a}],
            [{"cls": 0, "conf": 0.86, "xyxy": box_b}],
            [],
        )
    if case == "fail_single":
        return (
            [{"cls": 0, "conf": 0.91, "xyxy": box_a}],
            [],
            [],
        )
    if case == "fail_weak":
        return (
            [{"cls": 0, "conf": 0.30, "xyxy": box_a}],
            [{"cls": 0, "conf": 0.29, "xyxy": box_b}],
            [],
        )
    if case == "miss":
        return [], [], []

    raise ValueError(f"unknown case: {case}")


def _case_for_index(i: int) -> str:
    # total=64 fixed pattern
    if i < 32:
        return "pass3"
    if i < 44:
        return "pass2"
    if i < 52:
        return "fail_single"
    if i < 60:
        return "fail_weak"
    return "miss"


def _dummy_bundle(case: str, *, batch_id: str, infer_req: Dict[str, Any], mix: bool) -> Dict[str, Any]:
    items = infer_req["items"]

    y_results: List[Dict[str, Any]] = []
    r_results: List[Dict[str, Any]] = []
    d_results: List[Dict[str, Any]] = []

    for i, it in enumerate(items):
        image_id = str(it["image_id"])
        c = _case_for_index(i) if mix else case
        shift = float((i % 7) * 0.5)
        y_dets, r_dets, d_dets = _dets_for_case(c, shift=shift)

        y_results.append({"image_id": image_id, "detections": y_dets})
        r_results.append({"image_id": image_id, "detections": r_dets})
        d_results.append({"image_id": image_id, "detections": d_dets})

    return {
        "batch_id": batch_id,
        "items": items,
        "responses": {
            "yolov11": _model_resp("yolov11", batch_id, y_results),
            "rtm": _model_resp("rtm", batch_id, r_results),
            "rtdetr": _model_resp("rtdetr", batch_id, d_results),
        },
        "errors": {},
    }


def _count_verdicts(verdicts: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in verdicts:
        k = str(v.get("verdict", ""))
        out[k] = out.get(k, 0) + 1
    return out


def _merge_sanity_check(all_verdicts: List[Dict[str, Any]], all_attributions: List[Dict[str, Any]], expected_total: int) -> None:
    if len(all_verdicts) != expected_total:
        raise RuntimeError(f"merged verdicts count mismatch: got={len(all_verdicts)} expected={expected_total}")

    ids = [str(v.get("image_id", "")) for v in all_verdicts]
    if len(ids) != len(set(ids)):
        seen = set()
        dups = []
        for x in ids:
            if x in seen:
                dups.append(x)
            seen.add(x)
        raise RuntimeError(f"image_id collision across batches! duplicates(sample)={dups[:10]}")

    n_bad = sum(1 for v in all_verdicts if str(v.get("verdict")) in ("FAIL", "MISS"))
    if len(all_attributions) != n_bad:
        raise RuntimeError(f"attributions mismatch: got={len(all_attributions)} expected(FAIL+MISS)={n_bad}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="./_test_runs/dummy")
    ap.add_argument("--jsonl", action="store_true")

    ap.add_argument("--case", type=str, default="pass3", choices=["pass3", "pass2", "fail_single", "fail_weak", "miss"])
    ap.add_argument("--mix", action="store_true", help="mix cases across items when n-per-batch==64")
    ap.add_argument("--export-all", action="store_true", help="export all merged verdicts (default exports only first)")
    ap.add_argument("--merge", action="store_true", help="save merged verdicts/attributions + sanity checks")

    ap.add_argument("--dump-level", type=str, default="summary", choices=["none", "summary", "full"])
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--n-per-batch", type=int, default=64)

    ap.add_argument("--run-id", type=str, default=f"run_{int(time.time())}")
    ap.add_argument("--user-key", type=str, default="u1")
    ap.add_argument("--project-id", type=str, default="p1")

    ap.add_argument("--image-id", type=str, default="img_dummy", help="base image_id prefix")
    ap.add_argument("--image-path", type=str, default="/dummy/path/img_dummy.jpg")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)

    ap.add_argument("--weight-dir", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (out_dir / "test_main_dummy.log.jsonl") if args.jsonl else None
    wh = (int(args.w), int(args.h))

    log_line(
        f"start: out_dir={out_dir} case={args.case} batches={args.batches} n_per_batch={args.n_per_batch} "
        f"mix={bool(args.mix)} merge={bool(args.merge)} dump_level={args.dump_level}"
    )
    log_jsonl(
        log_fp,
        {
            "ts": _now(),
            "stage": "start",
            "case": args.case,
            "batches": int(args.batches),
            "n_per_batch": int(args.n_per_batch),
            "mix": bool(args.mix),
            "merge": bool(args.merge),
            "dump_level": args.dump_level,
            "out_dir": str(out_dir),
        },
    )

    all_verdicts: List[Dict[str, Any]] = []
    all_attributions: List[Dict[str, Any]] = []
    batch_summaries: List[Dict[str, Any]] = []

    for bi in range(int(args.batches)):
        batch_id = f"batch_dummy_{args.run_id}_b{bi:02d}"
        base_image_id = f"{args.image_id}_b{bi:02d}"  # avoid collision across batches

        infer_req = build_infer_request(
            batch_id=batch_id,
            base_image_id=base_image_id,
            base_image_path=args.image_path,
            wh=wh,
            run_id=args.run_id,
            user_key=args.user_key,
            project_id=args.project_id,
            weight_dir=(args.weight_dir.strip() or None),
            n=int(args.n_per_batch),
        )

        bundle = _dummy_bundle(args.case, batch_id=batch_id, infer_req=infer_req, mix=bool(args.mix))

        # dump-level full: save heavy artifacts
        if args.dump_level == "full":
            _write_json(out_dir / f"infer_req_b{bi:02d}.json", infer_req)
            _write_json(out_dir / f"bundle_b{bi:02d}.json", bundle)

        verdicts, attributions = ensemble_bundle(bundle)
        counts = _count_verdicts(verdicts)

        # dump-level full: save per-batch outputs
        if args.dump_level == "full":
            _write_json(out_dir / f"verdicts_b{bi:02d}.json", verdicts)
            _write_json(out_dir / f"attributions_b{bi:02d}.json", attributions)

        # dump-level summary: save only batch summary
        if args.dump_level == "summary":
            summary = {
                "batch_index": bi,
                "batch_id": batch_id,
                "n_items": len(infer_req.get("items", [])),
                "verdicts": len(verdicts),
                "attributions": len(attributions),
                "counts": counts,
            }
            _write_json(out_dir / f"batch_b{bi:02d}.summary.json", summary)
            batch_summaries.append(summary)

        log_line(f"[batch {bi:02d}] verdicts={len(verdicts)} attributions={len(attributions)} counts={counts}")
        log_jsonl(
            log_fp,
            {
                "ts": _now(),
                "stage": "batch_done",
                "batch_index": bi,
                "batch_id": batch_id,
                "verdicts": len(verdicts),
                "attributions": len(attributions),
                "counts": counts,
            },
        )

        all_verdicts.extend(verdicts)
        all_attributions.extend(attributions)

    expected_total = int(args.batches) * int(args.n_per_batch)
    merged_counts = _count_verdicts(all_verdicts)
    log_line(f"[merged] verdicts={len(all_verdicts)} attributions={len(all_attributions)} counts={merged_counts}")

    # run summary always
    run_summary = {
        "run_id": args.run_id,
        "user_key": args.user_key,
        "project_id": args.project_id,
        "batches": int(args.batches),
        "n_per_batch": int(args.n_per_batch),
        "expected_total": expected_total,
        "total_verdicts": len(all_verdicts),
        "total_attributions": len(all_attributions),
        "merged_counts": merged_counts,
        "mix": bool(args.mix),
        "case": args.case,
        "dump_level": args.dump_level,
    }
    if batch_summaries:
        run_summary["batch_summaries"] = batch_summaries
    _write_json(out_dir / "run_summary.json", run_summary)

    # merge outputs optional
    if args.merge:
        _merge_sanity_check(all_verdicts, all_attributions, expected_total)
        _write_json(out_dir / "verdicts_merged.json", all_verdicts)
        _write_json(out_dir / "attributions_merged.json", all_attributions)
        log_line("[merged] saved verdicts_merged.json / attributions_merged.json")

    if not all_verdicts:
        raise RuntimeError("No verdicts returned.")

    # export (merged 기준)
    exp_dir = out_dir / "export"
    labels_dir = exp_dir / "labels"
    ann_dir = exp_dir / "annotations"
    meta_dir = exp_dir / "meta"

    to_export = all_verdicts if args.export_all else [all_verdicts[0]]
    log_line(f"[export] exporting {len(to_export)} verdict(s) (export_all={bool(args.export_all)}) ...")

    n_pass = 0
    for v in to_export:
        verdict = str(v.get("verdict", ""))
        image_id = str(v.get("image_id", ""))

        yolo_p = export_yolo_txt(v, out_dir=labels_dir, image_wh=wh)
        coco_p = export_coco_json(v, out_dir=ann_dir, image_wh=wh)
        _ = export_meta_json(v, out_dir=meta_dir, image_wh=wh)

        if verdict in ("PASS_2", "PASS_3"):
            n_pass += 1
            if yolo_p is None or coco_p is None:
                log_line(f"[!] ERROR: PASS but missing export for image_id={image_id}")
        else:
            if yolo_p is not None or coco_p is not None:
                log_line(f"[!] ERROR: FAIL/MISS but exports generated for image_id={image_id}")

    log_line(f"[export] done. exported={len(to_export)} pass_in_exported={n_pass}")
    log_line("finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
