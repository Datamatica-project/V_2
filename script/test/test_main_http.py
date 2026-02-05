#!/usr/bin/env python3
"""
V2 HTTP End-to-End smoke test (compose-aligned).

Compose ports (your docker-compose.yml):
- yolov11 : http://localhost:8020
- rtdetr  : http://localhost:8021
- rtm     : http://localhost:8022
- judge   : http://localhost:8025
- (no gateway service in compose)

Modes:
- judge_infer (recommended for your compose):
    Send InferBatchRequest to judge. Judge fans out to models (internal urls) and runs ensemble.
- models_then_judge:
    Call 3 model servers directly, build bundle locally, then post bundle to judge (if judge has such endpoint).
- auto:
    If --gateway is given, use it; else try judge_infer first; else fallback.

Run examples (compose-aligned):
  # Recommended: judge does everything (fan-out + ensemble)
  PYTHONPATH=. python script/test/test_main_http.py \
    --mode judge_infer \
    --judge  http://localhost:8025 \
    --out-dir ./_test_runs/http_e2e --jsonl \
    --image-path /data/unlabeled/img1.jpg

  # Alternative: call models directly, then try bundle->judge (only if judge supports bundle endpoint)
  PYTHONPATH=. python script/test/test_main_http.py \
    --mode models_then_judge \
    --yolov11 http://localhost:8020 \
    --rtm    http://localhost:8022 \
    --rtdetr http://localhost:8021 \
    --judge  http://localhost:8025 \
    --out-dir ./_test_runs/http_e2e --jsonl \
    --image-path /data/unlabeled/img1.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# local export (always available)
from src.ensemble.export import export_coco_json, export_meta_json, export_yolo_txt


# -------------------------
# HTTP client (requests preferred)
# -------------------------
def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("requests is required for HTTP test_main_http. pip install requests") from e

    r = requests.post(url, json=payload, timeout=timeout)
    ct = (r.headers.get("content-type") or "").lower()
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} POST {url}: {r.text[:800]}")
    if "application/json" not in ct:
        try:
            return r.json()
        except Exception:
            raise RuntimeError(f"Non-JSON response from {url}: {r.text[:800]}")
    return r.json()


def _http_get_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("requests is required for HTTP test_main_http. pip install requests") from e

    r = requests.get(url, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} GET {url}: {r.text[:800]}")
    return r.json()


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


# -------------------------
# payload builders (SSOT-ish shape)
# -------------------------
def build_infer_request(
    *,
    batch_id: str,
    image_id: str,
    image_path: str,
    wh: Tuple[int, int],
    run_id: str,
    user_key: str,
    project_id: str,
    weight_dir: Optional[str],
) -> Dict[str, Any]:
    w, h = wh
    params: Dict[str, Any] = {
        "run_id": run_id,
        "user_key": user_key,
        "project_id": project_id,
    }
    if weight_dir:
        params["weight_dir"] = weight_dir

    return {
        "batch_id": batch_id,
        "items": [
            {
                "image_id": image_id,
                "image_path": image_path,
                "width": int(w),
                "height": int(h),
            }
        ],
        "params": params,
    }


def _try_post_first_ok(
    base_url: str,
    endpoint_candidates: List[str],
    payload: Dict[str, Any],
    timeout: int,
    *,
    label: str,
) -> Tuple[str, Dict[str, Any]]:
    last_err: Optional[Exception] = None
    for ep in endpoint_candidates:
        ep = ep.strip()
        if not ep:
            continue
        if not ep.startswith("/"):
            ep = "/" + ep
        url = base_url.rstrip("/") + ep
        try:
            out = _http_post_json(url, payload, timeout=timeout)
            return url, out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"{label} failed for all endpoints. last_error={last_err}")


# -------------------------
# Bundle builders
# -------------------------
def bundle_from_3_model_responses(
    *,
    batch_id: str,
    infer_req: Dict[str, Any],
    yolov11_resp: Dict[str, Any],
    rtm_resp: Dict[str, Any],
    rtdetr_resp: Dict[str, Any],
) -> Dict[str, Any]:
    # GatewayBundle(SSOT) json shape used in your code path:
    # {batch_id, items, responses: {model: ModelPredResponse}, errors:{}}
    return {
        "batch_id": batch_id,
        "items": infer_req.get("items", []),
        "responses": {
            "yolov11": yolov11_resp,
            "rtm": rtm_resp,
            "rtdetr": rtdetr_resp,
        },
        "errors": {},
    }


# -------------------------
# Response parsing helpers
# -------------------------
def _pick_first_verdict(j_out: Any, infer_req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept common judge response shapes and extract first verdict dict.
    """
    verdicts = None
    if isinstance(j_out, dict):
        verdicts = j_out.get("verdicts") or j_out.get("results") or j_out.get("verdict") or j_out.get("data")
    if verdicts is None:
        verdicts = j_out

    if isinstance(verdicts, list) and verdicts:
        v0 = verdicts[0]
        if isinstance(v0, dict):
            return v0
        raise RuntimeError(f"Judge verdicts[0] is not a dict: type={type(v0)}")
    if isinstance(verdicts, dict) and verdicts:
        return verdicts

    keys = list(j_out.keys()) if isinstance(j_out, dict) else []
    raise RuntimeError(f"Judge response does not contain verdicts. keys={keys} type={type(j_out)} image_id={infer_req['items'][0]['image_id']}")


# -------------------------
# main flow
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="./_test_runs/http_e2e")
    ap.add_argument("--jsonl", action="store_true")

    ap.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "judge_infer", "models_then_judge", "gateway_then_judge"],
        help="compose-aligned default is 'auto' (prefers judge_infer when no gateway).",
    )

    # Identity
    ap.add_argument("--run-id", type=str, default=f"run_{int(time.time())}")
    ap.add_argument("--user-key", type=str, default="u1")
    ap.add_argument("--project-id", type=str, default="p1")

    # Image stub (HTTP에서는 실제 파일 업로드가 아니라, 경로만 전달하는 구조 가정)
    ap.add_argument("--image-id", type=str, default="img1")
    ap.add_argument("--image-path", type=str, default="/data/unlabeled/img1.jpg")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)

    # Services (compose defaults)
    ap.add_argument("--gateway", type=str, default="", help="gateway base url (NOT in your compose by default)")
    ap.add_argument("--judge", type=str, default="http://localhost:8025", help="judge base url (compose: 8025)")

    ap.add_argument("--yolov11", type=str, default="http://localhost:8020", help="yolov11 base url (compose: 8020)")
    ap.add_argument("--rtdetr", type=str, default="http://localhost:8021", help="rtdetr base url (compose: 8021)")
    ap.add_argument("--rtm", type=str, default="http://localhost:8022", help="rtm base url (compose: 8022)")

    # Optional fixed endpoints (if auto-detect fails)
    ap.add_argument("--gateway-endpoint", type=str, default="", help="force gateway endpoint path (e.g. /infer)")
    ap.add_argument("--judge-endpoint", type=str, default="", help="force judge endpoint path (e.g. /infer or /judge)")

    # Weight scope hint (모델 infer에서 사용: params.weight_dir)
    ap.add_argument(
        "--weight-dir",
        type=str,
        default="",
        help="optional weight_dir scope hint (e.g. /weights/projects/u1/p1)",
    )

    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = (out_dir / "test_main_http.log.jsonl") if args.jsonl else None

    wh = (int(args.w), int(args.h))
    batch_id = f"batch_http_{args.run_id}"

    log_line(f"test_main_http start: out_dir={out_dir}")
    log_line(f"run_id={args.run_id} user_key={args.user_key} project_id={args.project_id} mode={args.mode}")
    log_jsonl(log_fp, {"ts": _now(), "stage": "start", "out_dir": str(out_dir), "mode": args.mode})

    infer_req = build_infer_request(
        batch_id=batch_id,
        image_id=args.image_id,
        image_path=args.image_path,
        wh=wh,
        run_id=args.run_id,
        user_key=args.user_key,
        project_id=args.project_id,
        weight_dir=(args.weight_dir.strip() or None),
    )

    # -------------------------
    # (0) health checks (best-effort)
    # -------------------------
    for name, base in [
        ("gateway", args.gateway),
        ("judge", args.judge),
        ("yolov11", args.yolov11),
        ("rtm", args.rtm),
        ("rtdetr", args.rtdetr),
    ]:
        if base.strip():
            try:
                h = _http_get_json(base.rstrip("/") + "/health")
                log_line(f"[health] {name}: {h}")
            except Exception as e:
                log_line(f"[health] {name}: SKIP/FAIL ({e})")

    # -------------------------
    # mode auto-resolution
    # -------------------------
    mode = args.mode
    if mode == "auto":
        if args.gateway.strip():
            mode = "gateway_then_judge"
        else:
            # compose-aligned: prefer judge_infer first
            mode = "judge_infer"

    # -------------------------
    # (1) Build verdict0 (either via judge_infer or via bundle->judge)
    # -------------------------
    verdict0: Dict[str, Any] = {}
    debug_payloads: Dict[str, Any] = {"infer_req": infer_req}

    def _do_judge_infer() -> Tuple[str, Dict[str, Any]]:
        # Judge endpoint candidates for "InferBatchRequest -> verdicts"
        if args.judge_endpoint.strip():
            j_eps = [args.judge_endpoint.strip()]
        else:
            j_eps = [
                "/infer",            # likely (single-batch infer)
                "/infer/batch",      # alt
                "/infer/once",       # alt
                "/api/infer",        # alt
                "/run/infer",        # alt
            ]
        log_line("[1] calling judge (infer_req)...")
        log_jsonl(log_fp, {"ts": _now(), "stage": "judge_infer_call_start", "candidates": j_eps})
        j_url, j_out = _try_post_first_ok(args.judge, j_eps, infer_req, timeout=args.timeout, label="Judge(infer)")
        log_line(f"[1] judge(infer) OK: url={j_url}")
        log_jsonl(log_fp, {"ts": _now(), "stage": "judge_infer_call_ok", "url": j_url})
        (out_dir / "judge_infer_response.json").write_text(json.dumps(j_out, ensure_ascii=False, indent=2), encoding="utf-8")
        return j_url, j_out

    def _do_models_then_judge_bundle() -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        # call models
        model_eps = ["/infer"]  # model api contract
        log_line("[1] calling models directly (compose ports)...")

        y_url, y_out = _try_post_first_ok(args.yolov11, model_eps, infer_req, timeout=args.timeout, label="YOLOv11")
        r_url, r_out = _try_post_first_ok(args.rtm, model_eps, infer_req, timeout=args.timeout, label="RTM")
        d_url, d_out = _try_post_first_ok(args.rtdetr, model_eps, infer_req, timeout=args.timeout, label="RT-DETR")

        log_line(f"[1] yolov11 OK: {y_url}")
        log_line(f"[1] rtm OK:     {r_url}")
        log_line(f"[1] rtdetr OK:  {d_url}")
        log_jsonl(log_fp, {"ts": _now(), "stage": "model_calls_ok", "y": y_url, "r": r_url, "d": d_url})

        bundle = bundle_from_3_model_responses(
            batch_id=batch_id,
            infer_req=infer_req,
            yolov11_resp=y_out,
            rtm_resp=r_out,
            rtdetr_resp=d_out,
        )

        # persist bundle
        (out_dir / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        log_line(f"[1] bundle saved: {out_dir / 'bundle.json'}")

        # post bundle to judge (only if judge has that endpoint)
        if args.judge_endpoint.strip():
            j_eps = [args.judge_endpoint.strip()]
        else:
            j_eps = [
                "/judge",            # common
                "/judge/bundle",     # alt
                "/bundle/judge",     # alt
                "/infer/judge",      # alt
                "/api/judge",        # alt
                "/ensemble",         # alt
            ]

        log_line("[2] calling judge (bundle)...")
        log_jsonl(log_fp, {"ts": _now(), "stage": "judge_bundle_call_start", "candidates": j_eps})

        j_url, j_out = _try_post_first_ok(args.judge, j_eps, bundle, timeout=args.timeout, label="Judge(bundle)")
        log_line(f"[2] judge(bundle) OK: url={j_url}")
        log_jsonl(log_fp, {"ts": _now(), "stage": "judge_bundle_call_ok", "url": j_url})
        (out_dir / "judge_bundle_response.json").write_text(json.dumps(j_out, ensure_ascii=False, indent=2), encoding="utf-8")

        return j_url, j_out, bundle

    if mode == "judge_infer":
        if not args.judge.strip():
            raise RuntimeError("--judge is required for judge_infer mode.")
        j_url, j_out = _do_judge_infer()
        debug_payloads["judge_url"] = j_url
        debug_payloads["judge_out"] = j_out
        verdict0 = _pick_first_verdict(j_out, infer_req)

    elif mode == "models_then_judge":
        if not args.judge.strip():
            raise RuntimeError("--judge is required for models_then_judge mode.")
        try:
            j_url, j_out, bundle = _do_models_then_judge_bundle()
            debug_payloads["judge_url"] = j_url
            debug_payloads["judge_out"] = j_out
            debug_payloads["bundle"] = bundle
            verdict0 = _pick_first_verdict(j_out, infer_req)
        except Exception as e:
            # fallback: judge_infer (compose-aligned)
            log_line(f"[!] models_then_judge failed: {e}")
            log_line("[!] fallback -> judge_infer")
            log_jsonl(log_fp, {"ts": _now(), "stage": "fallback_to_judge_infer", "error": str(e)})

            j_url, j_out = _do_judge_infer()
            debug_payloads["judge_url"] = j_url
            debug_payloads["judge_out"] = j_out
            verdict0 = _pick_first_verdict(j_out, infer_req)

    elif mode == "gateway_then_judge":
        if not args.gateway.strip():
            raise RuntimeError("--gateway is required for gateway_then_judge mode.")
        if not args.judge.strip():
            raise RuntimeError("--judge is required for gateway_then_judge mode.")

        # Gateway endpoint candidates
        if args.gateway_endpoint.strip():
            gw_eps = [args.gateway_endpoint.strip()]
        else:
            gw_eps = [
                "/infer",
                "/bundle",
                "/infer/bundle",
                "/gateway/infer",
                "/api/infer",
            ]

        log_line("[1] calling gateway...")
        log_jsonl(log_fp, {"ts": _now(), "stage": "gateway_call_start", "candidates": gw_eps})

        gw_url, gw_out = _try_post_first_ok(
            args.gateway,
            gw_eps,
            infer_req,
            timeout=args.timeout,
            label="Gateway",
        )
        log_line(f"[1] gateway OK: url={gw_url}")
        log_jsonl(log_fp, {"ts": _now(), "stage": "gateway_call_ok", "url": gw_url})

        bundle = gw_out
        (out_dir / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        log_line(f"[1] bundle saved: {out_dir / 'bundle.json'}")

        # Judge with bundle
        if args.judge_endpoint.strip():
            j_eps = [args.judge_endpoint.strip()]
        else:
            j_eps = ["/judge", "/judge/bundle", "/infer/judge", "/api/judge", "/run"]

        log_line("[2] calling judge (bundle)...")
        log_jsonl(log_fp, {"ts": _now(), "stage": "judge_call_start", "candidates": j_eps})

        j_url, j_out = _try_post_first_ok(args.judge, j_eps, bundle, timeout=args.timeout, label="Judge")
        log_line(f"[2] judge OK: url={j_url}")
        log_jsonl(log_fp, {"ts": _now(), "stage": "judge_call_ok", "url": j_url})

        (out_dir / "judge_response.json").write_text(json.dumps(j_out, ensure_ascii=False, indent=2), encoding="utf-8")
        verdict0 = _pick_first_verdict(j_out, infer_req)

    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    # Persist debug envelope
    (out_dir / "debug_payloads.json").write_text(json.dumps(debug_payloads, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = str(verdict0.get("verdict", ""))
    image_id = str(verdict0.get("image_id", infer_req["items"][0]["image_id"]))
    log_line(f"[2.1] verdict(image_id={image_id}) = {verdict}")

    # -------------------------
    # (3) Export (local)
    # -------------------------
    exp_dir = out_dir / "export"
    labels_dir = exp_dir / "labels"
    ann_dir = exp_dir / "annotations"
    meta_dir = exp_dir / "meta"

    yolo_p = export_yolo_txt(verdict0, out_dir=labels_dir, image_wh=wh)
    coco_p = export_coco_json(verdict0, out_dir=ann_dir, image_wh=wh)
    meta_p = export_meta_json(verdict0, out_dir=meta_dir, image_wh=wh)

    log_line("[3] export done:")
    log_line(f"    - yolo_txt: {str(yolo_p) if yolo_p else None}")
    log_line(f"    - coco_json: {str(coco_p) if coco_p else None}")
    log_line(f"    - meta_json: {str(meta_p)}")
    log_jsonl(
        log_fp,
        {
            "ts": _now(),
            "stage": "export_done",
            "verdict": verdict,
            "yolo_txt": str(yolo_p) if yolo_p else None,
            "coco_json": str(coco_p) if coco_p else None,
            "meta_json": str(meta_p),
        },
    )

    # -------------------------
    # (4) Policy sanity
    # -------------------------
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

    log_line("test_main_http finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
