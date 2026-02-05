#!/usr/bin/env python3
"""
run_gt_train.py
- V2 Control Plane: GT 학습을 3개 모델(yolov11/rtdetr/yolox)에 "동시에" 트리거
- job_id 생성
- configs/models.yaml + configs/train_gt.yaml 로드
- 각 모델의 /train/gt 호출 (async)
- 결과를 logs/train/job_<job_id>.json 로 저장
- common/logging.py 기반 구조화 로깅 포함

추가:
- 첫 GT baseline weight는 반드시: gt_<model>_<YYYYMMDD>.pt 로 저장
  => 모든 loop의 기준점으로 사용

사용 예)
  python run_gt_train.py
  python run_gt_train.py --models yolov11,yolox
  python run_gt_train.py --configs ./configs --dry-run
  python run_gt_train.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except Exception:
    print("[FATAL] Missing dependency: pyyaml (pip install pyyaml)", file=sys.stderr)
    raise

try:
    import httpx  # pip install httpx
except Exception:
    print("[FATAL] Missing dependency: httpx (pip install httpx)", file=sys.stderr)
    raise

try:
    from common.logging import setup_logger, LogContext, log_event, StageTimer
except Exception:
    print("[FATAL] Missing common.logging. Ensure ./common/logging.py is importable.", file=sys.stderr)
    raise


# ----------------------------
# helpers
# ----------------------------
def _utc_now_iso_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _local_now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def make_job_id(prefix: str = "job") -> str:
    suf = secrets.token_hex(2)  # 4 hex chars
    return f"{prefix}_{_local_now_compact()}_{suf}"


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_models(all_models: List[str], only: Optional[List[str]]) -> List[str]:
    if not only:
        return all_models
    only_set = {m.strip() for m in only if m.strip()}
    return [m for m in all_models if m in only_set]


def _redact_payload_for_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    """로그용으로 payload를 '핵심만' 남기고 축약."""
    gt = payload.get("gt_source", {}) or {}
    out = payload.get("output", {}) or {}
    tr = payload.get("train", {}) or {}

    return {
        "job_id": payload.get("job_id"),
        "type": payload.get("type"),
        "model": payload.get("model"),
        "gt_source": {
            "mode": gt.get("mode"),
            "local_root": ((gt.get("local", {}) or {}).get("root")),
            "splits": gt.get("splits"),
        },
        "train": {
            "epochs": tr.get("epochs"),
            "imgsz": tr.get("imgsz"),
            "batch": tr.get("batch"),
            "amp": tr.get("amp"),
        },
        "output": {
            "weights_dir_in_container": out.get("weights_dir_in_container"),
            "naming": out.get("naming"),
            "baseline": out.get("baseline"),
            "baseline_tag": out.get("baseline_tag"),
        },
        "yolox": {
            "exp_file": ((payload.get("yolox", {}) or {}).get("exp_file")),
            "batch": ((payload.get("yolox", {}) or {}).get("batch")),
            "fp16": ((payload.get("yolox", {}) or {}).get("fp16")),
            "ckpt": ((payload.get("yolox", {}) or {}).get("ckpt")),
        } if payload.get("yolox") else None,
    }


def _truncate(obj: Any, limit: int = 2000) -> Any:
    """응답이 너무 길어지는 걸 방지(로그 파일 폭발 방지)."""
    try:
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) <= limit:
            return obj
        return {"_truncated": True, "len": len(s), "head": s[:limit]}
    except Exception:
        s = str(obj)
        if len(s) <= limit:
            return s
        return {"_truncated": True, "len": len(s), "head": s[:limit]}


# ----------------------------
# config models
# ----------------------------
@dataclass
class ModelTarget:
    name: str
    base_url: str
    train_gt_endpoint: str


def parse_models_cfg(models_cfg: Dict[str, Any]) -> Tuple[Dict[str, ModelTarget], Dict[str, Any]]:
    models = models_cfg.get("models", {}) or {}
    timeouts = models_cfg.get("timeouts_sec", {}) or {}
    targets: Dict[str, ModelTarget] = {}

    for name, spec in models.items():
        base_url = str(spec.get("base_url", "")).rstrip("/")
        ep = spec.get("endpoints", {}) or {}
        train_gt_ep = str(ep.get("train_gt", "/train/gt"))

        if not base_url:
            raise ValueError(f"models.yaml: models.{name}.base_url is empty")
        if not train_gt_ep.startswith("/"):
            train_gt_ep = "/" + train_gt_ep

        targets[name] = ModelTarget(name=name, base_url=base_url, train_gt_endpoint=train_gt_ep)

    if not targets:
        raise ValueError("models.yaml: no models found under 'models:'")

    return targets, timeouts


# ----------------------------
# HTTP call (+logging)
# ----------------------------
async def call_train_gt(
    client: httpx.AsyncClient,
    target: ModelTarget,
    payload: Dict[str, Any],
    timeout_sec: float,
    *,
    logger,
    ctx: LogContext,
) -> Dict[str, Any]:
    url = f"{target.base_url}{target.train_gt_endpoint}"
    model = str(payload.get("model", target.name))

    # request log (요약 payload)
    log_event(
        logger,
        "train_gt request",
        ctx=ctx,
        target_model=model,
        url=url,
        timeout_sec=timeout_sec,
        payload=_redact_payload_for_log(payload),
    )

    t0 = time.time()
    try:
        resp = await client.post(url, json=payload, timeout=timeout_sec)
        dt_ms = int((time.time() - t0) * 1000)
        content_type = resp.headers.get("content-type", "")

        if "application/json" in content_type.lower():
            try:
                data = resp.json()
            except Exception as e:
                data = {"json_parse_error": repr(e), "raw": resp.text}
        else:
            data = {"raw": resp.text}

        out = {
            "ok": resp.is_success,
            "status_code": resp.status_code,
            "elapsed_ms": dt_ms,
            "url": url,
            "response": data,
        }

        # response log (요약)
        log_event(
            logger,
            "train_gt response",
            ctx=ctx,
            target_model=model,
            ok=out["ok"],
            status_code=out["status_code"],
            elapsed_ms=out["elapsed_ms"],
            url=url,
            response=_truncate(out.get("response"), limit=2000),
        )

        return out

    except Exception as e:
        dt_ms = int((time.time() - t0) * 1000)
        err = {
            "ok": False,
            "status_code": None,
            "elapsed_ms": dt_ms,
            "url": url,
            "error": repr(e),
            "error_type": type(e).__name__,
        }

        log_event(
            logger,
            "train_gt exception",
            ctx=ctx,
            target_model=model,
            url=url,
            elapsed_ms=dt_ms,
            error=repr(e),
            error_type=type(e).__name__,
            level="ERROR",
        )
        return err


# ----------------------------
# payload builders
# ----------------------------
def _validate_gt_source(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = train_cfg.get("data", {}) or {}
    source_type = str(data_cfg.get("source_type", "")).strip().upper()
    if source_type != "GT":
        raise ValueError(f"train_gt.yaml: data.source_type must be 'GT' (got {source_type!r})")
    return data_cfg


def _base_payload(
    train_cfg: Dict[str, Any],
    job_id: str,
    model_name: str,
    data_cfg: Dict[str, Any],
    *,
    baseline_tag: str,
) -> Dict[str, Any]:
    job_cfg = train_cfg.get("job", {}) or {}
    output_cfg = train_cfg.get("output", {}) or {}
    tracking_cfg = train_cfg.get("tracking", {}) or {}

    local_cfg = dict(data_cfg.get("local", {}) or {})

    baseline_weight_name = f"gt_{model_name}_{baseline_tag}.pt"

    return {
        "job_id": job_id,
        "type": "GT_TRAIN",
        "model": model_name,
        "created_at": _utc_now_iso_z(),
        "job": {
            "artifacts_root": str(job_cfg.get("artifacts_root", "")),
        },
        "gt_source": {
            "mode": str(data_cfg.get("mode", "local")),
            "s3": data_cfg.get("s3", {}) or {},
            "local": local_cfg,
            "splits": data_cfg.get("splits", {}) or {},
            "source_type": "GT",
        },
        "output": {
            "weights_dir_in_container": str(output_cfg.get("weights_dir_in_container", "/weights")),
            "naming": baseline_weight_name,  # ✅ 고정 baseline 파일명
            "also_save_last": bool(output_cfg.get("also_save_last", True)),
            "baseline": True,
            "baseline_tag": baseline_tag,
        },
        "tracking": tracking_cfg,
    }


def build_train_payload_ultralytics(
    train_cfg: Dict[str, Any],
    job_id: str,
    model_name: str,
    *,
    baseline_tag: str,
) -> Dict[str, Any]:
    data_cfg = _validate_gt_source(train_cfg)
    train_params = train_cfg.get("train", {}) or {}

    payload = _base_payload(train_cfg, job_id, model_name, data_cfg, baseline_tag=baseline_tag)
    payload["train"] = {
        "epochs": int(train_params.get("epochs", 30)),
        "imgsz": int(train_params.get("imgsz", 640)),
        "batch": int(train_params.get("batch", 16)),
        "lr": float(train_params.get("lr", 0.001)),
        "weight_decay": float(train_params.get("weight_decay", 0.0005)),
        "workers": int(train_params.get("workers", 8)),
        "seed": int(train_params.get("seed", 42)),
        "amp": bool(train_params.get("amp", True)),
    }
    return payload


def build_train_payload_yolox(
    train_cfg: Dict[str, Any],
    job_id: str,
    model_name: str,
    *,
    baseline_tag: str,
) -> Dict[str, Any]:
    data_cfg = _validate_gt_source(train_cfg)
    train_params = train_cfg.get("train", {}) or {}
    yolox_cfg = train_cfg.get("yolox", {}) or {}

    payload = _base_payload(train_cfg, job_id, model_name, data_cfg, baseline_tag=baseline_tag)

    payload["yolox"] = {
        "exp_file": str(yolox_cfg.get("exp_file", "exps/custom/yolox_x_canon9.py")),
        "devices": int(yolox_cfg.get("devices", 1)),
        "batch": int(yolox_cfg.get("batch", train_params.get("batch", 8))),
        "fp16": bool(yolox_cfg.get("fp16", train_params.get("amp", True))),
        "occupy": bool(yolox_cfg.get("occupy", True)),
        "ckpt": str(yolox_cfg.get("ckpt", "/weights/yolox_x.pth")),
    }

    payload["train"] = {
        "epochs": int(train_params.get("epochs", 30)),
        "imgsz": int(train_params.get("imgsz", 640)),
        "workers": int(train_params.get("workers", 8)),
        "seed": int(train_params.get("seed", 42)),
    }
    return payload


def build_payload_for_model(
    train_cfg: Dict[str, Any],
    job_id: str,
    model_name: str,
    *,
    baseline_tag: str,
) -> Dict[str, Any]:
    if model_name.lower() == "yolox":
        return build_train_payload_yolox(train_cfg, job_id, model_name, baseline_tag=baseline_tag)
    return build_train_payload_ultralytics(train_cfg, job_id, model_name, baseline_tag=baseline_tag)


# ----------------------------
# main run
# ----------------------------
async def run(args: argparse.Namespace) -> int:
    default_log_file = Path("./logs/control/run_gt_train.log")
    log_file = Path(args.log_file) if args.log_file else default_log_file
    logger = setup_logger(service="v2-control", level=args.log_level, log_file=log_file)

    cfg_dir = Path(args.configs).resolve()
    models_yaml = cfg_dir / "models.yaml"
    train_yaml = cfg_dir / "train_gt.yaml"

    models_cfg = load_yaml(models_yaml)
    train_cfg = load_yaml(train_yaml)

    targets, timeouts = parse_models_cfg(models_cfg)
    all_model_names = list(targets.keys())

    only_models: Optional[List[str]] = None
    if args.models:
        only_models = [x.strip() for x in args.models.split(",") if x.strip()]

    chosen = select_models(all_model_names, only_models)
    if not chosen:
        log_event(logger, "no models selected", level="ERROR", available=all_model_names)
        return 2

    job_prefix = str((train_cfg.get("job", {}) or {}).get("id_prefix", "job"))
    job_id = args.job_id or make_job_id(job_prefix)
    baseline_tag = _today_yyyymmdd()

    ctx = LogContext(job_id=job_id, stage="gt_train_group", model="control")

    job_cfg = train_cfg.get("job", {}) or {}
    registry_dir = Path(str(job_cfg.get("registry_dir", "./logs/train"))).resolve()
    registry_path = registry_dir / f"job_{job_id}.json"

    timeout_train = float(timeouts.get("train_gt", 72000))
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=20)

    log_event(
        logger,
        "run_gt_train start",
        ctx=ctx,
        configs_dir=str(cfg_dir),
        models=chosen,
        registry=str(registry_path),
        timeout_train_sec=timeout_train,
        baseline_tag=baseline_tag,
    )

    if args.dry_run:
        log_event(logger, "dry-run enabled", ctx=ctx, baseline_tag=baseline_tag)
        for m in chosen:
            payload = build_payload_for_model(train_cfg, job_id, m, baseline_tag=baseline_tag)
            log_event(
                logger,
                "payload",
                ctx=ctx,
                target_model=m,
                gt_root=str(((payload.get("gt_source", {}) or {}).get("local", {}) or {}).get("root", "")),
                baseline_weight=str(((payload.get("output", {}) or {}).get("naming", ""))),
                payload=_redact_payload_for_log(payload),
            )
        return 0

    started_at = _utc_now_iso_z()
    summary: Dict[str, Any] = {
        "job_id": job_id,
        "type": "GT_TRAIN_GROUP",
        "models": chosen,
        "baseline_tag": baseline_tag,
        "baseline_weights": {m: f"gt_{m}_{baseline_tag}.pt" for m in chosen},
        "started_at": started_at,
        "finished_at": None,
        "status": "RUNNING",
        "results": {},
        "configs": {
            "models_yaml": str(models_yaml),
            "train_yaml": str(train_yaml),
        },
    }

    with StageTimer(logger, "trigger /train/gt concurrently", ctx=ctx, models=chosen):
        async with httpx.AsyncClient(limits=limits) as client:
            tasks: Dict[str, asyncio.Task] = {}

            for m in chosen:
                target = targets[m]
                payload = build_payload_for_model(train_cfg, job_id, m, baseline_tag=baseline_tag)

                # dispatch log (짧게)
                log_event(
                    logger,
                    "dispatch train_gt",
                    ctx=ctx,
                    target_model=m,
                    url=f"{target.base_url}{target.train_gt_endpoint}",
                    gt_root=str(((payload.get("gt_source", {}) or {}).get("local", {}) or {}).get("root", "")),
                    baseline_weight=str(((payload.get("output", {}) or {}).get("naming", ""))),
                )

                tasks[m] = asyncio.create_task(
                    call_train_gt(
                        client,
                        target,
                        payload,
                        timeout_train,
                        logger=logger,
                        ctx=ctx,
                    )
                )

            results_by_model: Dict[str, Any] = {}
            for m, t in tasks.items():
                results_by_model[m] = await t

    all_ok = True
    for m in chosen:
        res = results_by_model[m]
        summary["results"][m] = res
        ok = bool(res.get("ok", False))
        if not ok:
            all_ok = False
            log_event(logger, "train_gt failed", ctx=ctx, target_model=m, result=_truncate(res, 2000), level="ERROR")
        else:
            log_event(logger, "train_gt done", ctx=ctx, target_model=m, result=_truncate(res, 2000))

    summary["finished_at"] = _utc_now_iso_z()
    summary["status"] = "DONE" if all_ok else "PARTIAL_FAIL"

    write_json(registry_path, summary)
    log_event(logger, "registry saved", ctx=ctx, status=summary["status"], registry=str(registry_path))

    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="./configs", help="configs directory containing models.yaml, train_gt.yaml")
    ap.add_argument("--models", default="", help="comma-separated model names to train (default: all in models.yaml)")
    ap.add_argument("--job-id", default="", help="override job_id (default: auto)")
    ap.add_argument("--dry-run", action="store_true", help="print payloads and exit")
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    ap.add_argument("--log-file", default="", help="log file path (default: ./logs/control/run_gt_train.log)")

    args = ap.parse_args()
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[INFO] cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[FATAL] {repr(e)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
