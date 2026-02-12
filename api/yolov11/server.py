# api/yolov11/server.py
from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.common.dto.dto import (
    Detection,
    GTTrainRequest,
    GTTrainResponse,
    InferBatchRequest,
    InferItemResult,
    ModelName,
    ModelPredResponse,
)
from src.common.utils.logging import LogContext, log_event, setup_logger

# -----------------------------
# Ultralytics YOLO
# -----------------------------
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


MODEL_NAME: ModelName = "yolov11"

# =============================================================================
# ✅ GPU policy (Simple default = GPU 1)
# - 요청에서 device가 와도 차단/기록하지 않고 무시.
# - ultralytics 호출에 device="1"를 항상 주입해서 GPU1을 사용하게 한다.
# =============================================================================
FIXED_GPU_INDEX = os.getenv("FIXED_GPU_INDEX", "1").strip() or "1"


def _ultra_device() -> str:
    return str(FIXED_GPU_INDEX)


GT_CURRENT_ROOT = Path(os.getenv("GT_CURRENT_ROOT", "/workspace/storage/GT/current")).resolve()
GT_VERSIONS_ROOT = Path(os.getenv("GT_VERSIONS_ROOT", "/workspace/storage/GT_versions")).resolve()

DEFAULT_REGISTRY_ROOT = os.getenv("MODEL_REGISTRY_ROOT", "/workspace/logs/model_train")
REGISTRY_DIR = (Path(DEFAULT_REGISTRY_ROOT) / str(MODEL_NAME)).resolve()
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_ROOT = Path(os.getenv("WEIGHTS_ROOT", "/weights")).resolve()
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "640"))
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.45"))

logger = setup_logger(
    service=os.getenv("MODEL_NAME", str(MODEL_NAME)),
    log_file=os.getenv("MODEL_LOG_FILE", f"/workspace/logs/{MODEL_NAME}/api.jsonl"),
    level=os.getenv("LOG_LEVEL", "INFO"),
)

app = FastAPI(title=f"{MODEL_NAME}-api", version="0.4.5")

log_event(
    logger,
    "service boot",
    ctx=LogContext(model=str(MODEL_NAME), stage="boot"),
    registry_dir=str(REGISTRY_DIR),
    weights_root=str(WEIGHTS_ROOT),
    gt_current_root=str(GT_CURRENT_ROOT),
    gt_versions_root=str(GT_VERSIONS_ROOT),
    device_policy="default_fixed",
    fixed_gpu_index=str(FIXED_GPU_INDEX),
    default_imgsz=DEFAULT_IMGSZ,
    default_conf=DEFAULT_CONF,
    default_iou=DEFAULT_IOU,
    ultralytics_yolo_available=(YOLO is not None),
)


# -----------------------------
# Registry helpers
# -----------------------------
def _job_path(train_job_id: str) -> Path:
    return REGISTRY_DIR / f"train_{train_job_id}.json"


def _write_job(train_job_id: str, payload: Dict[str, Any]) -> None:
    _job_path(train_job_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_job(train_job_id: str) -> Dict[str, Any]:
    p = _job_path(train_job_id)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


# -----------------------------
# Weights helpers
# -----------------------------
def _project_model_dir(user_key: str, project_id: str) -> Path:
    return (WEIGHTS_ROOT / "projects" / str(user_key) / str(project_id) / str(MODEL_NAME)).resolve()


def _pick_latest_weight(weights_dir: Path) -> Optional[Path]:
    if not weights_dir.exists():
        return None
    cands = [p for p in weights_dir.rglob("*.pt") if p.is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _pick_latest_project_weight(user_key: str, project_id: str) -> Optional[Path]:
    base = _project_model_dir(user_key, project_id)
    return _pick_latest_weight(base)


def _list_version_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    vers: List[Tuple[int, Path]] = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("v") and len(p.name) == 4 and p.name[1:].isdigit():
            vers.append((int(p.name[1:]), p))
    vers.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in vers]


def _next_project_version(user_key: str, project_id: str) -> int:
    base = _project_model_dir(user_key, project_id)
    vdirs = _list_version_dirs(base)
    if not vdirs:
        return 1
    return int(vdirs[0].name[1:]) + 1


def _base_bdd100k_weight() -> Path:
    base_dir = (WEIGHTS_ROOT / "_base" / str(MODEL_NAME)).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")

    cands = [p for p in base_dir.glob("base_bdd100k*.pt") if p.is_file()]
    if not cands:
        raise FileNotFoundError(f"base_bdd100k*.pt not found under: {base_dir}")

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_init_for_train(req: GTTrainRequest) -> Path:
    if req.baseline_weight_path:
        p = Path(req.baseline_weight_path)
        if not p.is_absolute():
            p = (WEIGHTS_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"baseline_weight_path not found: {p}")
        return p

    user_key = str(req.identity.user_key)
    project_id = str(req.identity.project_id)

    prev = _pick_latest_project_weight(user_key, project_id)
    if prev is not None and prev.exists():
        return prev

    return _base_bdd100k_weight()


def _resolve_weight(requested: Optional[str], weight_dir: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    if requested:
        p = Path(str(requested))
        if not p.is_absolute():
            p = (WEIGHTS_ROOT / p).resolve()
        return (p if p.exists() else None), (str(weight_dir) if weight_dir else None)

    if weight_dir:
        scoped = Path(str(weight_dir))
        if not scoped.is_absolute():
            scoped = (WEIGHTS_ROOT / scoped).resolve()
        w = _pick_latest_weight(scoped)
        return w, str(weight_dir)

    return _pick_latest_weight(WEIGHTS_ROOT), None


# -----------------------------
# GT helpers
# -----------------------------
def _select_dataset_root(req: GTTrainRequest) -> Path:
    if getattr(req, "gt", None) is not None and getattr(req.gt, "gt_version", None):
        return (GT_VERSIONS_ROOT / req.gt.gt_version).resolve()

    if req.dataset is not None and req.dataset.img_root:
        img_root = Path(req.dataset.img_root).resolve()

        if (img_root / "images").exists() or (img_root / "labels").exists() or (img_root / "annotations").exists():
            return img_root

        parent = img_root.parent
        if (parent / "images").exists() or (parent / "labels").exists() or (parent / "annotations").exists():
            return parent

        return img_root

    return GT_CURRENT_ROOT


def _infer_split_name(p: Optional[str], default: str) -> str:
    if not p:
        return default
    s = str(p).replace("\\", "/").rstrip("/")
    parts = s.split("/")
    return parts[-1] if parts else default


def _ensure_ultralytics_data_yaml(dataset_root: Path, *, split_train: str = "train", split_val: str = "val") -> Path:
    data_yaml = dataset_root / "data.yaml"

    raw: Dict[str, Any] = {}
    if data_yaml.exists():
        try:
            loaded = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    names: Any = raw.get("names", None)

    int_keys = sorted([k for k in raw.keys() if isinstance(k, int)])
    if (names is None) and int_keys:
        names = [str(raw[k]) for k in int_keys]
        for k in int_keys:
            raw.pop(k, None)

    if isinstance(names, dict):
        try:
            names = [str(names[k]) for k in sorted(names.keys())]
        except Exception:
            names = None

    env_names = os.getenv("CLASS_NAMES", "").strip()
    if env_names:
        names = [x.strip() for x in env_names.split(",") if x.strip()]

    if not isinstance(names, list) or not names:
        names = ["cls0"]

    env_nc = os.getenv("NC", "").strip()
    if env_nc.isdigit() and int(env_nc) > 0:
        nc = int(env_nc)
        if len(names) > nc:
            names = names[:nc]
        elif len(names) < nc:
            names = names + [f"cls{i}" for i in range(len(names), nc)]
    else:
        nc = len(names)

    train_rel = f"images"
    if (dataset_root / "images").exists():
        val_rel = f"images"
    elif (dataset_root / "images" / "val").exists():
        val_rel = "images"
    elif (dataset_root / "images" / "test").exists():
        val_rel = "images"
    else:
        val_rel = "images"

    payload: Dict[str, Any] = {
        "path": str(dataset_root),
        "train": train_rel,
        "val": val_rel,
        "nc": nc,
        "names": names,
    }

    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    log_event(
        logger,
        "data.yaml ensured",
        ctx=LogContext(stage="train_gt", model=str(MODEL_NAME)),
        data_yaml=str(data_yaml),
        nc=nc,
        has_names=True,
        split_train=split_train,
        split_val=split_val,
        train=train_rel,
        val=val_rel,
    )
    return data_yaml


def _empty_results(req: InferBatchRequest) -> List[InferItemResult]:
    return [InferItemResult(image_id=it.image_id, detections=[]) for it in req.items]


# -----------------------------
# HTTP endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": str(MODEL_NAME),
        "device_policy": "default_fixed",
        "fixed_gpu_index": str(FIXED_GPU_INDEX),
    }


# -----------------------------
# Inference (DEFAULT GPU1)
# -----------------------------
@app.post("/infer")
def infer(req: InferBatchRequest) -> Dict[str, Any]:
    t0 = time.time()

    params = req.params or {}
    run_id = params.get("run_id")
    weight_dir = params.get("weight_dir")
    ctx = LogContext(run_id=str(run_id) if run_id else None, stage="infer", model=str(MODEL_NAME))

    if YOLO is None:
        elapsed_ms = int((time.time() - t0) * 1000)
        log_event(logger, "infer unavailable (no ultralytics YOLO)", ctx=ctx, elapsed_ms=elapsed_ms, level="ERROR")
        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=False,
            error="ultralytics YOLO not available in this container",
            elapsed_ms=elapsed_ms,
            weight_used=None,
            results=_empty_results(req),
            raw={"meta": {"weight_dir_used": str(weight_dir) if weight_dir else None}},
        )
        return resp.model_dump()

    weight_req = params.get("weight") or params.get("weight_path")
    imgsz = int(params.get("imgsz", DEFAULT_IMGSZ))
    conf = float(params.get("conf", DEFAULT_CONF))
    iou = float(params.get("iou", DEFAULT_IOU))

    device = _ultra_device()  # ✅ 항상 GPU1

    log_event(
        logger,
        "infer request",
        ctx=ctx,
        batch_id=req.batch_id,
        n_images=len(req.items),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        device_policy="default_fixed",
        fixed_gpu_index=str(FIXED_GPU_INDEX),
        weight_req=str(weight_req) if weight_req else None,
        weight_dir=str(weight_dir) if weight_dir else None,
    )

    weight_path, weight_dir_used = _resolve_weight(
        str(weight_req) if weight_req else None,
        str(weight_dir) if weight_dir else None,
    )

    if weight_path is None:
        elapsed_ms = int((time.time() - t0) * 1000)
        log_event(
            logger,
            "infer no weight found (empty output)",
            ctx=ctx,
            batch_id=req.batch_id,
            elapsed_ms=elapsed_ms,
            level="WARNING",
            weight_dir_used=weight_dir_used,
        )
        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=True,
            error=None,
            elapsed_ms=elapsed_ms,
            weight_used=None,
            results=_empty_results(req),
            raw={"meta": {"weight_dir_used": weight_dir_used}},
        )
        return resp.model_dump()

    try:
        model = YOLO(str(weight_path))
        image_paths = [it.image_path for it in req.items]

        results_ultra = model.predict(
            source=image_paths,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,  # ✅ GPU1 고정 주입
            verbose=False,
        )

        results: List[InferItemResult] = []
        n = min(len(req.items), len(results_ultra))

        for it, r in zip(req.items[:n], results_ultra[:n]):
            dets: List[Detection] = []
            if getattr(r, "boxes", None) is not None and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    dets.append(Detection(cls=int(k), conf=float(c), xyxy=[float(x1), float(y1), float(x2), float(y2)]))
            results.append(InferItemResult(image_id=it.image_id, detections=dets))

        for it in req.items[n:]:
            results.append(InferItemResult(image_id=it.image_id, detections=[]))

        elapsed_ms = int((time.time() - t0) * 1000)
        log_event(
            logger,
            "infer response",
            ctx=ctx,
            ok=True,
            batch_id=req.batch_id,
            weight_used=str(weight_path),
            elapsed_ms=elapsed_ms,
            n_results=len(results),
            weight_dir_used=weight_dir_used,
            device=device,
            device_policy="default_fixed",
        )

        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=True,
            error=None,
            elapsed_ms=elapsed_ms,
            weight_used=str(weight_path),
            results=results,
            raw={"meta": {"weight_dir_used": weight_dir_used, "device_policy": "default_fixed", "device": device}},
        )
        return resp.model_dump()

    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        logger.exception(
            "infer failed",
            extra=ctx.as_extra()
            | {
                "batch_id": req.batch_id,
                "weight_used": str(weight_path),
                "elapsed_ms": elapsed_ms,
                "error": f"{type(e).__name__}: {e}",
                "weight_dir_used": weight_dir_used,
                "device": device,
                "device_policy": "default_fixed",
            },
        )
        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=elapsed_ms,
            weight_used=str(weight_path),
            results=_empty_results(req),
            raw={"meta": {"weight_dir_used": weight_dir_used, "device_policy": "default_fixed", "device": device}},
        )
        return resp.model_dump()


# -----------------------------
# Train (DEFAULT GPU1)
# -----------------------------
def _do_train(train_job_id: str, req: GTTrainRequest) -> None:
    ident = req.identity
    ctx = LogContext(job_id=ident.job_id, run_id=ident.run_id, stage="train_gt", model=str(MODEL_NAME))

    job = _read_job(train_job_id)
    job["status"] = "RUNNING"
    job["started_at"] = time.time()
    _write_job(train_job_id, job)

    log_event(
        logger,
        "train_gt start",
        ctx=ctx,
        train_job_id=train_job_id,
        user_key=ident.user_key,
        project_id=ident.project_id,
        gt=getattr(req, "gt", None).model_dump() if getattr(req, "gt", None) else None,
        dataset=req.dataset.model_dump() if req.dataset else None,
        train=req.train.model_dump(),
        device_policy="default_fixed",
        fixed_gpu_index=str(FIXED_GPU_INDEX),
    )

    try:
        if YOLO is None:
            raise RuntimeError("ultralytics YOLO not available in this container")
        if not ident.job_id:
            raise ValueError("identity.job_id is required")

        dataset_root = _select_dataset_root(req)
        if not dataset_root.exists():
            raise FileNotFoundError(f"resolved dataset_root not found: {dataset_root}")

        split_train = _infer_split_name(req.dataset.train if req.dataset else None, "train")
        split_val = _infer_split_name(req.dataset.val if req.dataset else None, "val")

        data_yaml = _ensure_ultralytics_data_yaml(dataset_root, split_train=split_train, split_val=split_val)
        init_weight = _resolve_init_for_train(req)

        user_key = str(ident.user_key)
        project_id = str(ident.project_id)
        next_v = _next_project_version(user_key, project_id)
        version_tag = f"v{next_v:03d}"
        out_dir = _project_model_dir(user_key, project_id) / version_tag
        save_path = out_dir / "best.pt"

        train_device = _ultra_device()  # ✅ 항상 GPU1

        job = _read_job(train_job_id)
        job["resolved"] = {
            "dataset_root": str(dataset_root),
            "split_train": split_train,
            "split_val": split_val,
            "gt": getattr(req, "gt", None).model_dump() if getattr(req, "gt", None) else None,
            "init_weight": str(init_weight),
            "data_yaml": str(data_yaml),
            "save_path": str(save_path),
            "version": version_tag,
            "device": train_device,
            "device_policy": "default_fixed",
            "fixed_gpu_index": str(FIXED_GPU_INDEX),
        }
        _write_job(train_job_id, job)

        log_event(
            logger,
            "train_gt resolved",
            ctx=ctx,
            train_job_id=train_job_id,
            dataset_root=str(dataset_root),
            split_train=split_train,
            split_val=split_val,
            init_weight=str(init_weight),
            data_yaml=str(data_yaml),
            save_path=str(save_path),
            version=version_tag,
            device=train_device,
            device_policy="default_fixed",
        )

        model = YOLO(str(init_weight))

        extra = req.train.extra or {}
        train_kwargs: Dict[str, Any] = dict(
            data=str(data_yaml),
            epochs=int(req.train.epochs),
            imgsz=int(req.train.imgsz),
            batch=int(req.train.batch),
            workers=int(extra.get("workers", 8)),
            seed=int(extra.get("seed", 42)),
            amp=bool(extra.get("amp", True)),
            device=train_device,  # ✅ GPU1 고정 주입
            verbose=False,
        )

        lr0 = extra.get("lr0") or extra.get("lr")
        if lr0 is not None:
            train_kwargs["lr0"] = float(lr0)
        wd = extra.get("weight_decay")
        if wd is not None:
            train_kwargs["weight_decay"] = float(wd)

        model.train(**train_kwargs)

        trainer_save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        if not trainer_save_dir:
            raise RuntimeError("trainer.save_dir not found after training")

        best_pt = Path(trainer_save_dir) / "weights" / "best.pt"
        last_pt = Path(trainer_save_dir) / "weights" / "last.pt"
        if not best_pt.exists():
            raise FileNotFoundError("best.pt not found after training")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(best_pt.read_bytes())

        artifacts: Dict[str, Any] = {
            "trained_weight": str(save_path),
            "version": version_tag,
            "best_pt": str(best_pt),
            "last_pt": str(last_pt) if last_pt.exists() else None,
            "init_weight": str(init_weight),
            "data_yaml": str(data_yaml),
            "dataset_root": str(dataset_root),
            "trainer_save_dir": str(trainer_save_dir),
            "device": train_device,
            "device_policy": "default_fixed",
            "fixed_gpu_index": str(FIXED_GPU_INDEX),
        }

        job = _read_job(train_job_id)
        job["status"] = "DONE"
        job["finished_at"] = time.time()
        job["artifacts"] = artifacts
        _write_job(train_job_id, job)

        log_event(
            logger,
            "train_gt done",
            ctx=ctx,
            train_job_id=train_job_id,
            trained_weight=str(save_path),
            version=version_tag,
            device=train_device,
            device_policy="default_fixed",
        )

    except Exception as e:
        job = _read_job(train_job_id)
        job["status"] = "FAIL"
        job["finished_at"] = time.time()
        job["error"] = f"{type(e).__name__}: {e}"
        job["traceback"] = traceback.format_exc()
        _write_job(train_job_id, job)

        logger.exception(
            "train_gt failed",
            extra=ctx.as_extra() | {"train_job_id": train_job_id, "error": f"{type(e).__name__}: {e}"},
        )


@app.post("/train/gt", response_model=GTTrainResponse)
def train_gt(req: GTTrainRequest, background: BackgroundTasks):
    if req.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"model mismatch: got {req.model}, expected {MODEL_NAME}")

    if not req.identity.job_id:
        raise HTTPException(status_code=400, detail="identity.job_id is required")

    if req.dataset is not None:
        if not req.dataset.img_root:
            raise HTTPException(status_code=400, detail="dataset.img_root is required when dataset is provided")
        if not req.dataset.train:
            raise HTTPException(status_code=400, detail="dataset.train is required when dataset is provided")

    train_job_id = f"{MODEL_NAME}_{uuid.uuid4().hex[:10]}"

    record = {
        "train_job_id": train_job_id,
        "identity": req.identity.model_dump(),
        "model": str(MODEL_NAME),
        "status": "QUEUED",
        "request": req.model_dump(),
        "created_at": time.time(),
        "device_policy": "default_fixed",
        "fixed_gpu_index": str(FIXED_GPU_INDEX),
    }
    _write_job(train_job_id, record)

    ctx = LogContext(job_id=req.identity.job_id, run_id=req.identity.run_id, stage="train_gt", model=str(MODEL_NAME))
    log_event(
        logger,
        "train_gt queued",
        ctx=ctx,
        train_job_id=train_job_id,
        device_policy="default_fixed",
        fixed_gpu_index=str(FIXED_GPU_INDEX),
    )

    background.add_task(_do_train, train_job_id, req)

    return GTTrainResponse(
        identity=req.identity,
        model=req.model,
        status="STARTED",
        detail="queued",
        artifacts={"train_job_id": train_job_id, "registry_path": str(_job_path(train_job_id))},
    )


@app.get("/train/status/{train_job_id}")
def train_status(train_job_id: str):
    ctx = LogContext(stage="train_gt_status", model=str(MODEL_NAME))

    try:
        job = _read_job(train_job_id)
    except FileNotFoundError:
        log_event(logger, "train status not found", ctx=ctx, train_job_id=train_job_id, level="WARNING")
        raise HTTPException(status_code=404, detail="train job not found")

    log_event(logger, "train status fetched", ctx=ctx, train_job_id=train_job_id, status=job.get("status"))
    return {"ok": True, "model": str(MODEL_NAME), "train_job": job}
