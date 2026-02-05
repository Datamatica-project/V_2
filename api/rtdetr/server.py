from __future__ import annotations

import os
import json
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from fastapi import FastAPI, BackgroundTasks, HTTPException

# ✅ logging import 통일 (SSOT 경로)
from src.common.utils.logging import setup_logger, LogContext, log_event

# ✅ DTO SSOT import (중복 정의 금지)
from src.common.dto.dto import (
    InferBatchRequest,
    ModelName,
    Detection,
    InferItemResult,
    ModelPredResponse,
    GTTrainRequest,
    GTTrainResponse,
)

# -----------------------------
# Ultralytics RT-DETR
# -----------------------------
try:
    from ultralytics import RTDETR  # type: ignore
except Exception:  # pragma: no cover
    RTDETR = None  # type: ignore


MODEL_NAME: ModelName = "rtdetr"

DEFAULT_REGISTRY_ROOT = os.getenv("MODEL_REGISTRY_ROOT", "/workspace/logs/model_train")
REGISTRY_DIR = Path(DEFAULT_REGISTRY_ROOT) / str(MODEL_NAME)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_ROOT = Path(os.getenv("WEIGHTS_ROOT", "/weights"))
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_DEVICE = os.getenv("DEVICE", "0")  # "0" / "cuda:0" / "cpu"
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "640"))
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))

logger = setup_logger(
    service=os.getenv("MODEL_NAME", str(MODEL_NAME)),
    log_file=os.getenv("MODEL_LOG_FILE", f"/workspace/logs/{MODEL_NAME}/api.jsonl"),
    level=os.getenv("LOG_LEVEL", "INFO"),
)

app = FastAPI(title=f"{MODEL_NAME}-api", version="0.4.0")

log_event(
    logger,
    "service boot",
    ctx=LogContext(model=str(MODEL_NAME), stage="boot"),
    registry_dir=str(REGISTRY_DIR),
    weights_root=str(WEIGHTS_ROOT),
    device=DEFAULT_DEVICE,
    default_imgsz=DEFAULT_IMGSZ,
    default_conf=DEFAULT_CONF,
    ultralytics_rtdetr_available=(RTDETR is not None),
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


def _pick_latest_weight(weights_dir: Path) -> Optional[Path]:
    if not weights_dir.exists():
        return None
    cands = [p for p in weights_dir.rglob("*.pt") if p.is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_weight(requested: Optional[str], weight_dir: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    """
    infer용 weight resolver (정책 반영)

    우선순위:
      1) requested (weight/weight_path) 지정 시: 그 경로 사용 (상대면 WEIGHTS_ROOT 기준)
      2) weight_dir 지정 시: WEIGHTS_ROOT/weight_dir 범위에서만 latest pick
      3) (호환 fallback) 없으면 WEIGHTS_ROOT 전체 latest

    returns:
      (weight_path, weight_dir_used)
    """
    # 1) 명시 weight
    if requested:
        p = Path(str(requested))
        if not p.is_absolute():
            p = (WEIGHTS_ROOT / p).resolve()
        return (p if p.exists() else None), (str(weight_dir) if weight_dir else None)

    # 2) 스코프 weight_dir
    if weight_dir:
        scoped = Path(str(weight_dir))
        if not scoped.is_absolute():
            scoped = (WEIGHTS_ROOT / scoped).resolve()
        w = _pick_latest_weight(scoped)
        return w, str(weight_dir)

    # 3) fallback
    return _pick_latest_weight(WEIGHTS_ROOT), None


def _project_model_dir(user_key: str, project_id: str) -> Path:
    return (WEIGHTS_ROOT / "projects" / str(user_key) / str(project_id) / str(MODEL_NAME)).resolve()


def _pick_latest_project_weight(user_key: str, project_id: str) -> Optional[Path]:
    """
    projects/{user}/{project}/{model}/ 아래에서 최신 .pt를 찾음 (v### 하위 포함)
    """
    base = _project_model_dir(user_key, project_id)
    return _pick_latest_weight(base)


def _next_project_version(user_key: str, project_id: str) -> int:
    """
    projects/{user}/{project}/{model}/v### 폴더 기준으로 다음 버전 계산
    v001부터 시작
    """
    base = _project_model_dir(user_key, project_id)
    if not base.exists():
        return 1

    max_v = 0
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("v") and len(p.name) == 4 and p.name[1:].isdigit():
            max_v = max(max_v, int(p.name[1:]))
    return max_v + 1


def _base_bdd100k_weight() -> Path:
    """
    v1 init: /weights/_base/{model}/base_bdd100k.*
    """
    base_dir = (WEIGHTS_ROOT / "_base" / str(MODEL_NAME)).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")

    cands: List[Path] = []
    for ext in ("pt", "pth", "ckpt"):
        cands += list(base_dir.glob(f"base_bdd100k*.{ext}"))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        raise FileNotFoundError(f"base_bdd100k.* not found under: {base_dir}")

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


# -----------------------------
# Dataset YAML auto writer (optional)
# -----------------------------
def _ensure_ultralytics_data_yaml(dataset_root: Path) -> Path:
    """
    ultralytics는 data.yaml이 필요하므로,
    dataset_root 아래 data.yaml 없으면 자동 생성해줌 (MVP 편의)

    - 여기서는 req.dataset.img_root를 "dataset root"로 해석한다.
      (ultralytics data.yaml의 path 기준)
    """
    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    nc = int(os.getenv("NC", "0"))
    names = os.getenv("CLASS_NAMES", "").strip()

    payload: Dict[str, Any] = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val" if (dataset_root / "images" / "val").exists() else "images/test",
    }
    if nc > 0:
        payload["nc"] = nc
    if names:
        payload["names"] = [x.strip() for x in names.split(",") if x.strip()]

    lines: List[str] = []
    for k, v in payload.items():
        if isinstance(v, list):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {v}")

    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def _empty_results(req: InferBatchRequest) -> List[InferItemResult]:
    return [InferItemResult(image_id=it.image_id, detections=[]) for it in req.items]


# -----------------------------
# Core: Inference
# -----------------------------
@app.post("/infer")
def infer(req: InferBatchRequest) -> Dict[str, Any]:
    t0 = time.time()

    params = req.params or {}
    run_id = params.get("run_id")
    ctx = LogContext(run_id=str(run_id) if run_id else None, stage="infer", model=str(MODEL_NAME))

    weight_req = params.get("weight") or params.get("weight_path")
    weight_dir = params.get("weight_dir")  # ✅ scope
    imgsz = int(params.get("imgsz", DEFAULT_IMGSZ))
    conf = float(params.get("conf", DEFAULT_CONF))
    device = str(params.get("device", DEFAULT_DEVICE))

    log_event(
        logger,
        "infer request",
        ctx=ctx,
        batch_id=req.batch_id,
        n_images=len(req.items),
        imgsz=imgsz,
        conf=conf,
        device=device,
        weight_req=str(weight_req) if weight_req else None,
        weight_dir=str(weight_dir) if weight_dir else None,
    )

    if RTDETR is None:
        elapsed_ms = int((time.time() - t0) * 1000)
        log_event(
            logger,
            "infer unavailable (no ultralytics RTDETR)",
            ctx=ctx,
            elapsed_ms=elapsed_ms,
            level="ERROR",
        )
        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=False,
            error="ultralytics RTDETR not available in this container",
            elapsed_ms=elapsed_ms,
            weight_used=None,
            results=_empty_results(req),
            raw={"meta": {"weight_dir_used": str(weight_dir) if weight_dir else None}},
        )
        return resp.model_dump()

    weight_path, weight_dir_used = _resolve_weight(
        str(weight_req) if weight_req else None,
        str(weight_dir) if weight_dir else None,
    )

    # weight가 없으면 빈 결과 + ok=True (기존 동작 유지)
    if weight_path is None:
        elapsed_ms = int((time.time() - t0) * 1000)
        log_event(
            logger,
            "infer no weight found (empty output)",
            ctx=ctx,
            batch_id=req.batch_id,
            elapsed_ms=elapsed_ms,
            level="WARNING",
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
        model = RTDETR(str(weight_path))

        image_paths = [it.image_path for it in req.items]

        results_ultra = model.predict(
            source=image_paths,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

        if len(results_ultra) != len(req.items):
            log_event(
                logger,
                "infer results length mismatch",
                ctx=ctx,
                batch_id=req.batch_id,
                expected=len(req.items),
                got=len(results_ultra),
                level="WARNING",
            )

        results: List[InferItemResult] = []
        for it, r in zip(req.items, results_ultra):
            dets: List[Detection] = []
            if getattr(r, "boxes", None) is not None and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    dets.append(
                        Detection(
                            cls=int(k),
                            conf=float(c),
                            xyxy=[float(x1), float(y1), float(x2), float(y2)],
                        )
                    )
            results.append(InferItemResult(image_id=it.image_id, detections=dets))

        # 결과 누락 보정
        if len(results) < len(req.items):
            done_ids = {r.image_id for r in results}
            for it in req.items:
                if it.image_id not in done_ids:
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
        )

        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=True,
            error=None,
            elapsed_ms=elapsed_ms,
            weight_used=str(weight_path),
            results=results,
            raw={
                "meta": {
                    "weight_dir_used": weight_dir_used,
                },
                "ultralytics": {
                    "n_images": len(image_paths),
                    "imgsz": imgsz,
                    "conf": conf,
                    "device": device,
                },
            },
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
            },
        )

        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=elapsed_ms,
            weight_used=str(weight_path) if weight_path else None,
            results=_empty_results(req),
            raw={"meta": {"weight_dir_used": weight_dir_used}},
        )
        return resp.model_dump()


# -----------------------------
# Core: Train (policy-aligned)
# -----------------------------
def _resolve_init_for_train(req: GTTrainRequest) -> Path:
    """
    정책:
      - req.baseline_weight_path 있으면 그걸 사용 (존재해야 함)
      - 없으면:
          - projects/{user}/{project}/{model}/ 에 기존 버전이 있으면 "직전 버전(latest)"에서 시작 (v2~)
          - 없으면: /weights/_base/{model}/base_bdd100k.* 에서 시작 (v1)
    """
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


def _do_train(train_job_id: str, req: GTTrainRequest) -> None:
    identity = req.identity
    ctx = LogContext(job_id=identity.job_id, run_id=identity.run_id, stage="train_gt", model=str(MODEL_NAME))

    job = _read_job(train_job_id)
    job["status"] = "RUNNING"
    job["started_at"] = time.time()
    _write_job(train_job_id, job)

    log_event(
        logger,
        "train_gt start",
        ctx=ctx,
        train_job_id=train_job_id,
        user_key=identity.user_key,
        project_id=identity.project_id,
        dataset=req.dataset.model_dump(),
        train=req.train.model_dump(),
    )

    try:
        if RTDETR is None:
            raise RuntimeError("ultralytics RTDETR not available in this container")

        if not identity.job_id:
            raise ValueError("identity.job_id is required for training")

        dataset_root = Path(req.dataset.img_root).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"dataset.img_root not found: {dataset_root}")

        data_yaml = _ensure_ultralytics_data_yaml(dataset_root)

        # ✅ init weight: v1=base_bdd100k, v2~=prev
        init_weight = _resolve_init_for_train(req)

        # ✅ output path: weights/projects/{user}/{project}/{model}/v###/best.pt
        user_key = str(identity.user_key)
        project_id = str(identity.project_id)
        next_v = _next_project_version(user_key, project_id)
        version_tag = f"v{next_v:03d}"
        out_dir = _project_model_dir(user_key, project_id) / version_tag
        save_path = out_dir / "best.pt"

        # Registry에 미리 기록(실패해도 target 추적 가능)
        job = _read_job(train_job_id)
        job["resolved"] = {
            "init_weight": str(init_weight),
            "data_yaml": str(data_yaml),
            "save_path": str(save_path),
            "version": version_tag,
        }
        _write_job(train_job_id, job)

        log_event(
            logger,
            "train_gt resolved",
            ctx=ctx,
            train_job_id=train_job_id,
            init_weight=str(init_weight),
            data_yaml=str(data_yaml),
            save_path=str(save_path),
            version=version_tag,
        )

        model = RTDETR(str(init_weight))

        model.train(
            data=str(data_yaml),
            epochs=req.train.epochs,
            imgsz=req.train.imgsz,
            batch=req.train.batch,
            workers=int(req.train.extra.get("workers", 8)),
            seed=int(req.train.extra.get("seed", 42)),
            amp=bool(req.train.extra.get("amp", True)),
            device=str(req.train.device or DEFAULT_DEVICE),
            verbose=False,
        )

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
            "trainer_save_dir": str(trainer_save_dir),
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


# -----------------------------
# HTTP endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "model": str(MODEL_NAME)}


@app.post("/train/gt", response_model=GTTrainResponse)
def train_gt(req: GTTrainRequest, background: BackgroundTasks):
    """
    정책 준수:
    - 모델 mismatch 차단
    - init은 서버 정책으로 결정 (req.baseline_weight_path가 있으면 그걸 최우선)
      - v1: /weights/_base/{model}/base_bdd100k.*
      - v2~: projects/{user}/{project}/{model} 직전 버전
    - 결과는 /weights/projects/{user}/{project}/{model}/v###/best.pt 로 저장
    """
    if req.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"model mismatch: got {req.model}, expected {MODEL_NAME}")

    if not req.identity.job_id:
        raise HTTPException(status_code=400, detail="identity.job_id is required")

    if not req.dataset.img_root:
        raise HTTPException(status_code=400, detail="dataset.img_root is required")
    if not req.dataset.train:
        raise HTTPException(status_code=400, detail="dataset.train is required")

    train_job_id = f"{MODEL_NAME}_{uuid.uuid4().hex[:10]}"

    record = {
        "train_job_id": train_job_id,
        "identity": req.identity.model_dump(),
        "model": str(MODEL_NAME),
        "status": "QUEUED",
        "request": req.model_dump(),
        "created_at": time.time(),
    }
    _write_job(train_job_id, record)

    ctx = LogContext(job_id=req.identity.job_id, run_id=req.identity.run_id, stage="train_gt", model=str(MODEL_NAME))
    log_event(logger, "train_gt queued", ctx=ctx, train_job_id=train_job_id)

    background.add_task(_do_train, train_job_id, req)

    return GTTrainResponse(
        identity=req.identity,
        model=req.model,
        status="STARTED",
        detail="queued",
        artifacts={
            "train_job_id": train_job_id,
            "registry_path": str(_job_path(train_job_id)),
        },
    )


@app.get("/train/status/{train_job_id}")
def train_status(train_job_id: str):
    """
    폴링용 상태 조회
    - DONE이면 artifacts.trained_weight에 정책 파일(best.pt)이 들어있음
    """
    ctx = LogContext(stage="train_gt_status", model=str(MODEL_NAME))

    try:
        job = _read_job(train_job_id)
    except FileNotFoundError:
        log_event(logger, "train status not found", ctx=ctx, train_job_id=train_job_id, level="WARNING")
        raise HTTPException(status_code=404, detail="train job not found")

    log_event(logger, "train status fetched", ctx=ctx, train_job_id=train_job_id, status=job.get("status"))
    return {"ok": True, "model": str(MODEL_NAME), "train_job": job}
