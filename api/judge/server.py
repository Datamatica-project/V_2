from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal

import yaml
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Body, Path as FPath, Query
from pydantic import BaseModel, Field

from src.judge.orchestrator.run_batch import run_batch
from src.judge.orchestrator.status import read_status
from src.common.utils.logging import setup_logger, LogContext, log_event
from src.common.dto.dto import GTTrainRequest, GTTrainResponse

from api.routers.gt_register import router as gt_router
from api.routers.unlabeled_register import router as unlabeled_router


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).resolve()


CONFIGS_DIR = _env_path("CONFIGS_DIR", "/workspace/configs")
FALLBACK_CONFIGS_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw) or {}
    except Exception as e:
        raise ValueError(f"invalid yaml: {path} ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"yaml root must be a mapping(dict): {path}")
    return data


# -----------------------------------------------------------------------------
# models.yaml loader
# -----------------------------------------------------------------------------
def _pick_models_yaml_path() -> Path:
    p = CONFIGS_DIR / "models.yaml"
    if p.exists():
        return p
    return FALLBACK_CONFIGS_DIR / "models.yaml"


def _load_models_cfg_with_path() -> Tuple[Dict[str, Any], Path]:
    p = _pick_models_yaml_path()
    return _load_yaml(p), p


def _validate_models_cfg(models_cfg: Dict[str, Any]) -> None:
    if "models" not in models_cfg:
        raise ValueError("models.yaml missing 'models'")
    for name in ("yolov11", "rtm", "rtdetr"):
        m = models_cfg["models"].get(name)
        if not isinstance(m, dict):
            raise ValueError(f"models.yaml missing models.{name}")
        if not m.get("base_url"):
            raise ValueError(f"models.yaml missing models.{name}.base_url")


def _build_gateway_model_cfg(models_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"models": {}}
    for name in ("yolov11", "rtm", "rtdetr"):
        m = models_cfg["models"][name]
        base_url = m["base_url"].rstrip("/")
        infer_ep = m.get("endpoints", {}).get("infer", "/infer")
        out["models"][name] = {"url": f"{base_url}{infer_ep}"}
    return out


# -----------------------------------------------------------------------------
# ensemble.yaml loader
# -----------------------------------------------------------------------------
def _pick_ensemble_yaml_path() -> Path:
    p = CONFIGS_DIR / "ensemble.yaml"
    if p.exists():
        return p
    return FALLBACK_CONFIGS_DIR / "ensemble.yaml"


def _load_ensemble_cfg_with_path() -> Tuple[Dict[str, Any], Path]:
    p = _pick_ensemble_yaml_path()
    return _load_yaml(p), p


# -----------------------------------------------------------------------------
# train_gt.yaml loader
# -----------------------------------------------------------------------------
def _pick_train_gt_yaml_path() -> Path:
    p = CONFIGS_DIR / "train_gt.yaml"
    if p.exists():
        return p
    return FALLBACK_CONFIGS_DIR / "train_gt.yaml"


def _load_train_gt_cfg_with_path() -> Tuple[Dict[str, Any], Path]:
    p = _pick_train_gt_yaml_path()
    return _load_yaml(p), p


def _train_timeout_sec(cfg: Dict[str, Any]) -> float:
    t = (cfg.get("timeouts_sec", {}) or {}).get("train_gt", None)
    try:
        if t is None:
            return 120.0
        return float(t)
    except Exception:
        return 120.0


# -----------------------------------------------------------------------------
# Infer DTO
# -----------------------------------------------------------------------------
class InferRunRequest(BaseModel):
    run_id: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    project_id: Optional[str] = Field(default=None)

    unlabeled_dir: str = Field(..., description="라벨 없는 이미지 디렉토리(컨테이너 내부 경로)")
    batch_size: int = Field(8, ge=1, le=4096)
    segment_size_batches: int = Field(100, ge=1, le=100000)
    infer_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class InferRunResponse(BaseModel):
    ok: bool = True
    run_id: str
    status: str


# -----------------------------------------------------------------------------
# Train DTO (identity + GT ref)
# -----------------------------------------------------------------------------
GtRefType = Literal["current", "version"]


class TrainGtRef(BaseModel):
    kind: GtRefType = Field(default="current", description="current | version")
    gt_version: Optional[str] = Field(default=None, description="kind=version일 때 예: GT_20260210")

    model_config = {"extra": "ignore"}


class TrainAllRequest(BaseModel):
    identity: Dict[str, Any] = Field(
        ...,
        description="최소: user_key, project_id, job_id",
        examples=[{"user_key": "user_001", "project_id": "project_demo", "job_id": "job_all_001"}],
    )
    gt: TrainGtRef = Field(default_factory=TrainGtRef, description="GT 참조 정보")

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# App & Logger
# -----------------------------------------------------------------------------
app = FastAPI(
    title="V2 Judge (Control Plane)",
    version="0.4.2",
    openapi_tags=[
        {"name": "Health", "description": "서버 상태 확인(헬스체크)"},
        {"name": "Loop: Inference", "description": "3모델 추론 fan-out + 앙상블 루프"},
        {"name": "Loop: Train GT", "description": "GT 학습 트리거/상태조회 오케스트레이션"},
        {"name": "Debug", "description": "설정 파일 로딩 상태 확인"},
    ],
)

app.include_router(gt_router)
app.include_router(unlabeled_router)

logger = setup_logger(
    service="judge",
    log_file=os.getenv("JUDGE_API_LOG_FILE", "/workspace/logs/judge/api.jsonl"),
    level=os.getenv("LOG_LEVEL", "INFO"),
)


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    return {"ok": True}


# -----------------------------------------------------------------------------
# Debug
# -----------------------------------------------------------------------------
@app.get("/debug/config/models", tags=["Debug"])
def debug_models_config():
    cfg, path = _load_models_cfg_with_path()
    _validate_models_cfg(cfg)
    return {
        "ok": True,
        "models_yaml_path": str(path),
        "configs_dir": str(CONFIGS_DIR),
        "fallback_dir": str(FALLBACK_CONFIGS_DIR),
        "gateway_model_cfg": _build_gateway_model_cfg(cfg),
    }


@app.get("/debug/config/ensemble", tags=["Debug"])
def debug_ensemble_config():
    cfg, path = _load_ensemble_cfg_with_path()
    return {
        "ok": True,
        "ensemble_yaml_path": str(path),
        "configs_dir": str(CONFIGS_DIR),
        "fallback_dir": str(FALLBACK_CONFIGS_DIR),
        "ensemble_cfg": cfg,
    }


@app.get("/debug/config/train_gt", tags=["Debug"])
def debug_train_gt_config():
    cfg, path = _load_train_gt_cfg_with_path()
    return {
        "ok": True,
        "train_gt_yaml_path": str(path),
        "configs_dir": str(CONFIGS_DIR),
        "fallback_dir": str(FALLBACK_CONFIGS_DIR),
        "train_gt_cfg": cfg,
        "resolved_timeouts_sec": {"train_gt": _train_timeout_sec(cfg)},
    }


# -----------------------------------------------------------------------------
# Infer loop (unchanged)
# -----------------------------------------------------------------------------
def _merge_infer_params(req: InferRunRequest) -> Dict[str, Any]:
    p = dict(req.infer_params or {})
    if req.run_id:
        p.setdefault("run_id", req.run_id)
    if req.user_id:
        p.setdefault("user_id", req.user_id)
    if req.project_id:
        p.setdefault("project_id", req.project_id)
    return p


def _do_infer(req: InferRunRequest) -> None:
    ctx = LogContext(run_id=req.run_id, stage="infer", model="judge")

    models_cfg, _ = _load_models_cfg_with_path()
    model_cfg = _build_gateway_model_cfg(models_cfg)

    ensemble_cfg, _ = _load_ensemble_cfg_with_path()

    log_event(
        logger,
        "loop infer background task start",
        ctx=ctx,
        unlabeled_dir=req.unlabeled_dir,
        batch_size=req.batch_size,
        segment_size_batches=req.segment_size_batches,
        infer_params_keys=sorted(list((_merge_infer_params(req) or {}).keys())),
    )

    run_batch(
        run_id=req.run_id,
        unlabeled_dir=req.unlabeled_dir,
        batch_size=req.batch_size,
        model_cfg=model_cfg,
        infer_params=_merge_infer_params(req),
        segment_size_batches=req.segment_size_batches,
        ensemble_cfg=ensemble_cfg,
    )

    log_event(logger, "loop infer background task done", ctx=ctx)


@app.post("/loop/infer/run", response_model=InferRunResponse, tags=["Loop: Inference"])
def loop_infer_run(
    req: InferRunRequest = Body(...),
    background: BackgroundTasks = None,
):
    run_id = req.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    req = req.model_copy(update={"run_id": run_id})

    if background is None:
        raise HTTPException(status_code=500, detail="BackgroundTasks injection failed")

    background.add_task(_do_infer, req)
    return InferRunResponse(run_id=run_id, status="RUNNING")


@app.get("/loop/infer/status/{run_id}", tags=["Loop: Inference"])
def loop_infer_status(run_id: str = FPath(...)):
    return read_status(run_id)


# -----------------------------------------------------------------------------
# Train – Judge Orchestration
# -----------------------------------------------------------------------------
def _call_train(*, model: str, req: GTTrainRequest, timeout_s: float) -> Dict[str, Any]:
    cfg, _ = _load_models_cfg_with_path()
    m = cfg["models"].get(model)
    if not m:
        raise HTTPException(status_code=400, detail=f"unknown model: {model}")

    base = m["base_url"].rstrip("/")
    ep = m.get("endpoints", {}).get("train_gt", "/train/gt")
    url = f"{base}{ep}"

    r = requests.post(url, json=req.model_dump(by_alias=True), timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _resolve_gt_root_from_cfg(*, cfg: Dict[str, Any], gt: TrainGtRef) -> str:
    data = cfg.get("data", {}) or {}
    local = (data.get("local", {}) or {})

    current_root = str(local.get("current_root", "/workspace/storage/GT/current"))
    versions_root = str(local.get("versions_root", "/workspace/storage/GT_versions"))

    if gt.kind == "current":
        return current_root

    if gt.kind == "version":
        if not gt.gt_version:
            raise HTTPException(status_code=422, detail="gt.gt_version is required when gt.kind='version'")
        return str(Path(versions_root) / gt.gt_version)

    raise HTTPException(status_code=422, detail=f"invalid gt.kind: {gt.kind}")


def _gt_payload_for_model(model: str, gt: TrainGtRef) -> Optional[Dict[str, Any]]:
    if gt.kind == "version" and gt.gt_version:
        return {"gt_version": str(gt.gt_version)}
    return None


def _build_dataset_from_layout(*, model: str, gt_root: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = cfg.get("data", {}) or {}
    splits = (data.get("splits", {}) or {})
    split_train = str(splits.get("train", "train"))
    split_val = str(splits.get("val", "val"))

    # models.yaml에서 layout 힌트 읽기(없으면 기본값)
    models_cfg, _ = _load_models_cfg_with_path()
    m = (models_cfg.get("models", {}) or {}).get(model, {}) or {}
    label_source = str(m.get("label_source", "yolo")).lower()
    layout = (m.get("data_layout", {}) or {})
    images_dir = str(layout.get("images_dir") or "images")
    labels_dir = str(layout.get("labels_dir") or "labels")
    ann_dir = str(layout.get("annotations_dir") or "annotations")

    # RTM은 네 설정에서 label_source=coco, annotations_dir=annotations
    if model == "rtm" or label_source == "coco":
        return {
            "format": "coco",
            "img_root": gt_root,
            "train": f"{ann_dir}/{split_train}.json",
            "val": f"{ann_dir}/{split_val}.json",
        }

    # 그 외는 yolo로 간주
    return {
        "format": "yolo",
        "img_root": gt_root,
        "train": f"{labels_dir}/{split_train}",
        "val": f"{labels_dir}/{split_val}",
    }


def _build_gt_train_req_from_cfg(
    *,
    model: str,
    ident: Dict[str, Any],
    gt: TrainGtRef,
    cfg: Dict[str, Any],
) -> GTTrainRequest:
    train = cfg.get("train", {}) or {}
    gt_root = _resolve_gt_root_from_cfg(cfg=cfg, gt=gt)

    dataset = _build_dataset_from_layout(model=model, gt_root=gt_root, cfg=cfg)

    train_params: Dict[str, Any] = {
        "epochs": int(train.get("epochs", 10)),
        "imgsz": int(train.get("imgsz", 640)),
        "batch": int(train.get("batch", 16)),
        "device": str(train.get("device", "0")),
        "extra": {
            "lr0": float(train.get("lr", 0.001)),
            "weight_decay": float(train.get("weight_decay", 0.0005)),
            "workers": int(train.get("workers", 8)),
            "seed": int(train.get("seed", 42)),
            "amp": bool(train.get("amp", True)),
        },
    }

    # rtm config inject (train_gt.yaml or env)
    if model == "rtm":
        rtm_block = cfg.get("rtm", {}) or {}
        rtm_config = rtm_block.get("config")
        if rtm_config:
            train_params["extra"]["config"] = str(rtm_config)

    payload = {
        "identity": ident,
        "model": model,
        "gt": _gt_payload_for_model(model, gt),
        "dataset": dataset,
        "train": train_params,
        "init_weight_type": "baseline",
    }
    return GTTrainRequest(**payload)


@app.post(
    "/loop/train/gt/run/{model}",
    response_model=GTTrainResponse,
    tags=["Loop: Train GT"],
    summary="GT 학습 실행 (단일 모델, identity + gt_ref)",
)
def loop_train_gt_run(
    model: str = FPath(..., description="yolov11 | rtm | rtdetr"),
    req: TrainAllRequest = Body(...),
):
    if model not in ("yolov11", "rtm", "rtdetr"):
        raise HTTPException(status_code=400, detail=f"unknown model: {model}")

    cfg, _ = _load_train_gt_cfg_with_path()
    timeout_s = _train_timeout_sec(cfg)

    gt_req = _build_gt_train_req_from_cfg(model=model, ident=req.identity, gt=req.gt, cfg=cfg)

    log_event(
        logger,
        "loop train_gt dispatch",
        ctx=LogContext(job_id=req.identity.get("job_id"), stage="train_gt_dispatch", model="judge"),
        target_model=model,
        timeout_s=timeout_s,
        gt_kind=req.gt.kind,
        gt_version=req.gt.gt_version,
        payload_has_gt=(gt_req.gt is not None),
        payload_dataset=gt_req.dataset.model_dump() if gt_req.dataset else None,
    )

    return _call_train(model=model, req=gt_req, timeout_s=timeout_s)


@app.post(
    "/loop/train/gt/run_all",
    tags=["Loop: Train GT"],
    summary="GT 학습 실행 (3모델 전체, identity + gt_ref)",
)
def loop_train_gt_run_all(req: TrainAllRequest = Body(...)):
    cfg, cfg_path = _load_train_gt_cfg_with_path()
    timeout_s = _train_timeout_sec(cfg)

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for model in ("yolov11", "rtm", "rtdetr"):
        try:
            gt_req = _build_gt_train_req_from_cfg(model=model, ident=req.identity, gt=req.gt, cfg=cfg)
            results[model] = _call_train(model=model, req=gt_req, timeout_s=timeout_s)
        except Exception as e:
            errors[model] = f"{type(e).__name__}: {e}"

    out: Dict[str, Any] = {
        "ok": not bool(errors),
        "train_gt_yaml_path": str(cfg_path),
        "timeout_s": timeout_s,
        "gt": {"kind": req.gt.kind, "gt_version": req.gt.gt_version},
        "results": results,
    }
    if errors:
        out["errors"] = errors
    return out


@app.get(
    "/loop/train/status/{model}/{train_job_id}",
    tags=["Loop: Train GT"],
    summary="GT 학습 상태 조회",
)
def loop_train_gt_status(
    model: str = FPath(..., description="yolov11 | rtm | rtdetr"),
    train_job_id: str = FPath(..., description="모델 컨테이너 학습 job id"),
    timeout_s: float = Query(5.0, description="상태 조회 timeout(초)"),
):
    cfg, _ = _load_models_cfg_with_path()
    m = cfg["models"].get(model)
    if not m:
        raise HTTPException(status_code=400, detail=f"unknown model: {model}")

    base = m["base_url"].rstrip("/")
    ep = m.get("endpoints", {}).get("train_status", "/train/status")
    url = f"{base}{ep}/{train_job_id}"

    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()
