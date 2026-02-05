from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
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
# Infer DTO
# -----------------------------------------------------------------------------
class InferRunRequest(BaseModel):
    run_id: Optional[str] = Field(
        default=None,
        description="(선택) 외부에서 지정하는 run_id. 미지정 시 서버가 run_YYYYmmdd_HHMMSS 형태로 자동 생성",
        examples=["run_20260205_141500"],
    )

    user_id: Optional[str] = Field(
        default=None,
        description=(
            "(선택) 프로젝트 스코프용 사용자 키. "
            "제공 시 모델 컨테이너에서 projects/{user}/{project}/{model} weight 스코프 선택 등에 사용 가능"
        ),
        examples=["user_001"],
    )
    project_id: Optional[str] = Field(
        default=None,
        description="(선택) 프로젝트 스코프용 프로젝트 키",
        examples=["project_demo"],
    )

    unlabeled_dir: str = Field(
        ...,
        description="라벨 없는 이미지 디렉토리(컨테이너 내부 경로). 이 경로를 스캔해 추론 대상 이미지를 수집",
        examples=["/workspace/storage/datasets/unlabeled/images"],
    )

    batch_size: int = Field(
        8,
        ge=1,
        le=4096,
        description="배치당 이미지 개수. 너무 크면 VRAM OOM/timeout/네트워크 병목 가능",
        examples=[8, 16],
    )

    segment_size_batches: int = Field(
        100,
        ge=1,
        le=100000,
        description="결과를 segment 단위로 저장할 때 segment에 포함될 배치 수(로그/저장 파일 분할용)",
        examples=[100],
    )

    infer_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "모델 infer API로 전달할 추가 파라미터.\n"
            "- 공통 예: imgsz, conf, device\n"
            "- YOLO 전용 예: iou\n"
            "- (선택) weight/weight_path, weight_dir(스코프)\n"
            "※ run_id/user_id/project_id는 서버가 infer_params에 자동 병합(setdefault)될 수 있음"
        ),
        examples=[{"conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "0"}],
    )

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "examples": [
                {
                    "run_id": "run_20260205_141500",
                    "user_id": "user_001",
                    "project_id": "project_demo",
                    "unlabeled_dir": "/workspace/storage/datasets/unlabeled/images",
                    "batch_size": 8,
                    "segment_size_batches": 100,
                    "infer_params": {"conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "0"},
                }
            ]
        },
    }


class InferRunResponse(BaseModel):
    ok: bool = True
    run_id: str
    status: str


# -----------------------------------------------------------------------------
# App & Logger (Swagger 문서화 강화)
# -----------------------------------------------------------------------------
app = FastAPI(
    title="V2 Judge (Control Plane)",
    version="0.3.0",
    openapi_tags=[
        {"name": "Health", "description": "서버 상태 확인(헬스체크)"},
        {
            "name": "Loop: Inference",
            "description": (
                "Unlabeled 이미지에 대해 배치 추론을 수행하고(3모델 fan-out), "
                "Judge에서 앙상블(PASS_3/PASS_2/FAIL/MISS) 결과를 생성하는 비동기 루프"
            ),
        },
        {
            "name": "Loop: Train GT",
            "description": (
                "GT 학습을 모델 컨테이너로 위임하여 트리거/상태조회 하는 오케스트레이션 API. "
                "단일 모델 실행 또는 3모델 전체 실행을 지원"
            ),
        },
        {"name": "Debug", "description": "설정 파일(models.yaml / ensemble.yaml) 로딩 상태 확인"},
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
@app.get(
    "/health",
    tags=["Health"],
    summary="헬스체크",
    description="Judge(Control Plane) 서비스가 정상 동작 중인지 확인합니다.",
    responses={200: {"description": "정상 동작"}},
)
def health():
    return {"ok": True}


# -----------------------------------------------------------------------------
# (선택) Debug: configs 확인 (Swagger에 도움됨)
# -----------------------------------------------------------------------------
@app.get(
    "/debug/config/models",
    tags=["Debug"],
    summary="models.yaml 로딩/정규화 결과 확인",
    description=(
        "Judge가 실제로 사용 중인 models.yaml 경로와, "
        "모델별 infer URL로 정규화된 설정을 확인합니다."
    ),
)
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


@app.get(
    "/debug/config/ensemble",
    tags=["Debug"],
    summary="ensemble.yaml 로딩 결과 확인",
    description="Judge가 실제로 사용 중인 ensemble.yaml 경로와 로딩된 설정을 반환합니다.",
)
def debug_ensemble_config():
    cfg, path = _load_ensemble_cfg_with_path()
    return {
        "ok": True,
        "ensemble_yaml_path": str(path),
        "configs_dir": str(CONFIGS_DIR),
        "fallback_dir": str(FALLBACK_CONFIGS_DIR),
        "ensemble_cfg": cfg,
    }


# -----------------------------------------------------------------------------
# Infer (기존 로직 유지)
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


@app.post(
    "/loop/infer/run",
    response_model=InferRunResponse,
    tags=["Loop: Inference"],
    summary="Unlabeled 추론+앙상블 루프 시작",
    description=(
        "unlabeled_dir 내 이미지를 스캔한 뒤 batch_size 단위로 3개 모델에 추론을 요청하고, "
        "Judge에서 앙상블(PASS_3/PASS_2/FAIL/MISS)을 생성하는 비동기 루프를 시작합니다.\n\n"
        "- 즉시 run_id를 반환하며, 진행상황은 /loop/infer/status/{run_id}에서 조회합니다.\n"
        "- user_id/project_id 제공 시 모델 컨테이너에서 프로젝트 스코프 weight 선택에 활용 가능합니다."
    ),
    responses={
        200: {"description": "루프가 정상적으로 시작됨"},
        400: {"description": "요청 파라미터 오류"},
        500: {"description": "서버 내부 오류"},
    },
)
def loop_infer_run(
    req: InferRunRequest = Body(
        ...,
        openapi_examples={
            "basic": {
                "summary": "기본 실행",
                "value": {
                    "unlabeled_dir": "/workspace/storage/datasets/unlabeled/images",
                    "batch_size": 8,
                    "segment_size_batches": 100,
                    "infer_params": {"conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "0"},
                },
            },
            "scoped": {
                "summary": "프로젝트 스코프 (user_id/project_id 포함)",
                "value": {
                    "user_id": "user_001",
                    "project_id": "project_demo",
                    "unlabeled_dir": "/workspace/storage/datasets/project_demo/unlabeled/images",
                    "batch_size": 16,
                    "segment_size_batches": 50,
                    "infer_params": {"conf": 0.35, "imgsz": 640, "device": "0"},
                },
            },
        },
    ),
    background: BackgroundTasks = None,  # FastAPI 주입용 (타입힌트 유지)
):
    run_id = req.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    req = req.model_copy(update={"run_id": run_id})

    # BackgroundTasks는 FastAPI가 주입하므로 None 방어
    if background is None:
        raise HTTPException(status_code=500, detail="BackgroundTasks injection failed")

    background.add_task(_do_infer, req)
    return InferRunResponse(run_id=run_id, status="RUNNING")


@app.get(
    "/loop/infer/status/{run_id}",
    tags=["Loop: Inference"],
    summary="추론 루프 상태 조회",
    description="run_id 기준으로 현재 진행상태(RUNNING/DONE/FAIL 등) 및 처리 정보를 조회합니다.",
    responses={200: {"description": "상태 조회 성공"}, 404: {"description": "run_id 상태 파일이 없음(미실행/만료)"}},
)
def loop_infer_status(
    run_id: str = FPath(..., description="Infer 루프 실행 식별자 (/loop/infer/run 응답의 run_id)"),
):
    return read_status(run_id)


# -----------------------------------------------------------------------------
# 🔥 Train – Judge Orchestration
# -----------------------------------------------------------------------------
def _call_train(model: str, req: GTTrainRequest) -> Dict[str, Any]:
    cfg, _ = _load_models_cfg_with_path()
    m = cfg["models"].get(model)
    if not m:
        raise HTTPException(status_code=400, detail=f"unknown model: {model}")

    base = m["base_url"].rstrip("/")
    ep = m.get("endpoints", {}).get("train_gt", "/train/gt")
    url = f"{base}{ep}"

    # ⚠️ timeout=10은 학습 자체가 아니라 "학습 트리거 요청" 왕복만 커버
    r = requests.post(url, json=req.model_dump(), timeout=10)
    r.raise_for_status()
    return r.json()


@app.post(
    "/loop/train/gt/run",
    response_model=GTTrainResponse,
    tags=["Loop: Train GT"],
    summary="GT 학습 실행 (단일 모델)",
    description=(
        "요청의 req.model에 해당하는 모델 컨테이너로 /train/gt 요청을 전달하여 GT 학습을 시작합니다.\n\n"
        "- 실제 학습은 모델 컨테이너 내부에서 비동기적으로 진행될 수 있습니다.\n"
        "- 학습 상태는 /loop/train/status/{model}/{train_job_id} 로 조회합니다."
    ),
)
def loop_train_gt_run(
    req: GTTrainRequest = Body(
        ...,
        openapi_examples={
            "yolo_train": {
                "summary": "YOLOv11 GT 학습 예시",
                "value": {
                    "identity": {"user_key": "user_001", "project_id": "project_demo", "job_id": "job_001"},
                    "model": "yolov11",
                    "dataset": {"format": "yolo", "img_root": "/workspace/storage/datasets/gt", "train": "labels/train"},
                    "train": {"epochs": 30, "imgsz": 640, "batch": 8, "device": "0", "extra": {}},
                    "init_weight_type": "baseline",
                },
            }
        },
    )
):
    return _call_train(req.model, req)


@app.post(
    "/loop/train/gt/run_all",
    tags=["Loop: Train GT"],
    summary="GT 학습 실행 (3모델 전체)",
    description=(
        "YOLOv11/RTM/RT-DETR 3개 모델에 대해 GT 학습 트리거를 순차 호출합니다.\n"
        "- 일부 모델이 실패해도 나머지 결과는 results에 담고, failures는 errors에 담아 반환합니다."
    ),
)
def loop_train_gt_run_all(
    req: GTTrainRequest = Body(
        ...,
        openapi_examples={
            "run_all": {
                "summary": "3모델 전체 GT 학습 예시",
                "value": {
                    "identity": {"user_key": "user_001", "project_id": "project_demo", "job_id": "job_all_001"},
                    "model": "yolov11",
                    "dataset": {"format": "yolo", "img_root": "/workspace/storage/datasets/gt", "train": "labels/train"},
                    "train": {"epochs": 30, "imgsz": 640, "batch": 8, "device": "0", "extra": {}},
                    "init_weight_type": "baseline",
                },
            }
        },
    )
):
    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for model in ("yolov11", "rtm", "rtdetr"):
        try:
            model_req = req.model_copy(update={"model": model})
            results[model] = _call_train(model, model_req)
        except Exception as e:
            errors[model] = f"{type(e).__name__}: {e}"

    if errors:
        return {"ok": False, "results": results, "errors": errors}

    return {"ok": True, "results": results}


@app.get(
    "/loop/train/status/{model}/{train_job_id}",
    tags=["Loop: Train GT"],
    summary="GT 학습 상태 조회",
    description=(
        "모델 컨테이너의 train status endpoint로 프록시하여 학습 상태를 조회합니다.\n"
        "- models.yaml의 models.{model}.endpoints.train_status 값을 사용합니다.\n"
        "- 기본값은 /train/status/{train_job_id} 형태를 가정합니다."
    ),
)
def loop_train_gt_status(
    model: str = FPath(..., description="대상 모델 이름 (yolov11 | rtm | rtdetr)"),
    train_job_id: str = FPath(..., description="모델 컨테이너가 발급/사용하는 학습 job id"),
    timeout_s: float = Query(5.0, description="상태 조회 요청 timeout(초). 모델 컨테이너가 응답이 느릴 때 조정"),
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
