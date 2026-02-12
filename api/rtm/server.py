# api/rtm/server.py
from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

MODEL_NAME: ModelName = "rtm"

# =============================================================================
# ✅ CODE-LEVEL GPU PINNING (docker-compose 안 먹을 때 최후의 보루)
#   - torch/mmengine/mmdet 등 CUDA를 만질 수 있는 모듈 import 전에 실행돼야 함.
#   - 컨테이너에 GPU가 여러 개 노출돼도, 이 프로세스는 "0번만 보이게" 강제한다.
# =============================================================================
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0").strip() or "0"

# 컨테이너 내부 torch 기준 device는 항상 cuda:0 (보이는 GPU가 1개뿐이므로)
FIXED_GPU_INDEX = "0"
FIXED_DEVICE = "cuda:0"


def _fixed_device() -> str:
    return str(FIXED_DEVICE)


# =============================================================================
# ✅ OVERRIDE POLICY (요청이 device를 보내도 에러내지 말고 "무시")
#   - 프론트/게이트웨이가 device를 고정으로 보내는 경우(예: "0")가 있어서 400이 나면 불편함
#   - 대신 로그로만 남기고, 실제 실행 device는 항상 FIXED_DEVICE를 사용한다.
# =============================================================================
def _ignore_device_override(params: Any) -> Optional[str]:
    """infer 요청 params.device가 들어오면 기록만 하고 무시"""
    if not isinstance(params, dict):
        return None
    dv = params.get("device")
    if dv is None:
        return None
    return str(dv)


def _ignore_train_device_override(req: GTTrainRequest) -> Optional[str]:
    """train 요청 train.device가 들어오면 기록만 하고 무시"""
    if getattr(req, "train", None) is None:
        return None
    dv = getattr(req.train, "device", None)
    if dv in (None, "", "null"):
        return None
    return str(dv)


# ✅ GT roots (v2: gt_version 지원)
GT_CURRENT_ROOT = Path(os.getenv("GT_CURRENT_ROOT", "/workspace/storage/GT/current")).resolve()
GT_VERSIONS_ROOT = Path(os.getenv("GT_VERSIONS_ROOT", "/workspace/storage/GT_versions")).resolve()

DEFAULT_REGISTRY_ROOT = os.getenv("MODEL_REGISTRY_ROOT", "/workspace/logs/model_train")
REGISTRY_DIR = (Path(DEFAULT_REGISTRY_ROOT) / str(MODEL_NAME)).resolve()
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_ROOT = Path(os.getenv("WEIGHTS_ROOT", "/weights")).resolve()
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "640"))
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))

RTM_CONFIG = os.getenv("RTM_CONFIG", "").strip()

logger = setup_logger(
    service=os.getenv("MODEL_NAME", str(MODEL_NAME)),
    log_file=os.getenv("MODEL_LOG_FILE", f"/workspace/logs/{MODEL_NAME}/api.jsonl"),
    level=os.getenv("LOG_LEVEL", "INFO"),
)

# =============================================================================
# ✅ IMPORTANT
#   - 여기서 src.rtm.infer / train_gt import 될 때 내부에서 torch/cuda init 될 수 있음.
#   - 위에서 CUDA_VISIBLE_DEVICES를 먼저 박았기 때문에 torch는 GPU "0만 존재"한다고 인식.
# =============================================================================
try:
    from src.rtm.infer import infer_batch as _infer_batch
except Exception:  # pragma: no cover
    _infer_batch = None

try:
    from src.rtm.train_gt import train_gt as _train_gt
except Exception:  # pragma: no cover
    _train_gt = None

app = FastAPI(title=f"{MODEL_NAME}-api", version="0.4.4")

log_event(
    logger,
    "service boot",
    ctx=LogContext(model=str(MODEL_NAME), stage="boot"),
    registry_dir=str(REGISTRY_DIR),
    weights_root=str(WEIGHTS_ROOT),
    gt_current_root=str(GT_CURRENT_ROOT),
    gt_versions_root=str(GT_VERSIONS_ROOT),
    device_policy="fixed",
    fixed_gpu_index=str(FIXED_GPU_INDEX),
    fixed_device=str(FIXED_DEVICE),
    cuda_visible_devices=str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    default_imgsz=DEFAULT_IMGSZ,
    default_conf=DEFAULT_CONF,
    rtm_config=(RTM_CONFIG or None),
    infer_runner_available=(_infer_batch is not None),
    train_runner_available=(_train_gt is not None),
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


def _iter_ckpts(weights_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    if not weights_dir.exists():
        return []
    cands: List[Path] = []
    for ext in exts:
        cands += [p for p in weights_dir.rglob(f"*.{ext}") if p.is_file()]
    return cands


def _pick_latest_ckpt(weights_dir: Path, exts: Tuple[str, ...] = ("pth", "pt", "ckpt")) -> Optional[Path]:
    cands = _iter_ckpts(weights_dir, exts)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _list_version_dirs(base: Path) -> List[Path]:
    """v### 폴더만 추출 후 번호 순 정렬(내림차순)"""
    if not base.exists():
        return []
    vers: List[Tuple[int, Path]] = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("v") and len(p.name) == 4 and p.name[1:].isdigit():
            vers.append((int(p.name[1:]), p))
    vers.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in vers]


def _pick_best_or_latest_in_dir(d: Path, exts: Tuple[str, ...]) -> Optional[Path]:
    if not d.exists():
        return None

    best_cands: List[Path] = []
    for ext in exts:
        best_cands += list(d.rglob(f"best*.{ext}"))
    best_cands = [p for p in best_cands if p.is_file()]
    if best_cands:
        best_cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return best_cands[0]

    return _pick_latest_ckpt(d, exts=exts)


def _pick_latest_project_weight(user_key: str, project_id: str) -> Optional[Path]:
    base = _project_model_dir(user_key, project_id)
    exts = ("pth", "pt", "ckpt")

    vdirs = _list_version_dirs(base)
    for vd in vdirs:
        picked = _pick_best_or_latest_in_dir(vd, exts=exts)
        if picked is not None:
            return picked

    return _pick_latest_ckpt(base, exts=exts)


def _next_project_version(user_key: str, project_id: str) -> int:
    base = _project_model_dir(user_key, project_id)
    vdirs = _list_version_dirs(base)
    if not vdirs:
        return 1
    max_v = int(vdirs[0].name[1:])
    return max_v + 1


def _base_bdd100k_weight() -> Path:
    base_dir = (WEIGHTS_ROOT / "_base" / str(MODEL_NAME)).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")

    cands: List[Path] = []
    for ext in ("pth", "pt", "ckpt"):
        cands += list(base_dir.glob(f"base_bdd100k*.{ext}"))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        raise FileNotFoundError(f"base_bdd100k.* not found under: {base_dir}")

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


# -----------------------------
# Inference helpers
# -----------------------------
def _empty_results(req: InferBatchRequest) -> List[InferItemResult]:
    return [InferItemResult(image_id=it.image_id, detections=[]) for it in (req.items or [])]


def _parse_detection(d: Any) -> Optional[Detection]:
    if not isinstance(d, dict):
        return None
    cls = d.get("cls")
    conf = d.get("conf")
    xyxy = d.get("xyxy")
    if not isinstance(cls, int):
        return None
    if not isinstance(conf, (int, float)):
        return None
    if not (isinstance(xyxy, list) and len(xyxy) == 4 and all(isinstance(x, (int, float)) for x in xyxy)):
        return None
    return Detection(cls=int(cls), conf=float(conf), xyxy=[float(x) for x in xyxy])


def _as_dict_maybe(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            return {}
    return {}


def _iter_meta_candidates(out: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    meta = out.get("meta")
    if isinstance(meta, dict):
        yield meta

    raw = out.get("raw")
    raw_d = _as_dict_maybe(raw)
    meta2 = raw_d.get("meta")
    if isinstance(meta2, dict):
        yield meta2

    raw2 = raw_d.get("raw")
    raw2_d = _as_dict_maybe(raw2)
    meta3 = raw2_d.get("meta")
    if isinstance(meta3, dict):
        yield meta3


def _extract_meta_str(out: Dict[str, Any], key: str) -> Optional[str]:
    for m in _iter_meta_candidates(out):
        v = m.get(key)
        if v:
            return str(v)
    return None


def _extract_weight_dir_used(out: Dict[str, Any], req_weight_dir: Optional[str]) -> Optional[str]:
    v = _extract_meta_str(out, "weight_dir_used")
    if v:
        return v
    return str(req_weight_dir) if req_weight_dir else None


def _extract_weight_used(out: Dict[str, Any]) -> Optional[str]:
    v = out.get("weight_used")
    if isinstance(v, (str, Path)):
        return str(v)
    if v is not None:
        return str(v)
    mv = _extract_meta_str(out, "weight_used")
    return str(mv) if mv else None


def _coerce_infer_output(
    raw_out: Any,
    req: InferBatchRequest,
    *,
    elapsed_ms: int,
) -> Tuple[List[InferItemResult], Optional[str], bool, Optional[str], str, Optional[str]]:
    out: Dict[str, Any] = _as_dict_maybe(raw_out)

    ok = bool(out.get("ok", True))
    batch_id = str(out.get("batch_id", req.batch_id))
    weight_used = _extract_weight_used(out)

    error = out.get("error")
    error = str(error) if error is not None else None

    req_weight_dir: Optional[str] = None
    try:
        params = req.params or {}
        if isinstance(params, dict):
            req_weight_dir = params.get("weight_dir")
    except Exception:
        req_weight_dir = None

    weight_dir_used = _extract_weight_dir_used(out, req_weight_dir)

    raw_results = out.get("results")
    if not isinstance(raw_results, list):
        raw_results = []

    fixed_results: List[InferItemResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id")
        if not isinstance(image_id, str):
            continue
        dets_raw = item.get("detections", [])
        if not isinstance(dets_raw, list):
            dets_raw = []
        dets: List[Detection] = []
        for d in dets_raw:
            dd = _parse_detection(d)
            if dd is not None:
                dets.append(dd)
        fixed_results.append(InferItemResult(image_id=image_id, detections=dets))

    seen = {r.image_id for r in fixed_results}
    for it in req.items:
        if it.image_id not in seen:
            fixed_results.append(InferItemResult(image_id=it.image_id, detections=[]))

    if not fixed_results:
        fixed_results = _empty_results(req)

    return fixed_results, weight_used, ok, error, batch_id, weight_dir_used


# -----------------------------
# GT helpers (v2: gt_version)
# -----------------------------
def _select_gt_root(req: GTTrainRequest) -> Path:
    """
    ✅ GT 루트 결정
    우선순위:
      1) req.gt.gt_version 있으면 GT_VERSIONS_ROOT/<gt_version>
      2) req.dataset.img_root 기반 (images로 들어오면 parent 보정)
      3) fallback: GT_CURRENT_ROOT
    """
    if getattr(req, "gt", None) is not None and getattr(req.gt, "gt_version", None):
        return (GT_VERSIONS_ROOT / req.gt.gt_version).resolve()

    if req.dataset is not None and req.dataset.img_root:
        img_root = Path(req.dataset.img_root).resolve()

        if (
            (img_root / "images").exists()
            or (img_root / "labels").exists()
            or (img_root / "annotation").exists()
            or (img_root / "annotations").exists()
        ):
            return img_root

        parent = img_root.parent
        if (
            (parent / "images").exists()
            or (parent / "labels").exists()
            or (parent / "annotation").exists()
            or (parent / "annotations").exists()
        ):
            return parent

        return img_root

    return GT_CURRENT_ROOT


def _resolve_rtm_ann_paths(gt_root: Path) -> Tuple[Path, Path, Path]:
    """
    ✅ RTM annotation 위치 자동 탐색
    """
    for dname in ("annotation", "annotations"):
        ann_dir = gt_root / dname
        tr = ann_dir / "train.json"
        va = ann_dir / "val.json"
        if tr.exists() and va.exists():
            return ann_dir, tr, va

    ann_dir = gt_root / "rtm" / "annotations"
    tr = ann_dir / "instances_train.json"
    va = ann_dir / "instances_val.json"
    if tr.exists() and va.exists():
        return ann_dir, tr, va

    raise FileNotFoundError(
        "RTM annotation not found. Expected one of:\n"
        f"  - {gt_root}/annotation/train.json + val.json\n"
        f"  - {gt_root}/annotations/train.json + val.json\n"
        f"  - {gt_root}/rtm/annotations/instances_train.json + instances_val.json"
    )


# -----------------------------
# HTTP endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": str(MODEL_NAME),
        "device_policy": "fixed",
        "fixed_gpu_index": "0",
        "fixed_device": "cuda:0",
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    }


@app.post("/infer")
def infer(req: InferBatchRequest) -> Dict[str, Any]:
    t0 = time.time()

    params = req.params or {}
    requested_device = _ignore_device_override(params)  # ✅ 기록만

    run_id = params.get("run_id")
    weight_dir = params.get("weight_dir")

    # ✅ 실제 실행은 고정 device
    device = _fixed_device()
    conf = float(params.get("conf", DEFAULT_CONF))
    imgsz = int(params.get("imgsz", DEFAULT_IMGSZ))

    ctx = LogContext(run_id=str(run_id) if run_id else None, stage="infer", model=str(MODEL_NAME))

    log_event(
        logger,
        "infer request",
        ctx=ctx,
        batch_id=req.batch_id,
        n_images=len(req.items),
        runner_available=(_infer_batch is not None),
        rtm_config=(RTM_CONFIG or None),
        weight_dir=str(weight_dir) if weight_dir else None,
        device=device,
        requested_device=requested_device,  # ✅ 들어온 값은 기록
        device_policy="fixed",
        fixed_gpu_index=str(FIXED_GPU_INDEX),
        conf=conf,
        imgsz=imgsz,
        cuda_visible_devices=str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    )

    if _infer_batch is None:
        elapsed_ms = int((time.time() - t0) * 1000)
        log_event(
            logger,
            "infer fallback (runner not available)",
            ctx=ctx,
            batch_id=req.batch_id,
            elapsed_ms=elapsed_ms,
            level="WARNING",
        )
        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=False,
            error="infer runner not available (import src.rtm.infer.infer_batch failed)",
            elapsed_ms=elapsed_ms,
            weight_used=None,
            results=_empty_results(req),
            raw={
                "meta": {
                    "weight_dir_used": str(weight_dir) if weight_dir else None,
                    "device": device,
                    "requested_device": requested_device,
                }
            },
        )
        return resp.model_dump()

    try:
        # ✅ runner가 params['device']를 참조할 수 있도록 "고정값" 주입
        req2 = req
        try:
            p2 = dict(req.params or {})
            p2["device"] = device
            p2.setdefault("conf", conf)
            p2.setdefault("imgsz", imgsz)
            req2 = req.model_copy(update={"params": p2})
        except Exception:
            req2 = req

        try:
            raw_out = _infer_batch(req2)  # type: ignore[misc]
        except TypeError:
            payload = req2.model_dump(by_alias=True)
            payload.setdefault("params", {})
            if isinstance(payload["params"], dict):
                payload["params"]["device"] = device
                payload["params"].setdefault("conf", conf)
                payload["params"].setdefault("imgsz", imgsz)
            raw_out = _infer_batch(payload)  # type: ignore[misc]

        elapsed_ms = int((time.time() - t0) * 1000)
        results, weight_used, ok, error, batch_id, weight_dir_used = _coerce_infer_output(
            raw_out, req, elapsed_ms=elapsed_ms
        )

        log_event(
            logger,
            "infer response",
            ctx=ctx,
            ok=ok,
            batch_id=batch_id,
            elapsed_ms=elapsed_ms,
            weight_used=weight_used,
            weight_dir_used=weight_dir_used,
            n_results=len(results),
            device=device,
            requested_device=requested_device,
            device_policy="fixed",
        )

        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=batch_id,
            ok=ok,
            error=error,
            elapsed_ms=elapsed_ms,
            weight_used=weight_used,
            results=results,
            raw={
                "meta": {
                    "weight_dir_used": weight_dir_used,
                    "weight_used": weight_used,
                    "device_policy": "fixed",
                    "device": device,
                    "requested_device": requested_device,
                    "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
                },
                "runner": "src.rtm.infer.infer_batch",
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
                "elapsed_ms": elapsed_ms,
                "error": f"{type(e).__name__}: {e}",
                "weight_dir": str(weight_dir) if weight_dir else None,
                "device": device,
                "requested_device": requested_device,
                "device_policy": "fixed",
            },
        )
        resp = ModelPredResponse(
            model=MODEL_NAME,
            batch_id=req.batch_id,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=elapsed_ms,
            weight_used=None,
            results=_empty_results(req),
            raw={
                "meta": {
                    "weight_dir_used": str(weight_dir) if weight_dir else None,
                    "device": device,
                    "requested_device": requested_device,
                }
            },
        )
        return resp.model_dump()


# -----------------------------
# Core: Train (v2 gt_version 지원) (FIXED DEVICE)
# -----------------------------
def _do_train(train_job_id: str, req: GTTrainRequest) -> None:
    ident = req.identity
    ctx = LogContext(job_id=ident.job_id, run_id=ident.run_id, stage="train_gt", model=str(MODEL_NAME))

    job = _read_job(train_job_id)
    job["status"] = "RUNNING"
    job["started_at"] = time.time()
    _write_job(train_job_id, job)

    requested_train_device = _ignore_train_device_override(req)  # ✅ 기록만

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
        device_policy="fixed",
        fixed_gpu_index=str(FIXED_GPU_INDEX),
        fixed_device=str(FIXED_DEVICE),
        requested_train_device=requested_train_device,
        cuda_visible_devices=str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    )

    try:
        if not ident.job_id:
            raise ValueError("identity.job_id is required")

        cfg = (req.train.extra or {}).get("config") or RTM_CONFIG
        if not cfg:
            raise ValueError("RTM config is required. Set env RTM_CONFIG or provide train.extra['config'].")

        if _train_gt is None:
            raise RuntimeError("RTM train runner not available (import src.rtm.train_gt.train_gt failed)")

        gt_root = _select_gt_root(req)
        if not gt_root.exists():
            raise FileNotFoundError(f"resolved gt_root not found: {gt_root}")

        ann_dir, train_path, val_path = _resolve_rtm_ann_paths(gt_root)
        init_weight = _resolve_init_for_train(req)

        user_key = str(ident.user_key)
        project_id = str(ident.project_id)
        next_v = _next_project_version(user_key, project_id)
        version_tag = f"v{next_v:03d}"
        out_dir = _project_model_dir(user_key, project_id) / version_tag
        save_path = out_dir / "best.pth"

        train_device = _fixed_device()  # ✅ ALWAYS cuda:0

        resolved = {
            "config": str(cfg),
            "gt_root": str(gt_root),
            "init_weight": str(init_weight),
            "save_path": str(save_path),
            "version": version_tag,
            "dataset_train": str(train_path),
            "dataset_val": str(val_path),
            "annotation_dir": str(ann_dir),
            "device_policy": "fixed",
            "fixed_gpu_index": str(FIXED_GPU_INDEX),
            "device": train_device,
            "requested_train_device": requested_train_device,
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        }
        job = _read_job(train_job_id)
        job["resolved"] = resolved
        _write_job(train_job_id, job)

        log_event(logger, "train_gt resolved", ctx=ctx, train_job_id=train_job_id, **resolved)

        out_dir.mkdir(parents=True, exist_ok=True)
        extra = dict(req.train.extra or {})
        extra["_dataset"] = {
            "img_root": str(gt_root),
            "train": str(train_path),
            "val": str(val_path),
            "annotation_dir": str(ann_dir),
        }

        artifacts = _train_gt(
            config=str(cfg),
            gt_root=str(gt_root),
            init_ckpt=str(init_weight),
            out_ckpt=str(save_path),
            epochs=int(req.train.epochs),
            imgsz=int(req.train.imgsz),
            device=train_device,  # ✅ FIXED
            extra=extra,
        )

        if not Path(save_path).exists():
            raise FileNotFoundError(f"train finished but output ckpt not found: {save_path}")

        job = _read_job(train_job_id)
        job["status"] = "DONE"
        job["finished_at"] = time.time()
        job["artifacts"] = {
            "trained_weight": str(save_path),
            "version": version_tag,
            "init_weight": str(init_weight),
            "gt_root": str(gt_root),
            "annotation_dir": str(ann_dir),
            "device_policy": "fixed",
            "fixed_gpu_index": str(FIXED_GPU_INDEX),
            "device": train_device,
            "requested_train_device": requested_train_device,
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "runner_artifacts": artifacts if isinstance(artifacts, dict) else {"note": "runner returned non-dict"},
        }
        _write_job(train_job_id, job)

        log_event(
            logger,
            "train_gt done",
            ctx=ctx,
            train_job_id=train_job_id,
            trained_weight=str(save_path),
            version=version_tag,
            device=train_device,
            requested_train_device=requested_train_device,
            device_policy="fixed",
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

    # ✅ 여기서도 그냥 기록만 (에러 X)
    requested_train_device = _ignore_train_device_override(req)

    if req.dataset is not None:
        if not req.dataset.img_root:
            raise HTTPException(status_code=400, detail="dataset.img_root is required when dataset is provided")

    train_job_id = f"{MODEL_NAME}_{uuid.uuid4().hex[:10]}"

    record = {
        "train_job_id": train_job_id,
        "identity": req.identity.model_dump(),
        "model": str(MODEL_NAME),
        "status": "QUEUED",
        "request": req.model_dump(),
        "created_at": time.time(),
        "device_policy": "fixed",
        "fixed_gpu_index": "0",
        "fixed_device": "cuda:0",
        "requested_train_device": requested_train_device,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    }
    _write_job(train_job_id, record)

    ctx = LogContext(job_id=req.identity.job_id, run_id=req.identity.run_id, stage="train_gt", model=str(MODEL_NAME))
    log_event(
        logger,
        "train_gt queued",
        ctx=ctx,
        train_job_id=train_job_id,
        device_policy="fixed",
        fixed_gpu_index="0",
        fixed_device="cuda:0",
        requested_train_device=requested_train_device,
        cuda_visible_devices=str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
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
