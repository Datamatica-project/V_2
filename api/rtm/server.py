from __future__ import annotations

import json
import os
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

from fastapi import FastAPI, BackgroundTasks, HTTPException

# ✅ SSOT import 규칙 준수 (api/는 src/만 import)
from src.common.utils.logging import setup_logger, LogContext, log_event

# ✅ DTO SSOT (중복 정의 금지)
from src.common.dto.dto import (
    InferBatchRequest,
    ModelName,
    Detection,
    InferItemResult,
    ModelPredResponse,
    GTTrainRequest,
    GTTrainResponse,
)

MODEL_NAME: ModelName = "rtm"

DEFAULT_REGISTRY_ROOT = os.getenv("MODEL_REGISTRY_ROOT", "/workspace/logs/model_train")
REGISTRY_DIR = Path(DEFAULT_REGISTRY_ROOT) / str(MODEL_NAME)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_ROOT = Path(os.getenv("WEIGHTS_ROOT", "/weights"))  # ✅ root 권장
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_DEVICE = os.getenv("DEVICE", "cuda:0")
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "640"))
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))

# RTM config - env로 주입 권장
RTM_CONFIG = os.getenv("RTM_CONFIG", "").strip()

logger = setup_logger(
    service=os.getenv("MODEL_NAME", str(MODEL_NAME)),
    log_file=os.getenv("MODEL_LOG_FILE", f"/workspace/logs/{MODEL_NAME}/api.jsonl"),
    level=os.getenv("LOG_LEVEL", "INFO"),
)

# -----------------------------
# Optional runner imports
# -----------------------------
try:
    # infer_batch(req: InferBatchRequest|dict) -> dict|pydantic
    from src.rtm.infer import infer_batch as _infer_batch  # type: ignore
except Exception:  # pragma: no cover
    _infer_batch = None  # type: ignore

try:
    # train_gt(*, config, gt_root, init_ckpt, out_ckpt, epochs, imgsz, device, extra) -> dict
    from src.rtm.train_gt import train_gt as _train_gt  # type: ignore
except Exception:  # pragma: no cover
    _train_gt = None  # type: ignore

app = FastAPI(title=f"{MODEL_NAME}-api", version="0.4.1")

log_event(
    logger,
    "service boot",
    ctx=LogContext(model=str(MODEL_NAME), stage="boot"),
    registry_dir=str(REGISTRY_DIR),
    weights_root=str(WEIGHTS_ROOT),
    device=DEFAULT_DEVICE,
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
# Weight policy helpers (V2)
# -----------------------------
def _project_model_dir(user_key: str, project_id: str) -> Path:
    # /weights/projects/{user}/{project}/{model}
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
    """v### 폴더만 추출 후 번호 순 정렬"""
    if not base.exists():
        return []
    vers: List[Tuple[int, Path]] = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("v") and len(p.name) == 4 and p.name[1:].isdigit():
            vers.append((int(p.name[1:]), p))
    vers.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in vers]


def _pick_best_or_latest_in_dir(d: Path, exts: Tuple[str, ...]) -> Optional[Path]:
    """
    디렉토리 안에서 best.* 우선, 없으면 mtime 최신.
    """
    if not d.exists():
        return None

    # best 우선
    best_cands: List[Path] = []
    for ext in exts:
        best_cands += list(d.rglob(f"best*.{ext}"))
    best_cands = [p for p in best_cands if p.is_file()]
    if best_cands:
        best_cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return best_cands[0]

    # 없으면 전체 최신
    return _pick_latest_ckpt(d, exts=exts)


def _pick_latest_project_weight(user_key: str, project_id: str) -> Optional[Path]:
    """
    정책:
      - 가장 큰 v### 폴더부터 탐색
      - 각 v폴더 내에서 best.* 우선, 없으면 최신 ckpt
      - v폴더가 없으면 projects/{user}/{project}/{model} 전체에서 최신 ckpt
    """
    base = _project_model_dir(user_key, project_id)
    exts = ("pth", "pt", "ckpt")

    vdirs = _list_version_dirs(base)
    for vd in vdirs:
        picked = _pick_best_or_latest_in_dir(vd, exts=exts)
        if picked is not None:
            return picked

    # v 폴더가 없거나 내부가 비었으면 전체에서 최신
    return _pick_latest_ckpt(base, exts=exts)


def _next_project_version(user_key: str, project_id: str) -> int:
    """
    projects/{user}/{project}/{model}/v### 폴더 기준으로 다음 버전 계산 (v001부터)
    """
    base = _project_model_dir(user_key, project_id)
    vdirs = _list_version_dirs(base)
    if not vdirs:
        return 1
    # _list_version_dirs는 내림차순이므로 첫 번째가 max
    max_v = int(vdirs[0].name[1:])
    return max_v + 1


def _base_bdd100k_weight() -> Path:
    """
    v1 init: /weights/_base/{model}/base_bdd100k.*
    (RTM은 보통 .pth)
    """
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
    """
    정책:
      - req.baseline_weight_path 있으면 그걸 사용 (존재해야 함)
      - 없으면:
          - projects/{user}/{project}/{model}/ 에 기존 버전이 있으면 "직전 버전(latest, best 우선)"에서 시작 (v2~)
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


# -----------------------------
# Infer response normalization -> ModelPredResponse(SSOT)
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
    """
    여러 구현/버전에서 meta가 올 수 있는 위치를 전부 훑는다.
      - out["meta"]
      - out["raw"]["meta"]
      - out["raw"]["raw"]["meta"]  (이중 래핑되는 케이스 방어)
    """
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
    """
    runner가 meta.weight_dir_used를 내려주면 그걸 우선 사용.
    없으면 요청의 weight_dir을 기록(추적 목적).
    """
    v = _extract_meta_str(out, "weight_dir_used")
    if v:
        return v
    return str(req_weight_dir) if req_weight_dir else None


def _extract_weight_used(out: Dict[str, Any]) -> Optional[str]:
    """
    weight_used 위치 다양성 대응:
      - out["weight_used"]
      - meta.weight_used (여러 위치)
    """
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
    """
    raw_out(whatever) -> (results, weight_used, ok, error, batch_id, weight_dir_used)

    - 결과는 SSOT InferItemResult / Detection로 강제 변환
    - 누락된 image_id는 빈 결과로 채움
    - weight_dir_used는 meta 우선, 없으면 request params.weight_dir 기록
    """
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

    # 누락 보정
    seen = {r.image_id for r in fixed_results}
    for it in req.items:
        if it.image_id not in seen:
            fixed_results.append(InferItemResult(image_id=it.image_id, detections=[]))

    # 결과가 비어있다면 요청 기준 empty로
    if not fixed_results:
        fixed_results = _empty_results(req)

    return fixed_results, weight_used, ok, error, batch_id, weight_dir_used


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "model": str(MODEL_NAME)}


@app.post("/infer")
def infer(req: InferBatchRequest) -> Dict[str, Any]:
    t0 = time.time()
    params = req.params or {}
    run_id = params.get("run_id")
    weight_dir = params.get("weight_dir")  # ✅ gateway가 주입하는 스코프
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
            raw={"meta": {"weight_dir_used": str(weight_dir) if weight_dir else None}},
        )
        return resp.model_dump()

    try:
        try:
            raw_out = _infer_batch(req)  # type: ignore[misc]
        except TypeError:
            # legacy runner는 dict를 받을 수 있음
            raw_out = _infer_batch(req.model_dump(by_alias=True))  # type: ignore[misc]

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
            raw={"meta": {"weight_dir_used": str(weight_dir) if weight_dir else None}},
        )
        return resp.model_dump()
def _do_train(train_job_id: str, req: GTTrainRequest) -> None:
    ident = req.identity
    ctx = LogContext(job_id=ident.job_id, run_id=ident.run_id, stage="train_gt", model=str(MODEL_NAME))

    job = _read_job(train_job_id)
    job["status"] = "RUNNING"
    job["started_at"] = time.time()
    _write_job(train_job_id, job)

    try:
        if not ident.job_id:
            raise ValueError("identity.job_id is required")

        # RTM config 필수
        cfg = (req.train.extra or {}).get("config") or RTM_CONFIG
        if not cfg:
            raise ValueError("RTM config is required. Set env RTM_CONFIG or provide train.extra['config'].")

        if _train_gt is None:
            raise RuntimeError("RTM train runner not available (import src.rtm.train_gt.train_gt failed)")

        # ---------------------------------------------------------------------
        # ✅ SSOT: RTM은 annotation/ 아래 COCO json만 사용
        #   - gt_register에서 annotation/train.json, annotation/val.json을 생성한다.
        #   - val은 train과 동일 파일이어도 OK(테스트 목적)
        # ---------------------------------------------------------------------
        gt_root = Path(req.dataset.img_root).resolve()
        if not gt_root.exists():
            raise FileNotFoundError(f"dataset.img_root not found: {gt_root}")

        # 방어적 지원: annotation/ 우선, 없으면 annotations/도 허용
        ann_dir = gt_root / "annotation"
        if not ann_dir.exists():
            ann_dir_alt = gt_root / "annotations"
            if ann_dir_alt.exists():
                ann_dir = ann_dir_alt

        if not ann_dir.exists():
            raise FileNotFoundError(f"RTM annotation dir not found: {ann_dir}")

        train_path = ann_dir / "train.json"
        val_path = ann_dir / "val.json"

        if not train_path.exists():
            raise FileNotFoundError(f"RTM train annotation not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"RTM val annotation not found: {val_path}")

        init_weight = _resolve_init_for_train(req)

        user_key = str(ident.user_key)
        project_id = str(ident.project_id)
        next_v = _next_project_version(user_key, project_id)
        version_tag = f"v{next_v:03d}"
        out_dir = _project_model_dir(user_key, project_id) / version_tag
        save_path = out_dir / "best.pth"

        resolved = {
            "config": str(cfg),
            "gt_root": str(gt_root),
            "init_weight": str(init_weight),
            "save_path": str(save_path),
            "version": version_tag,
            # ✅ RTM SSOT 기록
            "dataset_train": str(train_path),
            "dataset_val": str(val_path),
            "annotation_dir": str(ann_dir),
        }
        job = _read_job(train_job_id)
        job["resolved"] = resolved
        _write_job(train_job_id, job)

        log_event(logger, "train_gt resolved", ctx=ctx, train_job_id=train_job_id, **resolved)

        out_dir.mkdir(parents=True, exist_ok=True)

        # runner 시그니처가 고정이므로 train/val은 extra로 전달(호환성 유지)
        extra = dict(req.train.extra or {})
        extra["_dataset"] = {
            "img_root": str(gt_root),
            "train": str(train_path),
            "val": str(val_path),
        }
        # 선택: runner가 annotation_dir을 직접 쓰는 경우 대비
        extra["_dataset"]["annotation_dir"] = str(ann_dir)

        artifacts = _train_gt(
            config=str(cfg),
            gt_root=str(gt_root),
            init_ckpt=str(init_weight),
            out_ckpt=str(save_path),
            epochs=int(req.train.epochs),
            imgsz=int(req.train.imgsz),
            device=str(req.train.device or DEFAULT_DEVICE),
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
    # ✅ model mismatch 차단
    if req.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"model mismatch: got {req.model}, expected {MODEL_NAME}")

    if not req.identity.job_id:
        raise HTTPException(status_code=400, detail="identity.job_id is required")

    if not req.dataset.img_root:
        raise HTTPException(status_code=400, detail="dataset.img_root is required")

    # ✅ TEMP: COCO/recipe 기반 학습은 dataset.train/val을 config(overrides)에서 사용하므로 강제하지 않음
    # TODO: later - dataset.format으로 분기 (yolo면 train 필수, coco면 ann_file 필수)
    # if not req.dataset.train:
    #     raise HTTPException(status_code=400, detail="dataset.train is required")

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
