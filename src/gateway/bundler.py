# src/gateway/bundler.py
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ✅ DTO: 단일 계약 스키마로 통일
from ..common.dto.dto import InferBatchRequest, GatewayBundle, ModelPredResponse, ModelName

# ✅ clients: 상대 import로 통일
from .clients.yolov11 import infer_yolov11
from .clients.rtdetr import infer_rtdetr
from .clients.rtm import infer_rtm  # ✅ YOLOX -> RTM

# ✅ logging: common/util/logging.py 사용 (실제 위치 기준)
from ..common.utils.logging import setup_logger, LogContext, log_event

logger = setup_logger(
    service="gateway",
    logger_name="gateway.bundler",  # ✅ 모듈별 로거 분리 (file handler 충돌 방지)
    log_file="/workspace/logs/gateway/bundler.jsonl",
    level="INFO",
)

# gateway에서도 WEIGHTS_DIR 기준을 맞춰주면 모델 서버 구현에 덜 의존적
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/weights").rstrip("/")
WEIGHTS_DIR_PATH = Path(WEIGHTS_DIR)


# -----------------------------
# helpers
# -----------------------------
def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None


def _sanitize_key(x: str) -> str:
    """
    경로 탈출/구분자 혼입 방지용(최소 방어).
    - 슬래시/백슬래시 제거
    - 공백 trim
    """
    s = (x or "").strip()
    s = s.replace("/", "_").replace("\\", "_")
    return s


def _get_user_project(req: InferBatchRequest) -> tuple[Optional[str], Optional[str]]:
    """
    user_id / project_id를 가능한 여러 위치에서 찾는다.
    우선순위:
      1) req.params.*
      2) req.user_id / req.project_id (DTO에 있으면)
      3) req.identity.* (있으면)
    """
    try:
        # 1) params
        params = getattr(req, "params", None) or {}
        if isinstance(params, dict):
            user_id = (
                params.get("user_id")
                or params.get("uid")
                or params.get("user")
                or params.get("user_key")
            )
            project_id = (
                params.get("project_id")
                or params.get("pid")
                or params.get("project")
                or params.get("project_key")
            )
            u = _safe_str(user_id)
            p = _safe_str(project_id)
            if u and p:
                return u, p

        # 2) direct fields
        u2 = _safe_str(getattr(req, "user_id", None))
        p2 = _safe_str(getattr(req, "project_id", None))
        if u2 and p2:
            return u2, p2

        # 3) identity object (optional)
        ident = getattr(req, "identity", None)
        if ident is not None:
            u3 = _safe_str(getattr(ident, "user_id", None) or getattr(ident, "user_key", None))
            p3 = _safe_str(getattr(ident, "project_id", None) or getattr(ident, "project_key", None))
            if u3 and p3:
                return u3, p3

    except Exception:
        pass

    return None, None


def _has_explicit_weight_file(params: dict) -> bool:
    """
    사용자가 특정 weight '파일'을 명시했다면 gateway가 scope를 덮어쓰지 않는다.
    ⚠️ weight_dir는 '스코프 힌트'이므로 explicit으로 보지 않는다.
    """
    if not isinstance(params, dict):
        return False
    return bool(
        params.get("weight")
        or params.get("weights")
        or params.get("weight_path")
        or params.get("weight_file")
    )


def _build_weight_scope(user_id: str, project_id: str, model: ModelName) -> str:
    """
    모델 러너가 이 디렉토리만 뒤져서 latest를 잡을 수 있게 scope를 만든다.
    절대경로로 제공.
      /weights/projects/{user}/{project}/{model}
    """
    u = _sanitize_key(user_id)
    p = _sanitize_key(project_id)
    return str(WEIGHTS_DIR_PATH / "projects" / u / p / str(model))


def _req_with_weight_scope(req: InferBatchRequest, model: ModelName) -> InferBatchRequest:
    """
    모델별 weight_dir을 params에 주입한 "복사본 req"를 만든다.

    규칙:
      - 사용자가 weight/weight_path 등 '파일'을 명시했으면 건드리지 않는다.
      - user_id/project_id가 없으면 건드리지 않는다.
      - 그 외에는 weight_dir을 모델별 스코프로 보정/주입한다.

    추가 규칙(유연성):
      - 사용자가 params.weight_dir를 줬더라도,
        그것은 "루트 스코프"일 수 있으므로 gateway가 {model} suffix를 붙여 보정한다.
        예) weight_dir="/weights/projects/u/p" -> "/weights/projects/u/p/{model}"
            weight_dir="projects/u/p"         -> "/weights/projects/u/p/{model}"
            weight_dir="/weights/projects/u/p/rtm" -> 이미 모델명 포함이면 그대로 사용
    """
    base_params = getattr(req, "params", None) or {}
    params = dict(base_params) if isinstance(base_params, dict) else {}

    # ✅ 사용자가 weight 파일을 명시했으면 주입/보정 X
    if _has_explicit_weight_file(params):
        return req

    user_id, project_id = _get_user_project(req)
    if not user_id or not project_id:
        return req

    # ✅ 기존 weight_dir가 있으면 "루트 스코프"로 보고 보정
    existing_wdir = _safe_str(params.get("weight_dir") or params.get("weights_dir"))
    if existing_wdir:
        p = Path(existing_wdir).expanduser()
        if not p.is_absolute():
            p = (WEIGHTS_DIR_PATH / p).resolve()

        p_str = str(p).rstrip("/")
        last = Path(p_str).name
        if last != str(model):
            p_str = str(Path(p_str) / str(model))

        params["weight_dir"] = p_str
    else:
        params["weight_dir"] = _build_weight_scope(user_id, project_id, model)

    # pydantic v2: model_copy / v1: copy 호환 처리
    try:
        return req.model_copy(update={"params": params})  # type: ignore[attr-defined]
    except Exception:
        try:
            return req.copy(update={"params": params})  # type: ignore[attr-defined]
        except Exception:
            payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()  # type: ignore[attr-defined]
            payload["params"] = params
            return InferBatchRequest(**payload)


def _weight_dir_used(resp: ModelPredResponse) -> Optional[str]:
    """
    runner meta/응답에서 weight_dir_used를 최대한 안전하게 추출.
    """
    try:
        v = getattr(resp, "weight_dir_used", None)
        if v:
            return str(v)

        raw = getattr(resp, "raw", None)
        if isinstance(raw, dict):
            v2 = raw.get("weight_dir_used")
            if v2:
                return str(v2)

            meta = raw.get("meta")
            if isinstance(meta, dict):
                v3 = meta.get("weight_dir_used")
                if v3:
                    return str(v3)
    except Exception:
        pass
    return None


def _get_run_id(req: InferBatchRequest) -> Optional[str]:
    """req.params.run_id (있으면)"""
    try:
        params = getattr(req, "params", None) or {}
        if isinstance(params, dict):
            rid = params.get("run_id")
            return str(rid) if rid else None
    except Exception:
        pass
    return None


def _weight_used(resp: ModelPredResponse) -> Optional[str]:
    """
    표준 weight_used를 우선 사용하고,
    없으면 raw에서 weight_used를 fallback으로 추출.
    """
    try:
        w = getattr(resp, "weight_used", None)
        if w:
            return str(w)

        raw = getattr(resp, "raw", None)
        if isinstance(raw, dict):
            w2 = raw.get("weight_used")
            return str(w2) if w2 else None
    except Exception:
        pass
    return None


def infer_bundle(
    req: InferBatchRequest,
    *,
    model_cfg: Dict[str, Any],
) -> GatewayBundle:
    """
    - 동시 fan-out (ThreadPool)
    - 판정/정규화/저장 ❌
    - 순수 수집 + 묶기
    """
    t0 = time.time()

    run_id = _get_run_id(req)
    ctx = LogContext(run_id=run_id, stage="gateway_infer", model="gateway")

    timeout_s = float(model_cfg.get("timeout_s", 30.0))
    retries = int(model_cfg.get("retries", 0))
    backoff_s = float(model_cfg.get("backoff_s", 0.1))
    models: Dict[str, Any] = model_cfg.get("models", {}) or {}

    log_event(
        logger,
        "gateway fanout start",
        ctx=ctx,
        batch_id=req.batch_id,
        n_items=len(req.items),
        timeout_s=timeout_s,
        retries=retries,
        backoff_s=backoff_s,
        targets=list(models.keys()),
    )

    calls: Dict[ModelName, Callable[[], ModelPredResponse]] = {}

    if "yolov11" in models and models["yolov11"].get("url"):
        calls["yolov11"] = lambda: infer_yolov11(
            _req_with_weight_scope(req, "yolov11"),
            url=models["yolov11"]["url"],
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
        )

    if "rtm" in models and models["rtm"].get("url"):
        calls["rtm"] = lambda: infer_rtm(
            _req_with_weight_scope(req, "rtm"),
            url=models["rtm"]["url"],
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
        )

    if "rtdetr" in models and models["rtdetr"].get("url"):
        calls["rtdetr"] = lambda: infer_rtdetr(
            _req_with_weight_scope(req, "rtdetr"),
            url=models["rtdetr"]["url"],
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
        )

    responses: Dict[ModelName, ModelPredResponse] = {}
    errors: Dict[ModelName, str] = {}
    model_elapsed_ms: Dict[str, int] = {}

    if not calls:
        log_event(
            logger,
            "gateway fanout skipped",
            ctx=ctx,
            batch_id=req.batch_id,
            reason="no model targets configured",
            models_cfg_keys=list(models.keys()),
        )
        return GatewayBundle(
            batch_id=req.batch_id,
            items=req.items,
            responses=responses,
            errors=errors,
        )

    with ThreadPoolExecutor(max_workers=len(calls)) as ex:
        future_map: Dict[Any, ModelName] = {}
        start_map: Dict[Any, float] = {}

        for name, fn in calls.items():
            fut = ex.submit(fn)
            future_map[fut] = name
            start_map[fut] = time.time()

        for fut in as_completed(future_map):
            target = future_map[fut]
            started = start_map.get(fut)
            elapsed_ms = int(((time.time() - started) if started else 0.0) * 1000)
            model_elapsed_ms[str(target)] = elapsed_ms

            try:
                r: ModelPredResponse = fut.result()

                if r.model != target:
                    log_event(
                        logger,
                        "model infer response model mismatch",
                        ctx=ctx,
                        batch_id=req.batch_id,
                        target_model=str(target),
                        response_model=str(r.model),
                    )

                responses[r.model] = r

                if not r.ok:
                    errors[r.model] = r.error or "unknown error"

                log_event(
                    logger,
                    "model infer response",
                    ctx=ctx,
                    batch_id=req.batch_id,
                    target_model=str(target),
                    ok=bool(r.ok),
                    elapsed_ms=elapsed_ms,
                    weight_used=_weight_used(r),
                    weight_dir_used=_weight_dir_used(r),
                    error=(r.error if not r.ok else None),
                )

            except Exception as e:
                errors[target] = str(e)
                logger.exception(
                    "model infer exception",
                    extra=ctx.as_extra()
                    | {
                        "batch_id": req.batch_id,
                        "target_model": str(target),
                        "elapsed_ms": elapsed_ms,
                        "error": str(e),
                    },
                )

    total_elapsed_ms = int((time.time() - t0) * 1000)

    models_ok: Dict[str, bool] = {}
    for m, resp in responses.items():
        try:
            models_ok[str(m)] = bool(resp.ok)
        except Exception:
            models_ok[str(m)] = False

    errors_for_log = {str(k): v for k, v in errors.items()}

    log_event(
        logger,
        "gateway fanout done",
        ctx=ctx,
        batch_id=req.batch_id,
        elapsed_ms=total_elapsed_ms,
        models_ok=models_ok,
        errors=errors_for_log,
        model_elapsed_ms=model_elapsed_ms,
    )

    return GatewayBundle(
        batch_id=req.batch_id,
        items=req.items,
        responses=responses,
        errors=errors,
    )
