# common/utils/logging.py
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# -----------------------------------------------------------------------------
# 기본 유틸
# -----------------------------------------------------------------------------
def utc_now_iso_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# JSON Formatter
# -----------------------------------------------------------------------------
_DEFAULT_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
}


class JsonLineFormatter(logging.Formatter):
    """
    JSON Lines formatter:
    - record.__dict__ 중 reserved를 제외한 extra 필드를 모두 포함
    """

    def __init__(
        self,
        service: str,
        host: Optional[str] = None,
        include_pid: bool = True,
        include_thread: bool = False,
    ) -> None:
        super().__init__()
        self.service = service
        self.host = host or socket.gethostname()
        self.include_pid = include_pid
        self.include_thread = include_thread

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": utc_now_iso_z(),
            "level": record.levelname,
            "service": self.service,
            "host": self.host,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if self.include_pid:
            payload["pid"] = record.process
        if self.include_thread:
            payload["thread"] = record.threadName

        extra: Dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k in _DEFAULT_RESERVED:
                continue
            try:
                json.dumps(v, ensure_ascii=False)
                extra[k] = v
            except Exception:
                extra[k] = repr(v)

        if extra:
            payload.update(extra)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LogContext:
    """
    V2 공통 컨텍스트 필드
    """

    job_id: Optional[str] = None
    run_id: Optional[str] = None
    model: Optional[str] = None
    stage: Optional[str] = None

    def as_extra(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.job_id:
            d["job_id"] = self.job_id
        if self.run_id:
            d["run_id"] = self.run_id
        if self.model:
            d["model"] = self.model
        if self.stage:
            d["stage"] = self.stage
        return d


def _norm_path(p: str | Path) -> str:
    return str(Path(p).expanduser().resolve())


def setup_logger(
    service: str,
    *,
    logger_name: Optional[str] = None,
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    propagate: bool = False,
    include_pid: bool = True,
    include_thread: bool = False,
) -> logging.Logger:
    """
    - service: 로그 payload에 기록될 service 이름 (예: "gateway", "judge", "yolov11")
    - logger_name: 파이썬 로거 이름. 모듈별 분리하고 싶으면 사용.
        예) service="gateway", logger_name="gateway.http_client"

    ✅ 여러 번 호출해도:
      - stdout handler는 중복 추가하지 않음
      - 같은 파일에 대한 file handler도 중복 추가하지 않음
    """
    name = (logger_name or service).strip()
    logger = logging.getLogger(name)

    # 매 호출마다 최신 설정 반영
    logger.setLevel(level if isinstance(level, int) else str(level).upper())
    logger.propagate = propagate

    formatter = JsonLineFormatter(
        service=service,
        include_pid=include_pid,
        include_thread=include_thread,
    )

    # stdout handler (중복 방지)
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "_v2_stdout", False):
            h.setLevel(logger.level)
            h.setFormatter(formatter)
            break
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logger.level)
        sh.setFormatter(formatter)
        setattr(sh, "_v2_stdout", True)
        logger.addHandler(sh)

    # file handler (중복 방지)
    if log_file:
        target = _norm_path(log_file)
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "_v2_file", False):
                try:
                    if _norm_path(h.baseFilename) == target:
                        h.setLevel(logger.level)
                        h.setFormatter(formatter)
                        break
                except Exception:
                    continue
        else:
            fp = Path(target)
            ensure_dir(fp.parent)
            fh = logging.FileHandler(fp, encoding="utf-8")
            fh.setLevel(logger.level)
            fh.setFormatter(formatter)
            setattr(fh, "_v2_file", True)
            logger.addHandler(fh)

    return logger


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    ctx: Optional[LogContext] = None,
    level: str = "INFO",
    **fields: Any,
) -> None:
    extra: Dict[str, Any] = {}
    if ctx:
        extra.update(ctx.as_extra())
    extra.update(fields)

    lvl = level.upper()
    if lvl == "DEBUG":
        logger.debug(message, extra=extra)
    elif lvl in ("WARNING", "WARN"):
        logger.warning(message, extra=extra)
    elif lvl == "ERROR":
        logger.error(message, extra=extra)
    elif lvl == "CRITICAL":
        logger.critical(message, extra=extra)
    else:
        logger.info(message, extra=extra)


class StageTimer:
    def __init__(
        self,
        logger: logging.Logger,
        stage_msg: str,
        *,
        ctx: Optional[LogContext] = None,
        level: str = "INFO",
        **fields: Any,
    ) -> None:
        self.logger = logger
        self.stage_msg = stage_msg
        self.ctx = ctx
        self.level = level
        self.fields = fields
        self._t0 = 0.0

    def __enter__(self) -> "StageTimer":
        self._t0 = time.time()
        log_event(self.logger, f"{self.stage_msg} (start)", ctx=self.ctx, level=self.level, **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = int((time.time() - self._t0) * 1000)
        if exc is None:
            log_event(
                self.logger,
                f"{self.stage_msg} (done)",
                ctx=self.ctx,
                level=self.level,
                elapsed_ms=elapsed_ms,
                **self.fields,
            )
            return False

        extra: Dict[str, Any] = {}
        if self.ctx:
            extra.update(self.ctx.as_extra())
        extra.update(self.fields)
        extra["elapsed_ms"] = elapsed_ms
        self.logger.exception(f"{self.stage_msg} (failed)", extra=extra)
        return False


def get_default_service_name(fallback: str = "v2") -> str:
    return os.getenv("MODEL_NAME", "").strip() or os.getenv("SERVICE_NAME", "").strip() or fallback
