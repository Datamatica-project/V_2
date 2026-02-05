# src/gateway/http_client.py
"""
http_client.py

gateway -> model service HTTP helper

- requests 기반
- 간단한 retry + backoff
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import requests

from ..common.utils.logging import setup_logger, LogContext, log_event

logger = setup_logger(
    service="gateway",
    logger_name="gateway.http_client",
    log_file="/workspace/logs/gateway/http_client.jsonl",
    level="INFO",
)


def _extract_trace(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    payload에서 run_id/batch_id/n_items 뽑기.
    규약:
      - batch_id: payload["batch_id"]
      - run_id: payload["params"]["run_id"] (있으면)
      - items: payload["items"]
    """
    run_id: Optional[str] = None
    batch_id: Optional[str] = None
    n_items: Optional[int] = None

    try:
        bid = payload.get("batch_id")
        batch_id = str(bid) if bid is not None else None

        items = payload.get("items")
        if isinstance(items, list):
            n_items = len(items)

        params = payload.get("params") or {}
        if isinstance(params, dict):
            rid = params.get("run_id")
            run_id = str(rid) if rid else None
    except Exception:
        pass

    return run_id, batch_id, n_items


def _safe_text_snippet(text: Any, limit: int = 512) -> str:
    try:
        s = str(text or "").strip()
    except Exception:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


def post_json(
    *,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    retries: int,
    backoff_s: float,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """POST JSON and return JSON.

    timeout_s: requests timeout (seconds)
    retries: number of retries on network/server error
    backoff_s: sleep between retries (linear)
    """
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)

    run_id, batch_id, n_items = _extract_trace(payload)
    ctx = LogContext(run_id=run_id, stage="gateway_http", model="gateway")

    last_err: Optional[Exception] = None
    t_all = time.time()

    max_attempts = max(1, int(retries) + 1)

    for attempt in range(max_attempts):
        t0 = time.time()
        try:
            log_event(
                logger,
                "http post attempt",
                ctx=ctx,
                url=url,
                batch_id=batch_id,
                n_items=n_items,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                timeout_s=timeout_s,
            )

            r = requests.post(url, json=payload, headers=hdrs, timeout=timeout_s)
            status_code = r.status_code
            r.raise_for_status()

            elapsed_ms = int((time.time() - t0) * 1000)

            try:
                data = r.json()
            except Exception:
                data = {"_raw_text": _safe_text_snippet(r.text)}

            log_event(
                logger,
                "http post success",
                ctx=ctx,
                url=url,
                batch_id=batch_id,
                status_code=status_code,
                elapsed_ms=elapsed_ms,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
            return data

        except requests.Timeout as e:
            last_err = e
            elapsed_ms = int((time.time() - t0) * 1000)
            log_event(
                logger,
                "http post timeout",
                ctx=ctx,
                url=url,
                batch_id=batch_id,
                elapsed_ms=elapsed_ms,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                level="WARNING",
            )

        except requests.HTTPError as e:
            last_err = e
            elapsed_ms = int((time.time() - t0) * 1000)
            resp = getattr(e, "response", None)
            status_code = getattr(resp, "status_code", None)
            body_snip = _safe_text_snippet(getattr(resp, "text", ""))

            # 4xx는 요청 자체 문제일 확률이 높아서 no-retry
            if isinstance(status_code, int) and 400 <= status_code < 500:
                logger.exception(
                    "http post http_error (no-retry on 4xx)",
                    extra=ctx.as_extra()
                    | {
                        "url": url,
                        "batch_id": batch_id,
                        "status_code": status_code,
                        "elapsed_ms": elapsed_ms,
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "body_snip": body_snip,
                    },
                )
                break

            log_event(
                logger,
                "http post http_error",
                ctx=ctx,
                url=url,
                batch_id=batch_id,
                status_code=status_code,
                elapsed_ms=elapsed_ms,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                level="WARNING",
                body_snip=body_snip,
            )

        except requests.RequestException as e:
            # ConnectionError 등 네트워크 계열
            last_err = e
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.exception(
                "http post request_exception",
                extra=ctx.as_extra()
                | {
                    "url": url,
                    "batch_id": batch_id,
                    "elapsed_ms": elapsed_ms,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                },
            )

        except Exception as e:
            last_err = e
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.exception(
                "http post exception",
                extra=ctx.as_extra()
                | {
                    "url": url,
                    "batch_id": batch_id,
                    "elapsed_ms": elapsed_ms,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                },
            )

        if attempt < max_attempts - 1:
            sleep_s = max(0.0, float(backoff_s)) * (attempt + 1)
            log_event(
                logger,
                "http retry sleep",
                ctx=ctx,
                url=url,
                batch_id=batch_id,
                sleep_s=sleep_s,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
            time.sleep(sleep_s)

    total_ms = int((time.time() - t_all) * 1000)
    log_event(
        logger,
        "http post failed",
        ctx=ctx,
        url=url,
        batch_id=batch_id,
        elapsed_ms=total_ms,
        attempts=max_attempts,
        error=str(last_err),
        level="ERROR",
    )

    raise RuntimeError(f"POST {url} failed after {max_attempts} attempts: {last_err}")
