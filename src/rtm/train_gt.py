from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from typing import Any, Dict
import subprocess


def _device_to_visible(device: str) -> str:
    d = (device or "").strip().lower()
    if d.startswith("cuda:"):
        return d.split("cuda:")[-1]
    if d.isdigit():
        return d
    # cpu면 비워둠
    return ""


def _cfg_options(extra: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in (extra or {}).items():
        if v is None:
            continue
        if isinstance(v, bool):
            vv = "True" if v else "False"
        elif isinstance(v, (int, float)):
            vv = str(v)
        elif isinstance(v, str):
            vv = v
        else:
            vv = repr(v)   # ✅ dict/tuple/list 안전
        out.append(f"{k}={vv}")
    return out



def train_gt(
    *,
    config: str,
    gt_root: str,
    init_ckpt: str,
    out_ckpt: str,
    epochs: int,
    imgsz: int,
    device: str,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.time()

    config_path = Path(config).resolve()
    gt_root_p = Path(gt_root).resolve()
    init_ckpt_p = Path(init_ckpt).resolve()
    out_ckpt_p = Path(out_ckpt).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"RTM config not found: {config_path}")
    if not gt_root_p.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root_p}")
    if not init_ckpt_p.exists():
        raise FileNotFoundError(f"init_ckpt not found: {init_ckpt_p}")

    out_ckpt_p.parent.mkdir(parents=True, exist_ok=True)

    work_dir = out_ckpt_p.parent / f"_work_rtm_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ✅ cfg-options에 최소한의 주입(원하면 더 늘려도 됨)
    # - data_root를 config에서 쓰는 키로 맞춰야 함 (예시는 data_root)
    merged_extra = dict(extra or {})
    merged_extra.setdefault("data_root", str(gt_root_p))
    # epochs/imgsz도 config 키에 맞춰 override (프로젝트에서 쓰는 키로 맞춰)
    merged_extra.setdefault("train_cfg.max_epochs", int(epochs))

    cmd = [
        "python",
        "-m",
        "mmdetection.tools.train",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--load-from",
        str(init_ckpt_p),
    ]

    cfg_opts = _cfg_options(merged_extra)
    if cfg_opts:
        cmd += ["--cfg-options", *cfg_opts]

    env = os.environ.copy()
    vis = _device_to_visible(device)
    if vis != "":
        env["CUDA_VISIBLE_DEVICES"] = vis

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"RTMDet train failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    latest_ckpt = work_dir / "latest.pth"
    if not latest_ckpt.exists():
        # 일부 설정은 best.pth만 남길 수 있어서 fallback
        best_ckpt = work_dir / "best_bbox_mAP.pth"
        if best_ckpt.exists():
            latest_ckpt = best_ckpt
        else:
            raise FileNotFoundError(f"latest.pth not found in {work_dir}")

    shutil.copyfile(latest_ckpt, out_ckpt_p)

    elapsed = int((time.time() - t0) * 1000)
    return {
        "status": "DONE",
        "trained_weight": str(out_ckpt_p),
        "work_dir": str(work_dir),
        "elapsed_ms": elapsed,
        "init_ckpt": str(init_ckpt_p),
        "config": str(config_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "cmd": cmd,
    }
