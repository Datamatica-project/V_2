from __future__ import annotations

import importlib.util
import os
def _apply_device_env(device: str):
    # device: "cuda:2" / "2" / "cpu" 등 들어온다고 가정
    if device in ("cpu", "CPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return

    d = str(device).replace("cuda:", "")
    # "2" 또는 "0,1" 같은 형태만 남기기
    os.environ["CUDA_VISIBLE_DEVICES"] = d
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from mmengine.config import Config


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
            vv = repr(v)
        out.append(f"{k}={vv}")
    return out


def _is_recipe_file(p: Path) -> bool:
    """
    TODO: later - 명확한 flag/entrypoint로 식별(예: RECIPE_ID + build_config 필수).
    """
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return ("def build_config" in txt) and ("RECIPE_ID" in txt)


def _load_py_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    """
    - recipe 파일(.py + RECIPE_ID + build_config)이면: build_config(overrides) -> cfg dict 생성 -> generated_config.py로 dump -> mim train
    - 일반 mmdet config이면: --cfg-options 로 override 주입

    NOTE:
      imgsz는 현재 기본 파이프라인/레시피에 반영하지 않으면 효과가 없습니다.
    """
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

    merged_extra = dict(extra or {})

    # TODO: later - train/val split이 준비되면 아래 기본값을 제거하고,
    #              API에서 train_ann_file / val_ann_file / img_prefix를 명시 주입하도록 변경.
    merged_extra.setdefault("data_root", str(gt_root_p))
    merged_extra.setdefault("max_epochs", int(epochs))
    merged_extra.setdefault("train_cfg.max_epochs", int(epochs))
    merged_extra.setdefault("train_ann_file", "annotation/train.json")
    merged_extra.setdefault("val_ann_file", "annotation/val.json")
    merged_extra.setdefault("train_img_prefix", "images/")
    merged_extra.setdefault("val_img_prefix", "images/")

    merged_extra.setdefault("batch_size", 4)
    merged_extra.setdefault("num_workers", 4)
    merged_extra.setdefault("amp", True)

    # ✅ 레시피 모드에서 overrides로 받는 값들(핵심)
    #    - 레시피가 load_from/work_dir를 overrides에서 꺼내 cfg에 반영할 수 있게 보장
    merged_extra["load_from"] = str(init_ckpt_p)
    merged_extra["work_dir"] = str(work_dir)

    config_path_for_train = config_path
    used_mode = "mmdet_config"

    if _is_recipe_file(config_path):
        used_mode = "recipe_build_config"
        recipe = _load_py_module(config_path, "rtm_recipe_tmp")
        if not hasattr(recipe, "build_config"):
            raise RuntimeError(f"recipe has no build_config: {config_path}")

        cfg_dict = recipe.build_config(merged_extra)
        cfg_dict.setdefault("default_scope", "mmdet")

        gen_cfg_path = work_dir / "generated_config.py"
        Config(cfg_dict).dump(str(gen_cfg_path))
        config_path_for_train = gen_cfg_path

        # 레시피 모드에서는 cfg_dict에 load_from/work_dir가 들어가도록 했으므로,
        # 여기서 또 --cfg-options load_from=... 중복 주입은 피한다.
        cfg_opts: list[str] = []
    else:
        # 일반 config 모드는 --cfg-options로 주입
        cfg_opts = _cfg_options(merged_extra)

        # (안전장치) mmdet config에서 load_from를 확실히 override
        # merged_extra에 이미 넣어둔 load_from가 cfg_opts에 포함되지만,
        # 혹시 extra에서 load_from=None 등으로 지워질 가능성을 막기 위해 마지막에 한 번 더 보장
        cfg_opts = [opt for opt in cfg_opts if not opt.startswith("load_from=")]
        cfg_opts.append(f"load_from={str(init_ckpt_p)}")

    cmd = [
        "mim",
        "train",
        "mmdet",
        str(config_path_for_train),
        "--work-dir",
        str(work_dir),
        "--launcher",
        "none",
        "-G",
        "1",  # single GPU inside container
    ]

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
            "RTMDet train failed\n"
            f"MODE: {used_mode}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    latest_ckpt = work_dir / "latest.pth"
    if not latest_ckpt.exists():
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
        "config": str(config_path_for_train),
        "config_mode": used_mode,
        "epochs": epochs,
        "imgsz": imgsz,
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
