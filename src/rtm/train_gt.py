from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from typing import Any, Dict
import subprocess
import importlib.util

# ✅ mmdet/mmengine가 설치되어 있다는 전제 (이미 tools.train을 쓰고 있으니 OK)
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
            vv = repr(v)   # ✅ dict/tuple/list 안전
        out.append(f"{k}={vv}")
    return out


def _is_recipe_file(p: Path) -> bool:
    """
    ✅ 레시피( build_config(overrides)->dict ) 파일인지 휴리스틱으로 판단.
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
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
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
    RTM GT training runner.

    - config: (1) 일반 mmdetection config(.py) 또는 (2) 레시피(build_config) 파일 모두 지원
    - extra: cfg-options(일반 config) 또는 overrides(레시피)로 사용

    ⚠️ TEMP TEST NOTE:
      아래에 "train=val 동일 COCO(json)" 기본값을 넣어놨다.
      나중에 정식 split을 쓰면 TODO 블록 제거하고, API에서 train_ann/val_ann을 명시 주입하도록 변경.
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

    # ✅ merged_extra는 "일반 config면 cfg-options", "레시피면 overrides"로 사용
    merged_extra = dict(extra or {})

    # ----------------------------
    # ✅ TEMP (테스트용) defaults
    # ----------------------------
    # TODO: later - train/val split이 준비되면 아래 기본값을 제거하고,
    #              API에서 train_ann_file / val_ann_file / img_prefix를 명시 주입하도록 변경.
    merged_extra.setdefault("data_root", str(gt_root_p))
    merged_extra.setdefault("max_epochs", int(epochs))         # 레시피용
    merged_extra.setdefault("train_cfg.max_epochs", int(epochs))  # 일반 config용(혹시 쓰는 경우 대비)

    # ✅ COCO 단일 json을 train=val로 쓰는 임시 테스트
    merged_extra.setdefault("train_ann_file", "annotations/instances_test_30.json")
    merged_extra.setdefault("val_ann_file", "annotations/instances_test_30.json")
    merged_extra.setdefault("train_img_prefix", "images/")
    merged_extra.setdefault("val_img_prefix", "images/")

    # 레시피가 batch_size/num_workers/amp를 받으므로 기본값 보정
    merged_extra.setdefault("batch_size", 4)
    merged_extra.setdefault("num_workers", 4)
    merged_extra.setdefault("amp", True)

    # ----------------------------
    # ✅ config/recipe handling
    # ----------------------------
    config_path_for_train = config_path
    used_mode = "mmdet_config"

    if _is_recipe_file(config_path):
        used_mode = "recipe_build_config"
        recipe = _load_py_module(config_path, "rtm_recipe_tmp")
        if not hasattr(recipe, "build_config"):
            raise RuntimeError(f"recipe has no build_config: {config_path}")

        # ✅ overrides -> cfg dict 생성
        cfg_dict = recipe.build_config(merged_extra)  # type: ignore[attr-defined]

        # ✅ 임시 config.py로 덤프 (mmdetection.tools.train이 읽을 수 있게)
        gen_cfg_path = work_dir / "generated_config.py"
        Config(cfg_dict).dump(str(gen_cfg_path))
        config_path_for_train = gen_cfg_path

        # 레시피 모드에서는 이미 overrides가 반영되었으므로 cfg-options는 굳이 필요 없음
        cfg_opts = []
    else:
        # 일반 mmdet config면 기존 방식 유지: cfg-options로 덮어쓰기
        cfg_opts = _cfg_options(merged_extra)

    cmd = [
        "python",
        "-m",
        "mmdetection.tools.train",
        str(config_path_for_train),
        "--work-dir",
        str(work_dir),
        "--load-from",
        str(init_ckpt_p),
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
            f"RTMDet train failed\nMODE: {used_mode}\nCMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
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
        "config": str(config_path_for_train),
        "config_mode": used_mode,
        "epochs": epochs,
        "imgsz": imgsz,
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
