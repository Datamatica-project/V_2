# src/rtm/train_orchestrator.py
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Iterable


def load_recipe_module(recipe_py: Path):
    recipe_py = recipe_py.resolve()
    spec = importlib.util.spec_from_file_location("v2_recipe", str(recipe_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load recipe: {recipe_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def read_coco_categories(ann_path: Path) -> Tuple[int, Dict[int, str]]:
    d = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = d.get("categories", [])
    id2name: Dict[int, str] = {}
    for c in cats:
        if not isinstance(c, dict):
            continue
        if "id" not in c or "name" not in c:
            continue
        try:
            cid = int(c["id"])
            name = str(c["name"])
            id2name[cid] = name
        except Exception:
            continue
    return len(id2name), id2name


def _resolve_ann(data_root: Path, ann_file: str) -> Path:
    p = Path(ann_file)
    return p.resolve() if p.is_absolute() else (data_root / p).resolve()


def validate_recipe_vs_dataset(*, recipe_path: str, data_root: str, val_ann_file: str) -> Dict[str, Any]:
    """
    ✅ '레시피 자체'가 10클래스 고정인지 + '데이터셋 COCO categories'가 일치하는지 검증만 한다.
    (레시피 build_config()를 돌려서 cfg dict를 만들지 않는다)
    """
    mod = load_recipe_module(Path(recipe_path))

    if not hasattr(mod, "CLASSES"):
        raise ValueError("recipe missing CLASSES")
    classes = tuple(getattr(mod, "CLASSES"))
    if not isinstance(classes, tuple) and not isinstance(classes, list):
        raise ValueError("recipe.CLASSES must be tuple/list")

    classes = tuple(str(x) for x in classes)
    if len(classes) == 0:
        raise ValueError("recipe.CLASSES is empty")

    # COCO categories 검증
    data_root_p = Path(data_root).resolve()
    ann_abs = _resolve_ann(data_root_p, val_ann_file)

    ncat, id2name = read_coco_categories(ann_abs)

    if ncat != len(classes):
        raise ValueError(f"COCO categories({ncat}) != len(recipe.CLASSES)({len(classes)})")

    cat_names = set(id2name.values())
    class_names = set(classes)
    missing = class_names - cat_names
    if missing:
        raise ValueError(f"COCO categories missing class names: {sorted(missing)}")

    return {
        "recipe_path": str(Path(recipe_path).resolve()),
        "data_root": str(data_root_p),
        "val_ann_abs": str(ann_abs),
        "num_classes": len(classes),
        "classes": list(classes),
        "coco_categories": ncat,
    }


def build_cfg_options_from_yaml_and_request(
    *,
    train_yaml: Dict[str, Any],
    req: Dict[str, Any],
    data_root: str,
) -> Dict[str, Any]:
    """
    여기서 '하이퍼파라미터/리소스'를 cfg-options 키로 매핑해서 반환.
    - 이 mapping만 정확하면 train_gt.py는 그대로 실행기 역할만 하면 됨.
    """
    train = (train_yaml.get("train") or {})
    # request가 있으면 request 우선
    def pick(key: str, default=None):
        return req.get(key, train.get(key, default))

    # MMDet cfg-options 키로 매핑(중요: 실제 config 키에 맞춰야 함)
    # ✅ 최소 세트
    opts: Dict[str, Any] = {
        "train_cfg.max_epochs": int(pick("epochs", 30)),
        "train_dataloader.batch_size": int(pick("batch", 16)),
        "train_dataloader.num_workers": int(pick("workers", 4)),
        "val_dataloader.batch_size": int(pick("val_batch", pick("batch", 16))),
        "val_dataloader.num_workers": int(pick("val_workers", pick("workers", 4))),
        # optimizer
        "optim_wrapper.optimizer.lr": float(pick("lr", 0.0015)),
        "optim_wrapper.optimizer.weight_decay": float(pick("weight_decay", 0.05)),
    }

    # dataset 경로 (프로젝트마다 바뀌는 부분)
    # req에서 train_ann/val_ann/img_prefix를 받아서 주입하는 방식
    train_ann = req["train_ann_file"]
    val_ann = req.get("val_ann_file", train_ann)

    opts.update({
        "train_dataloader.dataset.data_root": str(data_root),
        "train_dataloader.dataset.ann_file": str(train_ann),
        "val_dataloader.dataset.data_root": str(data_root),
        "val_dataloader.dataset.ann_file": str(val_ann),
        "test_dataloader.dataset.data_root": str(data_root),
        "test_dataloader.dataset.ann_file": str(val_ann),
    })

    # img prefix
    train_img_prefix = req.get("train_img_prefix", "image/train/")
    val_img_prefix = req.get("val_img_prefix", "image/val/")
    opts.update({
        "train_dataloader.dataset.data_prefix.img": str(train_img_prefix),
        "val_dataloader.dataset.data_prefix.img": str(val_img_prefix),
        "test_dataloader.dataset.data_prefix.img": str(val_img_prefix),
    })

    # AMP는 CLI 플래그로 처리하는 게 보통이라 여기서는 생략해도 됨
    return opts
