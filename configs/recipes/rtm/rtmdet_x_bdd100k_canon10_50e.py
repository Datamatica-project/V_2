# configs/recipes/rtm/rtmdet_x_bdd100k_canon10_50e.py
from __future__ import annotations
from pathlib import Path
RECIPE_ID = "rtm.rtmdet_x.bdd100k_canon10.50e"

CLASSES = (
    "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "pedestrian", "rider", "traffic_sign", "traffic_light",
)

def build_config(overrides: dict) -> dict:
    """
    MMDetection config dict를 반환.
    overrides는 서버가 주입하는 '프로젝트별 가변값'만 받는다.
    """

    data_root = overrides["data_root"]                  # 예: /root/workspace/Must/datasets/annotations/
    train_ann = overrides["train_ann_file"]             # 예: annotations/bdd100k_train_canon10_coco.json
    val_ann   = overrides["val_ann_file"]               # 예: annotations/bdd100k_val_canon10_coco.json
    train_img_prefix = overrides["train_img_prefix"]    # 예: image/train/
    val_img_prefix   = overrides["val_img_prefix"]      # 예: image/val/

    load_from = overrides.get("load_from")              # 예: /mnt/c/.../rtmdet_x_8xb32-300e_coco.pth
    work_dir  = overrides.get("work_dir")               # 예: /workspace/logs/model_train/...

    max_epochs = int(overrides.get("max_epochs", 50))
    stage2_num_epochs = int(overrides.get("stage2_num_epochs", 10))
    val_interval = int(overrides.get("val_interval", 10))

    batch_size = int(overrides.get("batch_size", 16))
    num_workers = int(overrides.get("num_workers", 4))
    amp = bool(overrides.get("amp", True))

    # ----- pipelines (필요하면 레시피에서 고정) -----
    train_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="CachedMosaic", img_scale=(640, 640), pad_val=114.0),
        dict(type="RandomResize", scale=(1280, 1280), ratio_range=(0.1, 2.0), keep_ratio=True),
        dict(type="RandomCrop", crop_size=(640, 640)),
        dict(type="YOLOXHSVRandomAug"),
        dict(type="RandomFlip", prob=0.5),
        dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        dict(type="CachedMixUp", img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=20, pad_val=(114, 114, 114)),
        dict(type="PackDetInputs"),
    ]

    test_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="Resize", scale=(640, 640), keep_ratio=True),
        dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="PackDetInputs", meta_keys=("img_id","img_path","ori_shape","img_shape","scale_factor")),
    ]

    cfg: dict = {}

    # ---- model (중요 고정값: num_classes=10) ----
    cfg["model"] = dict(
        type="RTMDet",
        data_preprocessor=dict(
            type="DetDataPreprocessor",
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False,
            batch_augments=None,
        ),
        backbone=dict(
            type="CSPNeXt",
            arch="P5",
            expand_ratio=0.5,
            deepen_factor=1.33,
            widen_factor=1.25,
            channel_attention=True,
            norm_cfg=dict(type="SyncBN"),
            act_cfg=dict(type="SiLU", inplace=True),
        ),
        neck=dict(
            type="CSPNeXtPAFPN",
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4,
            expand_ratio=0.5,
            norm_cfg=dict(type="SyncBN"),
            act_cfg=dict(type="SiLU", inplace=True),
        ),
        bbox_head=dict(
            type="RTMDetSepBNHead",
            num_classes=10,  # ✅ 사고 방지 핵심
            in_channels=320,
            stacked_convs=2,
            feat_channels=320,
            anchor_generator=dict(type="MlvlPointGenerator", offset=0, strides=[8,16,32]),
            bbox_coder=dict(type="DistancePointBBoxCoder"),
            loss_cls=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
            with_objectness=False,
            exp_on_reg=True,
            share_conv=True,
            pred_kernel_size=1,
            norm_cfg=dict(type="SyncBN"),
            act_cfg=dict(type="SiLU", inplace=True),
        ),
        train_cfg=dict(assigner=dict(type="DynamicSoftLabelAssigner", topk=13), allowed_border=-1, pos_weight=-1, debug=False),
        test_cfg=dict(
            nms_pre=30000, min_bbox_size=0, score_thr=0.001,
            nms=dict(type="nms", iou_threshold=0.65), max_per_img=300
        ),
    )

    # ---- dataloaders (가변: data_root/ann_file/img_prefix) ----
    metainfo = dict(classes=CLASSES)

    cfg["train_dataloader"] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        sampler=dict(type="DefaultSampler", shuffle=True),
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            ann_file=train_ann,
            data_prefix=dict(img=train_img_prefix),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            metainfo=metainfo,
            pipeline=train_pipeline,
            backend_args=None,
        ),
    )

    cfg["val_dataloader"] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            ann_file=val_ann,
            data_prefix=dict(img=val_img_prefix),
            test_mode=True,
            metainfo=metainfo,
            pipeline=test_pipeline,
            backend_args=None,
        ),
    )

    cfg["test_dataloader"] = cfg["val_dataloader"]

    # ---- evaluator (classwise=True 켜기) ----
    cfg["val_evaluator"] = dict(
        type="CocoMetric",
        ann_file=str(Path(data_root) / val_ann),
        metric="bbox",
        classwise=True,
        proposal_nums=(100, 1, 10),
    )
    cfg["test_evaluator"] = cfg["val_evaluator"]

    # ---- train schedule ----
    cfg["train_cfg"] = dict(
        type="EpochBasedTrainLoop",
        max_epochs=max_epochs,
        val_interval=val_interval,
        dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
    )
    cfg["val_cfg"] = dict(type="ValLoop")
    cfg["test_cfg"] = dict(type="TestLoop")

    # ---- optim/lr ----
    base_lr = float(overrides.get("base_lr", 0.004))
    cfg["optim_wrapper"] = (
        dict(
            type="AmpOptimWrapper",
            loss_scale="dynamic",
            optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
            paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
        ) if amp else
        dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
            paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
        )
    )
    cfg["param_scheduler"] = [
        dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=1000),
        dict(
            type="CosineAnnealingLR",
            eta_min=base_lr * 0.05,
            begin=max_epochs // 2,
            end=max_epochs,
            T_max=max_epochs // 2,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
    ]

    # ---- misc ----
    if load_from:
        cfg["load_from"] = load_from
    if work_dir:
        cfg["work_dir"] = work_dir

    return cfg
