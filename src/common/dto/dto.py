# V_2/src/common/dto/dto.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# -----------------------------------------------------------------------------
# Model / Verdict
# -----------------------------------------------------------------------------
# ✅ 전환기간(migration-safe): yolox + rtm 모두 허용
ModelName = Literal["yolov11", "rtdetr", "yolox", "rtm"]

# ✅ 이미지 단위 verdict
Verdict = Literal["PASS_3", "PASS_2", "FAIL", "MISS"]

# ✅ 객체 단위 레벨 (PASS/FAIL은 객체에도 쓰임)
ObjectLevel = Literal["PASS_3", "PASS_2", "FAIL"]

# ✅ FAIL 세부 타입 (회의 반영: 단일모델 + (2+) 불안정)
FailSubtype = Literal["FAIL_SINGLE", "FAIL_WEAK"]

# ✅ MISS 세부 타입
MissType = Literal["ALL_MISS", "DISAGREE_3MODELS"]


# -----------------------------------------------------------------------------
# Common identity (policy-aligned)
# -----------------------------------------------------------------------------
class RunIdentity(BaseModel):
    """
    사용자/프로젝트 단위 식별자
    - user_key: 이메일 자체가 아니라 안전한 키(slug/hash) 권장
    """

    user_key: str = Field(..., description="user email 기반 안전 키 (slug/hash)")
    project_id: str = Field(..., description="project id")
    run_id: Optional[str] = Field(None, description="(선택) 오케스트레이터 run id")
    job_id: Optional[str] = Field(None, description="(선택) job id (train/export 등)")

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# Input (Inference)
# -----------------------------------------------------------------------------
class InferItem(BaseModel):
    """
    단일 이미지 추론 입력 단위
    - 공통 표준 필드: image_path
    - 하위호환: path 로 들어와도 수용(alias)
    """

    image_id: str = Field(..., description="unlabeled 내에서 유일한 이미지 식별자(해시/경로기반 등)")
    image_path: str = Field(
        ...,
        alias="path",
        description="로컬/마운트 기준 이미지 경로 (legacy key: path도 허용)",
    )
    width: Optional[int] = Field(None, description="(선택) 원본 이미지 width")
    height: Optional[int] = Field(None, description="(선택) 원본 이미지 height")

    # Pydantic v2
    model_config = {"populate_by_name": True, "extra": "ignore"}


class InferBatchRequest(BaseModel):
    """
    오케스트레이터가 묶은 배치 추론 요청
    """

    batch_id: str = Field(..., description="오케스트레이터가 만든 배치 식별자")
    items: List[InferItem]
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# Standardized prediction schema (Judge/Ensemble reads THIS)
# -----------------------------------------------------------------------------
class Detection(BaseModel):
    """
    표준 Detection 한 개
    - xyxy: [x1, y1, x2, y2]
    """

    cls: int = Field(..., description="class id")
    conf: float = Field(..., description="confidence score")
    xyxy: List[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")

    model_config = {"extra": "ignore"}


class InferItemResult(BaseModel):
    """
    이미지 1장에 대한 표준 결과
    """

    image_id: str
    detections: List[Detection] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# Model response (gateway collects; judge consumes standardized fields if present)
# -----------------------------------------------------------------------------
class ModelPredResponse(BaseModel):
    """
    모델 1개의 배치 추론 응답

    핵심 원칙 (migration-safe):
    - 모델 컨테이너는 다양한 스키마를 낼 수 있다.
    - Gateway가 가능한 경우 `weight_used`/`results`를 추출해 채운다.
    - Judge는 `results`가 존재할 때만 표준 결과로 사용한다.
    """

    model: ModelName
    batch_id: str
    ok: bool = True
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None

    # ✅ 표준 필드(가능하면 gateway가 채움)
    weight_used: Optional[str] = Field(None, description="모델이 사용한 weight 경로/버전")
    results: Optional[List[InferItemResult]] = Field(
        None,
        description="표준화된 배치 결과 (없으면 gateway/judge가 raw에서 추출 시도 가능)",
    )

    # ✅ 모델별 원본 응답(그대로 보관)
    raw: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# Gateway bundle (no judging here)
# -----------------------------------------------------------------------------
class GatewayBundle(BaseModel):
    """
    3모델 응답을 묶은 번들
    - Judge 이전 단계: 판단/결합을 하지 않음
    """

    batch_id: str
    items: List[InferItem]

    responses: Dict[ModelName, ModelPredResponse] = Field(default_factory=dict)
    errors: Dict[ModelName, str] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# Judge outputs (UPDATED: objects / fail_singles / miss_evidence)
# -----------------------------------------------------------------------------
class ModelBox(BaseModel):
    """
    모델 단위 박스(근거/멤버)
    """

    cls: int
    conf: float
    xyxy: List[float] = Field(..., min_length=4, max_length=4)

    model_config = {"extra": "ignore"}


class FusedBox(BaseModel):
    """
    앙상블 fused 박스(대표)
    """

    xyxy: List[float] = Field(..., min_length=4, max_length=4)
    conf: float
    method: str = Field("conf_weighted_avg", description="fusion method name")

    model_config = {"extra": "ignore"}


class JudgeObjectRecord(BaseModel):
    """
    클러스터(객체 후보) 단위 레코드
    - PASS_3 / PASS_2 / FAIL(=FAIL_WEAK) 대상
    """

    object_index: int = Field(..., description="per-image object index (stable within this response)")
    cls: int = Field(..., description="final class id (cluster cls)")
    agree_count: int = Field(..., description="how many models agreed in this cluster")
    level: ObjectLevel = Field(..., description="PASS_3 | PASS_2 | FAIL")

    # FAIL일 때(=2+ agree지만 품질 미달)만 채우는 subtype
    fail_subtype: Optional[FailSubtype] = Field(
        None,
        description="FAIL subtype. For clustered FAIL, use FAIL_WEAK",
    )

    fused: Optional[FusedBox] = Field(
        None,
        description="fused representative box (PASS_3/PASS_2/FAIL_WEAK). For some policies, FAIL_WEAK can still have fused.",
    )

    # 멤버 박스(모델별)
    members: Dict[ModelName, ModelBox] = Field(default_factory=dict)

    # 디버깅/피처(코어가 만든 conf_stats, iou_spread, l1_spread 등)
    features: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class FailSingleRecord(BaseModel):
    """
    합의 클러스터에 포함되지 못한 단일 모델 박스(=FAIL_SINGLE)
    - UI에서 굵게 FAIL로 표시 대상
    """

    object_index: int
    level: Literal["FAIL"] = "FAIL"
    fail_subtype: Literal["FAIL_SINGLE"] = "FAIL_SINGLE"

    source_model: ModelName
    cls: int
    conf: float
    xyxy: List[float] = Field(..., min_length=4, max_length=4)

    model_config = {"extra": "ignore"}


class HintBox(BaseModel):
    """
    MISS에서 합의 실패 시 inspection hint로 쓰는 박스
    (절대 라벨/정답 박스가 아님)
    """

    type: Literal["union_box"] = "union_box"
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    inspection_only: bool = True
    source: Optional[str] = None

    model_config = {"extra": "ignore"}


class MissEvidence(BaseModel):
    """
    MISS 근거 데이터
    - raw_by_model: 모델별 '원본 박스' 전부(또는 정책상 raw)
    - hint_box: 3모델 모두 검출하지만 합의 불가일 때 union 힌트 박스
    """

    miss_type: MissType
    raw_by_model: Dict[ModelName, List[ModelBox]] = Field(default_factory=dict)
    hint_box: Optional[HintBox] = None

    model_config = {"extra": "ignore"}


class JudgeVerdictRecord(BaseModel):
    """
    이미지 단위 verdict 레코드 (UPDATED)
    - PASS_3/PASS_2: objects에 fused 포함
    - FAIL: objects(FAIL_WEAK) + fail_singles(FAIL_SINGLE) 가 존재할 수 있음
    - MISS: miss_evidence가 존재(모델별 박스 근거)
    """

    image_id: str
    batch_id: str
    verdict: Verdict

    model_ok: Dict[ModelName, bool] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = None

    # ✅ 합의 클러스터 기반 객체들 (PASS_3/PASS_2/FAIL_WEAK)
    objects: List[JudgeObjectRecord] = Field(default_factory=list)

    # ✅ 합의에 포함되지 못한 단일 모델 박스들 (FAIL_SINGLE)
    fail_singles: List[FailSingleRecord] = Field(default_factory=list)

    # ✅ MISS 근거
    miss_evidence: Optional[MissEvidence] = None

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# Attribution (fail / miss reasons)
# -----------------------------------------------------------------------------
class AttributionType(BaseModel):
    """
    FAIL/MISS 원인(사유) 레코드
    """

    type: Literal[
        "MODEL_MISS",
        "SINGLE_MODEL_DETECTION",
        "MODEL_LOW_CONF",
        "CLASS_CONFLICT",
        "LOC_DISAGREE",
        "ALL_MISS",
        "STRUCTURAL_ANOMALY",
        "MODEL_ERROR",
    ]
    model: Optional[ModelName] = None
    detail: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class JudgeAttributionRecord(BaseModel):
    """
    attribution 레코드
    - verdict는 FAIL/MISS 둘 다 가능
    """

    image_id: str
    batch_id: str
    verdict: Literal["FAIL", "MISS"]
    reasons: List[AttributionType] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# GT Training DTOs (policy-aligned)
# -----------------------------------------------------------------------------
class GTVersionRef(BaseModel):
    """
    GT 버전 선택용 참조 (경로 대신 버전 키만 전달)
    - 예: GT_1, GT_20260210_001
    - 서버는 versions_root/<gt_version>을 실제 루트로 해석한다.
    """

    gt_version: str = Field(
        ...,
        description="GT 버전 디렉토리명 (예: GT_1). 서버가 versions_root 아래에서 해석",
        examples=["GT_1"],
    )

    model_config = {"extra": "ignore"}


class GTDatasetSpec(BaseModel):
    """
    GT 학습 데이터셋 명세
    - 포맷/경로만 명시 (학습 파이프라인 구현은 모델 컨테이너 책임)
    """

    format: Literal["coco", "yolo"] = Field(
        ...,
        description="GT 데이터셋 포맷 (coco=json / yolo=labels txt)",
        examples=["yolo"],
    )
    img_root: str = Field(
        ...,
        description="images 루트 경로 (컨테이너 내부). 예: /workspace/data/images",
        examples=["/workspace/data/images"],
    )
    train: str = Field(
        ...,
        description="train annotation 경로 (COCO면 json, YOLO면 labels 디렉토리 등)",
        examples=["/workspace/data/labels/train"],
    )
    val: Optional[str] = Field(
        None,
        description="val annotation 경로 (옵션)",
        examples=["/workspace/data/labels/val"],
    )
    classes: Optional[str] = Field(
        None,
        description="(옵션) classes.json 경로 또는 클래스 매핑 파일 경로",
        examples=["/workspace/data/classes.json"],
    )

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "examples": [
                {
                    "format": "yolo",
                    "img_root": "/workspace/data/images",
                    "train": "/workspace/data/labels/train",
                    "val": "/workspace/data/labels/val",
                    "classes": "/workspace/data/classes.json",
                }
            ]
        },
    }


class GTTrainParams(BaseModel):
    """
    공통 학습 파라미터 (모델별 추가 파라미터는 extra로)
    """

    epochs: int = Field(30, ge=1, le=100000, description="학습 epoch 수", examples=[30])
    imgsz: int = Field(640, ge=32, le=4096, description="입력 해상도(imgsz)", examples=[640])
    batch: int = Field(8, ge=1, le=4096, description="학습 배치 크기", examples=[8])
    device: Optional[str] = Field(
        None, description="학습 디바이스. 예: '0', 'cuda:0', 'cpu'", examples=["0"]
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="모델별 추가 학습 파라미터(자유롭게 확장). 예: lr, workers, amp, etc.",
        examples=[{"lr": 0.0005, "workers": 4}],
    )

    model_config = {"extra": "ignore"}


class GTTrainRequest(BaseModel):
    """
    사용자 GT 학습 트리거 요청(모델별 공통 계약)

    ✅ v2 확장:
    - 경로 대신 gt.gt_version (예: "GT_1")만 주면 서버가 versions_root/<GT_1>로 dataset을 구성한다.
    - 하위호환: dataset(경로 직접 지정) 방식도 그대로 허용.
    - 둘 중 하나(gt 또는 dataset)는 반드시 제공되어야 한다.
    """

    identity: RunIdentity = Field(
        ...,
        description="요청 식별자(유저/프로젝트/job_id 등). 멀티테넌시/버전관리/로그 스코프에 사용",
    )

    model: ModelName = Field(
        ...,
        description="학습 대상 모델명 (yolov11 | rtm | rtdetr)",
        examples=["rtm"],
    )

    # ✅ NEW: 버전키로 GT 선택 (경로 노출 없이)
    gt: Optional[GTVersionRef] = Field(
        None,
        description="(옵션) GT 버전 선택. 제공 시 서버는 versions_root/<gt_version>을 GT 루트로 사용",
    )

    # ✅ 기존 방식(하위호환): 경로 직접 지정
    dataset: Optional[GTDatasetSpec] = Field(
        None,
        description="(옵션) GT 데이터셋 위치/포맷을 직접 지정(레거시/고급). "
        "gt가 제공되면 서버는 dataset을 무시하거나 보완할 수 있음.",
    )

    train: GTTrainParams = Field(
        default_factory=GTTrainParams,
        description="공통 학습 파라미터",
    )

    init_weight_type: Literal["baseline"] = Field(
        "baseline",
        description="초기 weight 선택 정책(현재 baseline 고정). baseline_weight_path가 있으면 그걸 우선 사용",
        examples=["baseline"],
    )

    baseline_weight_path: Optional[str] = Field(
        None,
        description="(옵션) 초기 weight 경로를 명시. 지정 시 해당 weight에서 파인튜닝 시작",
        examples=["/weights/_base/rtm/base_bdd100k.pth"],
    )

    # ✅ 필수조건: gt 또는 dataset 둘 중 하나는 있어야 함
    @model_validator(mode="after")
    def _validate_gt_or_dataset(self) -> "GTTrainRequest":
        if self.gt is None and self.dataset is None:
            raise ValueError("either 'gt' (gt_version) or 'dataset' must be provided")
        return self

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "examples": [
                # ✅ 권장: 버전키만 보내는 방식
                {
                    "identity": {"user_key": "user_001", "project_id": "project_demo", "job_id": "gt_0001"},
                    "model": "yolov11",
                    "gt": {"gt_version": "GT_1"},
                    "train": {"epochs": 30, "imgsz": 640, "batch": 8, "device": "0", "extra": {"lr": 0.0005}},
                    "init_weight_type": "baseline",
                },
                # ✅ 하위호환: 경로 직접 지정 방식
                {
                    "identity": {"user_key": "user_001", "project_id": "project_demo", "job_id": "gt_0002"},
                    "model": "rtm",
                    "dataset": {
                        "format": "yolo",
                        "img_root": "/workspace/data/images",
                        "train": "/workspace/data/labels/train",
                        "val": "/workspace/data/labels/val",
                        "classes": "/workspace/data/classes.json",
                    },
                    "train": {"epochs": 30, "imgsz": 640, "batch": 8, "device": "0", "extra": {"lr": 0.0005}},
                    "init_weight_type": "baseline",
                    "baseline_weight_path": "/weights/_base/rtm/base_bdd100k.pth",
                },
            ]
        },
    }


class GTTrainResponse(BaseModel):
    """
    GT 학습 결과 응답
    """

    identity: RunIdentity
    model: ModelName
    status: Literal["STARTED", "DONE", "FAILED"]
    detail: Optional[str] = None

    weight_saved: Optional[str] = Field(None, description="saved weight path")
    version: Optional[int] = Field(None, description="vN (per user_key/project_id/model)")

    artifacts: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# Export DTOs (policy-aligned)
# -----------------------------------------------------------------------------
class ExportRequest(BaseModel):
    """
    Approved 결과를 zip으로 내보내는 요청(컨트롤 플레인/백엔드용)
    """

    identity: RunIdentity
    export_version: Optional[int] = Field(None, description="(선택) export vK; 없으면 서버가 자동 증가")
    include_images: bool = Field(False, description="zip에 images 포함 여부")
    formats: List[Literal["yolo_txt", "coco_json"]] = Field(default_factory=lambda: ["yolo_txt", "coco_json"])

    model_config = {"extra": "ignore"}


# -----------------------------------------------------------------------------
# (추가) job 기반 GT Train/Export DTO (기존 유지)
# -----------------------------------------------------------------------------
class GTSourceLocal(BaseModel):
    root: str = Field(..., description="GT local root (inside container)")
    splits: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class GTSource(BaseModel):
    mode: str = "local"
    local: GTSourceLocal
    splits: Dict[str, Any] = Field(default_factory=dict)
    source_type: str = "GT"

    model_config = {"extra": "ignore"}


class TrainParams(BaseModel):
    epochs: int = 30
    imgsz: int = 640
    batch: int = 16
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    workers: int = 8
    seed: int = 42
    amp: bool = True

    model_config = {"extra": "ignore"}


class OutputParams(BaseModel):
    weights_dir_in_container: str = "/weights"
    naming: str = Field(..., description="baseline weight file name e.g. {model}/gt/{job_id}/best.pth")
    also_save_last: bool = True
    baseline: bool = True
    baseline_tag: str = ""

    model_config = {"extra": "ignore"}


class TrainGTRequest(BaseModel):
    job_id: str
    type: str = "GT_TRAIN"
    model: ModelName
    created_at: Optional[str] = None

    gt_source: "GTSource"
    train: "TrainParams"
    output: "OutputParams"

    job: Dict[str, Any] = Field(default_factory=dict)
    tracking: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class ExportResponse(BaseModel):
    """
    Export 결과 응답 (zip/manifest 경로 반환)
    """

    identity: RunIdentity
    status: Literal["STARTED", "DONE", "FAILED"]
    detail: Optional[str] = None

    export_version: Optional[int] = None
    zip_path: Optional[str] = None
    manifest_path: Optional[str] = None

    artifacts: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}
