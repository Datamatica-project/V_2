# run_v2.ps1
$ErrorActionPreference = "Stop"

# 프로젝트 루트
$ROOT = (Get-Location).Path
$NET  = "v2-net"

Write-Host "ROOT: $ROOT"

# -----------------------------------------------------------------------------
# Network (없으면 생성)
# -----------------------------------------------------------------------------
try {
  docker network inspect $NET 1>$null 2>$null | Out-Null
  Write-Host "Network exists: $NET"
} catch {
  Write-Host "Creating network: $NET"
  docker network create $NET | Out-Null
}

# -----------------------------------------------------------------------------
# Cleanup containers (없어도 에러로 죽지 않게)
# -----------------------------------------------------------------------------
Write-Host "Removing old containers (if any)..."
try {
  docker rm -f v2-yolov11 v2-rtdetr v2-rtm v2-judge v2-minio v2-mlflow 1>$null 2>$null | Out-Null
} catch {
  # ignore
}

# -----------------------------------------------------------------------------
# Image pick helper
# -----------------------------------------------------------------------------
function Pick-Image([string]$hint, [string]$fallback) {
  $x = docker image ls --format "{{.Repository}}:{{.Tag}}" | Select-String $hint | Select-Object -First 1
  if ($x) { return $x.ToString().Trim() }
  return $fallback
}

$IMG_YOLO   = Pick-Image "v_2-yolov11" "v_2-yolov11:latest"
$IMG_RTDETR = Pick-Image "v_2-rtdetr"  "v_2-rtdetr:latest"
$IMG_RTM    = Pick-Image "v_2-rtm"     "v_2-rtm:latest"
$IMG_JUDGE  = Pick-Image "v_2-judge"   "v_2-judge:latest"

Write-Host "Using images:"
Write-Host "  YOLO  : $IMG_YOLO"
Write-Host "  RTDETR: $IMG_RTDETR"
Write-Host "  RTM   : $IMG_RTM"
Write-Host "  JUDGE : $IMG_JUDGE"
Write-Host ""

# -----------------------------------------------------------------------------
# 이미지 존재 체크 (없으면 안내만 하고 계속 진행하되, run에서 에러가 나면 그때 출력됨)
# -----------------------------------------------------------------------------
function Ensure-Image([string]$img, [string]$serviceName) {
  docker image inspect $img 1>$null 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Image not found: $img"
    Write-Host "  -> 먼저 빌드하세요: docker compose build $serviceName"
    Write-Host ""
  }
}
Ensure-Image $IMG_YOLO "yolov11"
Ensure-Image $IMG_RTDETR "rtdetr"
Ensure-Image $IMG_RTM "rtm"
Ensure-Image $IMG_JUDGE "judge"

# -----------------------------------------------------------------------------
# Volume strings (PowerShell ':' 파싱 방지 위해 ${ROOT} 사용)
# -----------------------------------------------------------------------------
$VOL_WORKSPACE = "${ROOT}:/workspace"
$VOL_WEIGHTS   = "${ROOT}\storage\weights:/weights"
$VOL_DATASETS  = "${ROOT}\storage\datasets:/datasets:ro"

$VOL_MINIO     = "${ROOT}\storage\minio:/data"
$VOL_MLFLOW    = "${ROOT}\storage\mlflow:/mlflow"

$VOL_ART_YOLO  = "${ROOT}\storage\artifacts\yolov11:/artifacts"
$VOL_ART_RTDE  = "${ROOT}\storage\artifacts\rtdetr:/artifacts"
$VOL_ART_RTM   = "${ROOT}\storage\artifacts\rtm:/artifacts"
$VOL_ART_JUDGE = "${ROOT}\storage\artifacts\judge:/artifacts"

$VOL_SCRIPT    = "${ROOT}\script:/script:ro"
$VOL_LOGS      = "${ROOT}\logs:/logs"
$VOL_CONFIGS   = "${ROOT}\configs:/configs:ro"

# -----------------------------------------------------------------------------
# Start minio (이미 돌고있어도 에러 안나게)
# -----------------------------------------------------------------------------
Write-Host "Starting minio..."
try {
  docker run -d --name v2-minio --network $NET `
    -p 9000:9000 -p 9001:9001 `
    -v "$VOL_MINIO" `
    -e MINIO_ROOT_USER=minioadmin `
    -e MINIO_ROOT_PASSWORD=minioadmin `
    minio/minio:latest server /data --console-address ":9001" | Out-Null
} catch {
  Write-Host "⚠ minio start skipped (maybe already running / port in use)."
}

# -----------------------------------------------------------------------------
# Start mlflow
# -----------------------------------------------------------------------------
Write-Host "Starting mlflow..."
try {
  docker run -d --name v2-mlflow --network $NET `
    -p 5000:5000 `
    -v "$VOL_MLFLOW" `
    -e MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db `
    -e MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://v2-artifacts `
    -e AWS_ACCESS_KEY_ID=minioadmin `
    -e AWS_SECRET_ACCESS_KEY=minioadmin `
    -e AWS_DEFAULT_REGION=us-east-1 `
    -e MLFLOW_S3_ENDPOINT_URL=http://v2-minio:9000 `
    ghcr.io/mlflow/mlflow:latest `
    mlflow server --host 0.0.0.0 --port 5000 `
    --backend-store-uri sqlite:////mlflow/mlflow.db `
    --default-artifact-root s3://v2-artifacts | Out-Null
} catch {
  Write-Host "⚠ mlflow start skipped (maybe already running / port in use)."
}

# -----------------------------------------------------------------------------
# yolov11 (GPU 0)
# -----------------------------------------------------------------------------
Write-Host "Starting yolov11 on GPU 0..."
docker run -d --name v2-yolov11 --network $NET --shm-size 8g `
  --gpus "device=0" `
  -p 8020:8020 `
  -v "$VOL_WORKSPACE" -v "$VOL_WEIGHTS" -v "$VOL_ART_YOLO" -v "$VOL_DATASETS" `
  -e MODEL_NAME=yolov11 `
  -e PYTHONPATH=/workspace `
  -e WEIGHTS_ROOT=/weights `
  -e ARTIFACTS_DIR=/artifacts `
  -e DEVICE=0 `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -e S3_ENDPOINT=http://v2-minio:9000 `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_DEFAULT_REGION=us-east-1 `
  $IMG_YOLO `
  uvicorn api.yolov11.server:app --app-dir /workspace --host 0.0.0.0 --port 8020 `
  --reload --reload-dir /workspace/api --reload-dir /workspace/src | Out-Null

# -----------------------------------------------------------------------------
# rtdetr (GPU 1)
# -----------------------------------------------------------------------------
Write-Host "Starting rtdetr on GPU 1..."
docker run -d --name v2-rtdetr --network $NET --shm-size 8g `
  --gpus "device=1" `
  -p 8021:8021 `
  -v "$VOL_WORKSPACE" -v "$VOL_WEIGHTS" -v "$VOL_ART_RTDE" -v "$VOL_DATASETS" `
  -e MODEL_NAME=rtdetr `
  -e PYTHONPATH=/workspace `
  -e WEIGHTS_ROOT=/weights `
  -e ARTIFACTS_DIR=/artifacts `
  -e DEVICE=0 `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -e S3_ENDPOINT=http://v2-minio:9000 `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_DEFAULT_REGION=us-east-1 `
  $IMG_RTDETR `
  uvicorn api.rtdetr.server:app --app-dir /workspace --host 0.0.0.0 --port 8021 `
  --reload --reload-dir /workspace/api --reload-dir /workspace/src | Out-Null

# -----------------------------------------------------------------------------
# rtm (GPU 2)
# -----------------------------------------------------------------------------
Write-Host "Starting rtm on GPU 2..."
docker run -d --name v2-rtm --network $NET --shm-size 8g `
  --gpus "device=2" `
  -p 8022:8022 `
  -v "$VOL_WORKSPACE" -v "$VOL_WEIGHTS" -v "$VOL_ART_RTM" -v "$VOL_DATASETS" `
  -e MODEL_NAME=rtm `
  -e PYTHONPATH=/workspace `
  -e WEIGHTS_ROOT=/weights `
  -e ARTIFACTS_DIR=/artifacts `
  -e RTM_CONFIG=/workspace/configs/recipes/rtm/rtmdet_x_bdd100k_canon10_50e.py `
  -e DEVICE=cuda:0 `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -e S3_ENDPOINT=http://v2-minio:9000 `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_DEFAULT_REGION=us-east-1 `
  $IMG_RTM `
  uvicorn api.rtm.server:app --app-dir /workspace --host 0.0.0.0 --port 8022 `
  --reload --reload-dir /workspace/api --reload-dir /workspace/src | Out-Null

# -----------------------------------------------------------------------------
# judge
# -----------------------------------------------------------------------------
Write-Host "Starting judge..."
docker run -d --name v2-judge --network $NET --shm-size 8g `
  -p 8025:8025 `
  -v "$VOL_WORKSPACE" -v "$VOL_SCRIPT" -v "$VOL_LOGS" -v "$VOL_CONFIGS" `
  -v "$VOL_ART_JUDGE" -v "$VOL_DATASETS" `
  -e PYTHONPATH=/workspace `
  -e PROJECT_ROOT=/workspace `
  -e CONFIGS_DIR=/configs `
  -e RUN_GT_TRAIN_SCRIPT=/script/tools/run_gt_train.py `
  -e PYTHON_BIN=python `
  -e YOLOV11_URL=http://v2-yolov11:8020/infer `
  -e RTDETR_URL=http://v2-rtdetr:8021/infer `
  -e RTM_URL=http://v2-rtm:8022/infer `
  -e S3_ENDPOINT=http://v2-minio:9000 `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_DEFAULT_REGION=us-east-1 `
  -e MLFLOW_TRACKING_URI=http://v2-mlflow:5000 `
  -e ARTIFACTS_DIR=/artifacts `
  $IMG_JUDGE `
  uvicorn api.judge.server:app --app-dir /workspace --host 0.0.0.0 --port 8025 `
  --reload --reload-dir /workspace/api --reload-dir /workspace/src --reload-dir /workspace/configs | Out-Null

Write-Host "`n✅ Started containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host ""
Write-Host "GPU check commands:"
Write-Host "  docker exec -it v2-yolov11 nvidia-smi -L"
Write-Host "  docker exec -it v2-rtdetr  nvidia-smi -L"
Write-Host "  docker exec -it v2-rtm     nvidia-smi -L"
