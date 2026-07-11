# Jetson TensorRT runtime

The COCO detector can use an explicitly configured, fixed-shape TensorRT engine on a Jetson. PyTorch remains the default and the automatic fallback.

## Build the engine

Build on the target Jetson in a one-off model-server container. It mounts the
same persistent `/app/models` volume but does not compete with the live server
process. TensorRT engines are coupled to the target GPU and software stack and
must not be committed to Git.

```bash
docker compose -f docker-compose.split.yml run --rm --no-deps model-server \
  python /app/scripts/export_tensorrt_engine.py \
  --source /app/models/coco_primary/yolo26s.pt \
  --output /app/models/coco_primary/yolo26s.engine \
  --imgsz 960 \
  --workspace 2
```

The export writes `yolo26s.engine` and `yolo26s.engine.json`. The sidecar records hashes for both the source model and engine, the fixed image size, precision, task, and TensorRT version. Export is deliberately offline because it can take several minutes and consume most of an 8 GB Jetson's memory.

Sites with mixed camera resolutions can also build a smaller engine without
replacing the primary engine:

```bash
docker compose -f docker-compose.split.yml run --rm --no-deps model-server \
  python /app/scripts/export_tensorrt_engine.py \
  --source /app/models/coco_primary/yolo26s.pt \
  --output /app/models/coco_primary/yolo26s-512.engine \
  --imgsz 512 \
  --workspace 2
```

Set `SAFETYLENS_COCO_LOW_RES_TENSORRT_ENGINE` to this artifact. SafetyLens uses
it only when the decoded frame's largest dimension is no greater than the
engine's fixed image size and that size is smaller than the requested inference
size or equal to it. Higher-resolution cameras continue to use the primary 960px engine. A
missing, invalid, unloadable, or failing optional engine falls through to the
primary runtime for the same request. Configured low-resolution engines are
validated, loaded, and warmed during model-server initialization so the first
camera inference does not pay engine deserialization and warm-up latency.

For a fixed-prompt YOLOE PPE engine, repeat `--class` in the exact order the camera plan uses and provide the MobileCLIP encoder:

```bash
docker compose -f docker-compose.split.yml run --rm --no-deps model-server \
  python /app/scripts/export_tensorrt_engine.py \
  --source /app/models/yoloe_open_vocab/yoloe-26s-seg.pt \
  --output /app/models/yoloe_open_vocab/yoloe-26s-seg-helmet.engine \
  --text-encoder /app/models/mobileclip2_b.ts \
  --class "motorcycle helmet" --class-group rider_helmet_required \
  --class "rider helmet" --class-group rider_helmet_required \
  --class "helmet" --class-group rider_helmet_required
```

Set `SAFETYLENS_PPE_TENSORRT_ENGINE` to enable it. PPE and dynamic long-tail YOLOE use separate runtimes when this is configured. If the requested PPE prompts differ from the engine manifest, SafetyLens releases the engine and falls back to the dynamic PyTorch model.

## Enable it

Set the absolute engine path for the model-server service and recreate that service:

```bash
export SAFETYLENS_COCO_TENSORRT_ENGINE=/app/models/coco_primary/yolo26s.engine
docker compose -f docker-compose.split.yml up -d --force-recreate model-server
```

At startup, SafetyLens verifies the sidecar, model hash, engine hash, task, and fixed image size before loading the engine. A missing, changed, mislabeled, or unloadable artifact is rejected and PyTorch is loaded. If TensorRT fails during inference, the same request is retried once after loading the source PyTorch model; runtime status exposes the active backend and fallback error.

Configured fixed-shape COCO and PPE TensorRT runtimes are also executed during
model-server initialization. Model readiness therefore includes engine
deserialization and warm-up instead of deferring multi-second cold work to the
first camera frame. A warm-up failure activates the same safe PyTorch fallback.

## Roll back

Unset `SAFETYLENS_COCO_TENSORRT_ENGINE` and recreate the model-server service. Keep the `.pt` source on disk even when TensorRT is enabled because it is the safe runtime fallback.
