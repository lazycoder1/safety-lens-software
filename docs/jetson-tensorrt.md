# Jetson TensorRT runtime

The COCO detector can use an explicitly configured, fixed-shape TensorRT engine on a Jetson. PyTorch remains the default and the automatic fallback.

## Build the engine

Build on the target Jetson, outside the live model-server process. TensorRT engines are coupled to the target GPU and software stack and should not be committed to Git.

```bash
python3 scripts/export_tensorrt_engine.py \
  --source /models/coco_primary/yolo26s.pt \
  --output /models/coco_primary/yolo26s.engine \
  --imgsz 960 \
  --workspace 2
```

The export writes `yolo26s.engine` and `yolo26s.engine.json`. The sidecar records hashes for both the source model and engine, the fixed image size, precision, task, and TensorRT version. Export is deliberately offline because it can take several minutes and consume most of an 8 GB Jetson's memory.

## Enable it

Set the absolute engine path for the model-server service and recreate that service:

```bash
SAFETYLENS_COCO_TENSORRT_ENGINE=/models/coco_primary/yolo26s.engine
docker compose -f docker-compose.split.yml up -d --force-recreate model-server
```

At startup, SafetyLens verifies the sidecar, model hash, engine hash, task, and fixed image size before loading the engine. A missing, changed, mislabeled, or unloadable artifact is rejected and PyTorch is loaded. If TensorRT fails during inference, the same request is retried once after loading the source PyTorch model; runtime status exposes the active backend and fallback error.

## Roll back

Unset `SAFETYLENS_COCO_TENSORRT_ENGINE` and recreate the model-server service. Keep the `.pt` source on disk even when TensorRT is enabled because it is the safe runtime fallback.
