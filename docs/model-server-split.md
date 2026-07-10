# Rakshak Lens Edge / Model Server Split

Rakshak Lens can run as two processes:

- **Edge backend + frontend** on the client laptop. This process owns camera/video decoding, RTSP reconnects, config, alerting, MJPEG streams, WebSocket alerts, snapshots, and the React frontend.
- **Model server** on a GPU host. This process owns YOLO/YOLOE model loading and frame inference.

The frontend API does not change. The browser still talks to the edge backend at the same `/api/*`, `/api/stream/{camera_id}`, and `/ws/alerts` paths.

## Runtime Contract

1. The edge backend reads a frame from a file or RTSP camera with OpenCV.
2. The edge backend JPEG-encodes the frame and posts it to the model server.
3. The model server returns normalized detections:
   - `class_id`
   - `confidence`
   - `bbox`
4. The edge backend draws overlays, evaluates safety rules, creates alerts, stores snapshots, and streams MJPEG to the frontend.

## Start a GPU Model Server

```bash
cd /Users/gauthamgsabahit/workspace/techser/video-analytics
python -m uvicorn model_server:app --app-dir backend --host 0.0.0.0 --port 8100
```

Required shared-secret auth whenever the server is reachable outside a private
local Compose network:

```bash
export SAFETYLENS_MODEL_SERVER_TOKEN='change-me'
```

Health check:

```bash
curl http://MODEL_SERVER_HOST:8100/api/health
```

## Start the Lightweight Edge Backend

```bash
cd /Users/gauthamgsabahit/workspace/techser/video-analytics
export SAFETYLENS_MODEL_SERVER_URL='http://MODEL_SERVER_HOST:8100'
export SAFETYLENS_MODEL_SERVER_TOKEN='change-me'
python -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8000
```

`SAFETYLENS_MODEL_SERVER_URL` and `SAFETYLENS_MODEL_SERVER_TOKEN` are now first-boot defaults. Admins can change the active model server from **System Settings -> Model Server** without changing environment variables or rebuilding the edge container. If the saved model-server URL is empty or remote inference is disabled, the backend behaves like the original all-in-one process and loads models locally.

The admin setting accepts either a full URL or an IP/host with port:

```text
https://models.example.com
203.0.113.10:8100
```

## Docker Compose

For a local split stack:

```bash
docker compose -f docker-compose.split.yml up --build
```

The checked-in Compose file does not publish port `8100` on the host. The edge
reaches the model server over the internal network at
`http://model-server:8100`. Remote deployments must add an explicit port or
reverse-proxy override, set a non-empty token on both services, and restrict the
listener with the host firewall.

For a real sales laptop deployment, run only the `edge` service on the laptop and set the hosted GPU model server in the admin UI. `SAFETYLENS_MODEL_SERVER_URL` can pre-seed that value for first boot.

## Current Boundary

Remote inference is wired for YOLO/YOLOE object detection models used by live camera workers. Face recognition remains a local lazy-loaded worker path when enabled; keep face recognition disabled on CPU-only sales laptops unless it is moved behind the same model-server boundary.
