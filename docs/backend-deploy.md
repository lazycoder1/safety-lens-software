# Rakshak Lens Backend Deployment

This repo now includes a backend deployment helper:

```bash
cd /path/to/video-analytics
./scripts/deploy_backend.sh --install-systemd
```

What it does:

- creates or reuses `.venv`
- installs backend Python dependencies from `requirements.txt`
- creates `backend/config.json` from `backend/config.example.json` if needed
- validates that `backend/server.py` imports cleanly
- generates a systemd unit in `.deploy/`
- optionally installs and starts the `rakshak-lens-backend` service

## Deployment Split

For the hosted demo setup:

- `license-hub` deploys to Vercel from GitHub
- `video-analytics/frontend` deploys to Vercel from GitHub
- `video-analytics/backend` deploys to Vast over SSH from GitHub Actions
- demo videos, model weights, and `backend/config.json` are synced separately because they are intentionally not stored in git

## Backend Topology

The backend is the part that must stay close to the cameras and GPU. It owns:

- RTSP or file video ingestion
- YOLO / YOLOE inference
- MJPEG stream generation with bounding boxes already drawn on frames
- WebSocket alert fanout
- PostgreSQL access
- license validation

That means the backend should run on the edge box or GPU server, not on Vercel.

## Vercel Frontend

Yes, the frontend can be hosted on Vercel and still show live video with bounding
boxes, because the browser loads the annotated MJPEG stream directly from the
backend using:

- `GET /api/stream/:cameraId` for video
- `GET /ws/alerts` for live alerts

The important caveats are:

- set `VITE_API_URL` to the public backend base URL
- set `VITE_WS_URL` to the backend WebSocket URL
- serve the backend over `https://` and `wss://` if the frontend is on Vercel
- keep the backend reachable from the user browser; Vercel is not proxying the video stream for you

Example Vercel env vars:

```bash
VITE_API_URL=https://edge-api.example.com
VITE_WS_URL=wss://edge-api.example.com
```

Vercel setup for `video-analytics/frontend`:

- set the Vercel project root directory to `frontend`
- keep the framework as Vite
- the included [frontend/vercel.json](../frontend/vercel.json) serves static files first and then falls back to `index.html`, so direct links like `/live` and `/configure/cameras` keep working
- set the two env vars above in the Vercel project before production deploys

You can use [frontend/.env.example](../frontend/.env.example) as the local reference for those values.

## GitHub To Vast Backend Deploy

The repo now includes [.github/workflows/deploy-backend-vast.yml](../.github/workflows/deploy-backend-vast.yml).

That workflow:

- triggers on pushes to `master` for backend and deployment files
- connects to the Vast box over SSH
- rsyncs backend code to the remote project directory
- writes the backend `.env` file from a GitHub secret if provided
- runs `./scripts/deploy_backend.sh --install-systemd` on the host

Required GitHub repository secrets:

- `VAST_HOST`
- `VAST_PORT` (optional, defaults to `22`)
- `VAST_USER` (optional, defaults to `root`)
- `VAST_DEPLOY_PATH` (optional, defaults to `/opt/rakshak-lens/video-analytics`)
- `VAST_SSH_PRIVATE_KEY`
- `VIDEO_ANALYTICS_BACKEND_ENV` for the remote `.env` contents

## Syncing Demo Videos And Models

The backend GitHub deploy intentionally does not ship:

- `test-videos/`
- `models/`
- `backend/config.json`

Those files are gitignored and should stay out of the repo.

Use the local sync helper after the Vast instance is reachable:

```bash
cd /path/to/video-analytics
./scripts/sync_demo_assets.sh --host <vast-ip> --port <ssh-port>
```

By default it uploads:

- `backend/config.json`
- `test-videos/`
- `models/`

Add `--with-env` only if you also want to copy the local `.env` to the host instead of relying on the GitHub secret.

## Deployment Notes

- The backend entrypoint currently needs to run from `backend/`, so the service starts `uvicorn server:app` with `WorkingDirectory=.../backend`.
- The backend imports `bcrypt`, `jwt`, and file-upload support at runtime, so those packages are included in `requirements.txt`.
- If `models/coco_primary/yolo26n.pt` or `models/yoloe_open_vocab/yoloe-26s-seg.pt` are missing, the backend can still boot, but cameras that need those models stay paused until the models are installed.
- For a Vercel-hosted frontend, put Caddy or Nginx in front of the backend so the browser gets TLS and WebSockets on a stable public hostname.
- The current `test-videos/` folder is about `1.5 GB`, so keeping it out of Git and syncing it directly is the right shape for the demo backend.
