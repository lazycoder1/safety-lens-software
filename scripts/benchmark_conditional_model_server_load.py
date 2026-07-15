#!/usr/bin/env python3
"""Replay primary-plus-conditional-specialist camera load against a model server."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


PPE_CLASSES = ["motorcycle helmet", "rider helmet", "helmet"]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | None]:
    """Return a stable latency/age summary without retaining samples in output."""
    if not values:
        return {
            "median": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "maximum": round(max(values), 3),
    }


def _jain_index(values: list[float]) -> float | None:
    """Return Jain's fairness index, or None when there is no service."""
    bounded = [max(0.0, float(value)) for value in values]
    if not bounded or not any(bounded):
        return None
    numerator = sum(bounded) ** 2
    denominator = len(bounded) * sum(value * value for value in bounded)
    return round(numerator / denominator, 6) if denominator else None


def _camera_fps_profile(
    raw_profile: str | None,
    *,
    cameras: int,
    default_fps: float,
) -> list[float]:
    """Expand a comma-separated camera-rate pattern across all cameras."""
    if not raw_profile:
        return [default_fps] * cameras
    try:
        pattern = [float(value.strip()) for value in raw_profile.split(",")]
    except ValueError as exc:
        raise ValueError("camera FPS profile must contain only numbers") from exc
    if not pattern or any(not math.isfinite(value) or value <= 0 for value in pattern):
        raise ValueError("camera FPS profile values must be positive and finite")
    return [pattern[index % len(pattern)] for index in range(cameras)]


def _maximum_gap_ms(
    completion_times: list[float],
    *,
    window_start: float,
    window_end: float,
) -> float:
    """Return the largest service gap, including both measurement edges."""
    if window_end <= window_start:
        return 0.0
    observed = sorted(
        value for value in completion_times if window_start <= value <= window_end
    )
    points = [window_start, *observed, window_end]
    return round(max(right - left for left, right in zip(points, points[1:])) * 1000, 3)


def _resize(frame: np.ndarray, maximum_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, maximum_dimension / max(height, width))
    if scale == 1.0:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(
        cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    )


def _phase_offset(
    camera_index: int,
    cameras: int,
    period: float,
    mode: str,
    group_size: int,
    remainder_weight: float = 1.0,
) -> float:
    if mode == "aligned":
        return 0.0
    if mode == "paired":
        groups = math.ceil(cameras / 2)
        return (camera_index // 2) * period / groups
    if mode == "grouped":
        groups = math.ceil(cameras / group_size)
        group_index = camera_index // group_size
        remainder = cameras % group_size
        if not remainder or remainder_weight == 1.0:
            return group_index * period / groups
        weights = [1.0] * groups
        weights[-1] = remainder_weight
        return period * sum(weights[:group_index]) / sum(weights)
    return camera_index * period / cameras


def _specialist_due(sequence: int, duty: float) -> bool:
    """Spread specialist work deterministically without random benchmark noise."""
    if duty <= 0:
        return False
    if duty >= 1:
        return True
    return math.floor((sequence + 1) * duty) > math.floor(sequence * duty)


def _phase_group_cardinality(
    camera_index: int,
    cameras: int,
    mode: str,
    group_size: int,
) -> int:
    if mode == "aligned":
        return cameras
    if mode == "paired":
        group_size = 2
    elif mode == "staggered":
        group_size = 1
    group_start = (camera_index // group_size) * group_size
    return min(group_size, cameras - group_start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--edge-url-override",
        help=(
            "Explicit model-server URL for isolated edge-transport benchmarks "
            "that do not mount the production camera configuration."
        ),
    )
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument("--cameras", type=int, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument(
        "--camera-fps-profile",
        help=(
            "Optional comma-separated quiet/uncertain/active FPS pattern. "
            "The pattern repeats when fewer rates than cameras are supplied."
        ),
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--stale-after-ms",
        type=float,
        default=0.0,
        help=(
            "Drop a scheduled frame before submit when it is already older "
            "than this threshold; zero preserves catch-up/FIFO behavior."
        ),
    )
    parser.add_argument(
        "--maximum-drain-seconds",
        type=float,
        default=15.0,
        help=(
            "Bound how long FIFO-like overdue work may drain after the "
            "measurement window before remaining slots are abandoned."
        ),
    )
    parser.add_argument("--specialist-duty", type=float, default=0.111)
    parser.add_argument(
        "--specialist-mode",
        choices=("additive", "substitute"),
        default="additive",
        help="Run due PPE work alongside primary or in place of that primary slot.",
    )
    parser.add_argument("--phone-probe-interval", type=float, default=1.0)
    parser.add_argument("--phone-probe-width", type=int, default=960)
    parser.add_argument(
        "--avoid-phone-specialist-overlap",
        action="store_true",
        help="Defer a due specialist pass by one frame when a phone probe is due.",
    )
    parser.add_argument(
        "--phone-context-duty",
        type=float,
        default=1.0,
        help="Fraction of scheduled phone-probe slots with primary person context.",
    )
    parser.add_argument("--max-inflight", type=int, default=2)
    parser.add_argument("--admission-timeout", type=float, default=0.065)
    parser.add_argument(
        "--start-at-monotonic",
        type=float,
        default=0.0,
        help="Optional shared host monotonic timestamp for synchronized workloads",
    )
    parser.add_argument(
        "--substitute-cameras",
        nargs="*",
        type=int,
        default=[],
        help="Camera indexes whose due primary slot is supplied by another detector",
    )
    parser.add_argument(
        "--substitute-duty",
        type=float,
        default=0.0,
        help="Fraction of primary slots replaced on --substitute-cameras",
    )
    parser.add_argument(
        "--substitute-sequence-offset",
        type=int,
        default=0,
        help=(
            "Shift the deterministic substitution cadence by this many camera "
            "sequences; use a positive offset to model deferred specialist work."
        ),
    )
    parser.add_argument(
        "--substitution-source",
        choices=("external", "rtdetr"),
        default="external",
        help="Supply substituted slots externally or through the edge RT-DETR route.",
    )
    parser.add_argument(
        "--transport",
        choices=("edge", "raw", "jpeg"),
        default="raw",
        help=(
            "Frame transport used for the grouped request. 'edge' exercises "
            "model_manager admission and its configured transport directly."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--phase-mode",
        choices=("aligned", "paired", "grouped", "staggered"),
        default="staggered",
    )
    parser.add_argument(
        "--phase-group-size",
        type=int,
        choices=(2, 4, 8),
        default=4,
        help="Camera arrival group used by --phase-mode grouped",
    )
    parser.add_argument(
        "--phase-remainder-hint",
        action="store_true",
        help="Pass the known phase-group cardinality to edge microbatch routing",
    )
    parser.add_argument(
        "--phase-remainder-weight",
        type=float,
        default=1.0,
        help=(
            "Relative interval reserved after a partial final camera group. "
            "Values below one move capacity to preceding full groups."
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cameras < 1 or args.fps <= 0 or args.duration <= 0:
        parser.error("cameras, fps, and duration must be positive")
    if not math.isfinite(args.stale_after_ms) or args.stale_after_ms < 0:
        parser.error("stale-after-ms must be finite and non-negative")
    if (
        not math.isfinite(args.maximum_drain_seconds)
        or args.maximum_drain_seconds < 0
        or args.maximum_drain_seconds > 120
    ):
        parser.error("maximum-drain-seconds must be between zero and 120")
    try:
        camera_fps = _camera_fps_profile(
            args.camera_fps_profile,
            cameras=args.cameras,
            default_fps=args.fps,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.specialist_duty <= 1:
        parser.error("specialist-duty must be between 0 and 1")
    if not 0 <= args.phone_context_duty <= 1:
        parser.error("phone-context-duty must be between 0 and 1")
    if not 160 <= args.phone_probe_width <= 1920:
        parser.error("phone-probe-width must be between 160 and 1920")
    if args.max_inflight < 1 or args.admission_timeout < 0:
        parser.error("max-inflight must be positive and admission-timeout non-negative")
    if args.start_at_monotonic < 0:
        parser.error("start-at-monotonic must be non-negative")
    if not 0 <= args.substitute_duty <= 1:
        parser.error("substitute-duty must be between 0 and 1")
    if not 0 <= args.substitute_sequence_offset <= 10_000:
        parser.error("substitute-sequence-offset must be between 0 and 10000")
    if len(set(args.substitute_cameras)) != len(args.substitute_cameras) or any(
        camera < 0 or camera >= args.cameras for camera in args.substitute_cameras
    ):
        parser.error("substitute-cameras must be unique valid camera indexes")
    if bool(args.substitute_cameras) != (args.substitute_duty > 0):
        parser.error(
            "substitute-cameras and a positive substitute-duty are required together"
        )
    if args.substitution_source == "rtdetr" and args.transport != "edge":
        parser.error("RT-DETR substitution requires --transport edge")
    if args.specialist_mode == "substitute" and args.transport != "edge":
        parser.error("PPE substitution requires --transport edge")
    if not 20 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be between 20 and 100")
    if not 0.1 <= args.phase_remainder_weight <= 1.0:
        parser.error("phase-remainder-weight must be between 0.1 and 1.0")

    frame_sets: list[dict[int, np.ndarray]] = []
    for path in args.frames:
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            parser.error(f"could not decode frame: {path}")
        frame_sets.append(
            {
                maximum_dimension: _resize(source, maximum_dimension)
                for maximum_dimension in {640, args.phone_probe_width}
            }
        )

    token = os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", "")
    edge_model_manager = None
    if args.transport == "edge":
        import model_manager

        edge_model_manager = model_manager
        if args.edge_url_override:
            override_url = args.edge_url_override.strip().rstrip("/")
            if not override_url.startswith(("http://", "https://")):
                parser.error("edge-url-override must be an HTTP(S) URL")
            edge_model_manager._remote_settings = lambda: {
                "enabled": True,
                "url": override_url,
                "token": token,
                "timeout_seconds": 30.0,
            }
        edge_settings = edge_model_manager._remote_settings()
        if not edge_settings.get("url"):
            parser.error("edge transport requires an enabled remote model server")
        edge_model_manager._REMOTE_JOB_ADMISSION_WAIT_SECONDS = args.admission_timeout
    admission = threading.BoundedSemaphore(args.max_inflight)
    barrier = threading.Barrier(args.cameras + 1)
    start_event = threading.Event()
    prewarm_failures: list[int] = []
    prewarm_lock = threading.Lock()
    phase_period = 1.0 / max(camera_fps)
    reports: list[dict] = [{} for _ in range(args.cameras)]
    substitution_cameras = set(args.substitute_cameras)

    def post(
        camera_index: int,
        sequence: int,
        phone_probe: bool,
        specialist: bool,
        frame_batch_size_hint: int | None,
    ) -> None:
        maximum_dimension = args.phone_probe_width if phone_probe else 640
        frame = frame_sets[camera_index % len(frame_sets)][maximum_dimension]
        batch = [
            {
                "request_id": f"coco-{camera_index}-{sequence}",
                "model_key": "coco_primary",
                "conf": 0.15 if phone_probe else 0.3,
                "device": "cuda",
                "imgsz": maximum_dimension,
                "classes": [],
            }
        ]
        if specialist:
            batch.append(
                {
                    "request_id": f"ppe-{camera_index}-{sequence}",
                    "model_key": "ppe_specialist",
                    "conf": 0.3,
                    "device": "cuda",
                    "imgsz": 640,
                    "classes": PPE_CLASSES,
                }
            )
        if edge_model_manager is not None:
            results = edge_model_manager.predict_record_batches(
                frame,
                batch,
                frame_batch_size_hint=frame_batch_size_hint,
            )
            expected = {item["request_id"] for item in batch}
            if set(results) != expected:
                raise RuntimeError("edge returned an incomplete grouped result")
            return
        batch_header = json.dumps(batch, separators=(",", ":"))
        if args.transport == "raw":
            height, width, channels = frame.shape
            body = frame.tobytes()
            endpoint = "/api/infer/raw/batch"
            headers = {
                "Content-Type": "application/octet-stream",
                "X-Rakshak-Inference-Batch": batch_header,
                "X-Rakshak-Frame-Width": str(width),
                "X-Rakshak-Frame-Height": str(height),
                "X-Rakshak-Frame-Channels": str(channels),
            }
        else:
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                raise RuntimeError("could not encode benchmark frame")
            body = encoded.tobytes()
            endpoint = "/api/infer/jpeg/batch"
            headers = {
                "Content-Type": "image/jpeg",
                "X-Rakshak-Inference-Batch": batch_header,
            }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{args.url.rstrip('/')}{endpoint}",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10.0) as response:
            if response.status != 200:
                raise RuntimeError(f"model server returned HTTP {response.status}")
            payload = json.load(response)
        expected = {item["request_id"] for item in batch}
        if set(payload.get("results") or {}) != expected:
            raise RuntimeError("model server returned an incomplete grouped result")

    def post_rtdetr(camera_index: int) -> None:
        if edge_model_manager is None:
            raise RuntimeError("RT-DETR substitution requires edge model transport")
        frame = frame_sets[camera_index % len(frame_sets)][640]
        records = edge_model_manager.predict_rtdetr_phone_records(
            frame,
            person_conf=0.3,
            phone_conf=0.15,
        )
        if not isinstance(records, list):
            raise RuntimeError("edge returned an invalid RT-DETR record batch")

    def post_ppe(
        camera_index: int,
        frame_batch_size_hint: int | None,
    ) -> None:
        if edge_model_manager is None:
            raise RuntimeError("PPE substitution requires edge model transport")
        frame = frame_sets[camera_index % len(frame_sets)][640]
        results = edge_model_manager.predict_record_batches(
            frame,
            [
                {
                    "request_id": "ppe_specialist",
                    "model_key": "ppe_specialist",
                    "conf": 0.3,
                    "device": "cuda",
                    "imgsz": 640,
                    "classes": PPE_CLASSES,
                }
            ],
            frame_batch_size_hint=frame_batch_size_hint,
        )
        if set(results) != {"ppe_specialist"}:
            raise RuntimeError("edge returned an incomplete PPE substitution result")

    def run_camera(camera_index: int) -> None:
        if edge_model_manager is not None:
            try:
                edge_model_manager._remote_get(
                    "/api/health",
                    timeout_seconds=2.0,
                )
            except Exception:
                with prewarm_lock:
                    prewarm_failures.append(camera_index)
        barrier.wait()
        start_event.wait()
        if camera_index in prewarm_failures:
            return
        started = benchmark_start + _phase_offset(
            camera_index,
            args.cameras,
            phase_period,
            args.phase_mode,
            args.phase_group_size,
            args.phase_remainder_weight,
        )
        camera_target_fps = camera_fps[camera_index]
        period = 1.0 / camera_target_fps
        probe_every = max(1, round(args.phone_probe_interval * camera_target_fps))
        deadline = benchmark_start + args.duration
        sequence = 0
        latencies: list[float] = []
        latencies_within_window: list[float] = []
        schedule_lateness_ms: list[float] = []
        frame_age_ms: list[float] = []
        frame_age_within_window_ms: list[float] = []
        stale_drop_age_ms: list[float] = []
        completion_times: list[float] = []
        primary_successes_within_window = 0
        effective_successes_within_window = 0
        completions_after_deadline = 0
        external_substitutions_unmeasured = 0
        abandoned_after_drain_deadline = 0
        stale_before_submit_drops = 0
        overloads = 0
        failures = 0
        specialist_requests = 0
        substituted_requests = 0
        substitution_attempts = 0
        rtdetr_latencies: list[float] = []
        ppe_substitution_latencies: list[float] = []
        specialist_deferred = False
        frame_batch_size_hint = (
            _phase_group_cardinality(
                camera_index,
                args.cameras,
                args.phase_mode,
                args.phase_group_size,
            )
            if args.phase_remainder_hint
            else None
        )
        scheduled_slots = max(0, math.ceil((deadline - started) / period - 1e-12))
        drain_deadline = deadline + args.maximum_drain_seconds

        def record_completion(
            completed: float,
            scheduled: float,
            *,
            primary: bool,
            frame_age_observed: bool = True,
        ) -> None:
            nonlocal primary_successes_within_window
            nonlocal effective_successes_within_window
            nonlocal completions_after_deadline
            completion_times.append(completed)
            within_window = completed <= deadline
            if within_window:
                effective_successes_within_window += 1
                primary_successes_within_window += int(primary)
            else:
                completions_after_deadline += 1
            if frame_age_observed:
                age = (completed - scheduled) * 1000.0
                frame_age_ms.append(age)
                if within_window:
                    frame_age_within_window_ms.append(age)

        while sequence < scheduled_slots:
            if time.monotonic() >= drain_deadline:
                abandoned_after_drain_deadline = scheduled_slots - sequence
                break
            scheduled = started + sequence * period
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            submit_ready = time.monotonic()
            scheduled_age_ms = max(0.0, (submit_ready - scheduled) * 1000.0)
            schedule_lateness_ms.append(scheduled_age_ms)
            if args.stale_after_ms and scheduled_age_ms > args.stale_after_ms:
                stale_before_submit_drops += 1
                stale_drop_age_ms.append(scheduled_age_ms)
                sequence += 1
                continue
            specialist = specialist_deferred or _specialist_due(
                sequence,
                args.specialist_duty,
            )
            phone_probe_slot = sequence % probe_every == 0
            phone_probe_index = sequence // probe_every
            phone_probe = phone_probe_slot and _specialist_due(
                phone_probe_index + camera_index,
                args.phone_context_duty,
            )
            if args.avoid_phone_specialist_overlap and phone_probe and specialist:
                specialist = False
                specialist_deferred = True
            elif specialist:
                specialist_deferred = False
            if specialist and args.specialist_mode == "substitute":
                request_started = time.perf_counter()
                try:
                    post_ppe(camera_index, frame_batch_size_hint)
                except Exception as exc:
                    if edge_model_manager is not None and isinstance(
                        exc,
                        edge_model_manager.RemoteInferenceOverloadedError,
                    ):
                        overloads += 1
                    else:
                        failures += 1
                else:
                    specialist_requests += 1
                    substituted_requests += 1
                    completed = time.monotonic()
                    record_completion(completed, scheduled, primary=False)
                    ppe_substitution_latencies.append(
                        (time.perf_counter() - request_started) * 1000.0
                    )
                sequence += 1
                continue
            substitute_primary = (
                camera_index in substitution_cameras
                and _specialist_due(
                    sequence + args.substitute_sequence_offset,
                    args.substitute_duty,
                )
            )
            if substitute_primary:
                # The external detector supplies this primary decision. Keep
                # PPE coverage by moving a coincident specialist pass forward
                # one camera frame instead of silently dropping it.
                specialist_deferred = specialist_deferred or specialist
                substitution_attempts += 1
                if args.substitution_source == "external":
                    substituted_requests += 1
                    completed = time.monotonic()
                    external_substitutions_unmeasured += 1
                    record_completion(
                        completed,
                        scheduled,
                        primary=False,
                        frame_age_observed=False,
                    )
                    sequence += 1
                    continue
                request_started = time.perf_counter()
                try:
                    post_rtdetr(camera_index)
                except Exception as exc:
                    if edge_model_manager is not None and isinstance(
                        exc,
                        edge_model_manager.RemoteInferenceOverloadedError,
                    ):
                        overloads += 1
                    else:
                        failures += 1
                else:
                    substituted_requests += 1
                    completed = time.monotonic()
                    record_completion(completed, scheduled, primary=False)
                    rtdetr_latencies.append(
                        (time.perf_counter() - request_started) * 1000.0
                    )
                sequence += 1
                continue
            admitted_externally = edge_model_manager is None
            if admitted_externally and not admission.acquire(
                timeout=args.admission_timeout
            ):
                overloads += 1
                sequence += 1
                continue
            request_started = time.perf_counter()
            try:
                post(
                    camera_index,
                    sequence,
                    phone_probe,
                    specialist,
                    frame_batch_size_hint,
                )
            except Exception as exc:
                if edge_model_manager is not None and isinstance(
                    exc,
                    edge_model_manager.RemoteInferenceOverloadedError,
                ):
                    overloads += 1
                else:
                    failures += 1
            else:
                completed = time.monotonic()
                request_latency = (time.perf_counter() - request_started) * 1000.0
                latencies.append(request_latency)
                if completed <= deadline:
                    latencies_within_window.append(request_latency)
                record_completion(completed, scheduled, primary=True)
                specialist_requests += int(specialist)
            finally:
                if admitted_externally:
                    admission.release()
            sequence += 1
        camera_finished = time.monotonic()
        reports[camera_index] = {
            "camera": camera_index,
            "target_fps": camera_target_fps,
            "scheduled": scheduled_slots,
            "processed_slots": sequence,
            "abandoned_after_drain_deadline": abandoned_after_drain_deadline,
            "successes": len(latencies),
            "successes_within_window": primary_successes_within_window,
            "specialist_requests": specialist_requests,
            "substituted_requests": substituted_requests,
            "external_substitutions_unmeasured": external_substitutions_unmeasured,
            "substitution_attempts": substitution_attempts,
            "effective_successes_within_window": effective_successes_within_window,
            "completions_after_deadline": completions_after_deadline,
            "overloads": overloads,
            "failures": failures,
            "stale_before_submit_drops": stale_before_submit_drops,
            "achieved_fps": round(
                primary_successes_within_window / args.duration,
                3,
            ),
            "effective_fps": round(
                effective_successes_within_window / args.duration,
                3,
            ),
            "service_ratio": round(
                effective_successes_within_window / max(1, scheduled_slots),
                6,
            ),
            "drain_elapsed_seconds": round(max(0.0, camera_finished - deadline), 3),
            "schedule_lateness_ms": _distribution(schedule_lateness_ms),
            "stale_drop_age_ms": _distribution(stale_drop_age_ms),
            "frame_age_ms": _distribution(frame_age_ms),
            "frame_age_within_window_ms": _distribution(
                frame_age_within_window_ms
            ),
            "maximum_measurement_service_gap_ms": _maximum_gap_ms(
                completion_times,
                window_start=started,
                window_end=deadline,
            ),
            "maximum_intercompletion_gap_ms": (
                round(
                    max(
                        right - left
                        for left, right in zip(completion_times, completion_times[1:])
                    )
                    * 1000.0,
                    3,
                )
                if len(completion_times) > 1
                else None
            ),
            "latencies_ms": latencies,
            "latencies_within_window_ms": latencies_within_window,
            "schedule_lateness_samples_ms": schedule_lateness_ms,
            "stale_drop_age_samples_ms": stale_drop_age_ms,
            "frame_age_samples_ms": frame_age_ms,
            "frame_age_within_window_samples_ms": frame_age_within_window_ms,
            "rtdetr_latencies_ms": rtdetr_latencies,
            "ppe_substitution_latencies_ms": ppe_substitution_latencies,
        }

    benchmark_start = 0.0
    threads = [
        threading.Thread(target=run_camera, args=(index,), daemon=True)
        for index in range(args.cameras)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    ready_at = time.monotonic()
    benchmark_start = args.start_at_monotonic or ready_at + 0.25
    if benchmark_start < ready_at + 0.05:
        raise RuntimeError("shared benchmark start is not far enough in the future")
    start_event.set()
    for thread in threads:
        thread.join(timeout=args.duration + args.maximum_drain_seconds + 5.0)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("camera load worker did not stop")
    benchmark_finished = time.monotonic()
    if prewarm_failures:
        raise RuntimeError(
            f"edge session prewarm failed for {len(prewarm_failures)} camera workers"
        )

    all_latencies = [
        latency for report in reports for latency in report.get("latencies_ms", [])
    ]
    all_latencies_within_window = [
        latency
        for report in reports
        for latency in report.get("latencies_within_window_ms", [])
    ]
    all_rtdetr_latencies = [
        latency
        for report in reports
        for latency in report.get("rtdetr_latencies_ms", [])
    ]
    all_ppe_substitution_latencies = [
        latency
        for report in reports
        for latency in report.get("ppe_substitution_latencies_ms", [])
    ]
    all_schedule_lateness = [
        latency
        for report in reports
        for latency in report.get("schedule_lateness_samples_ms", [])
    ]
    all_stale_drop_ages = [
        latency
        for report in reports
        for latency in report.get("stale_drop_age_samples_ms", [])
    ]
    all_frame_ages = [
        latency
        for report in reports
        for latency in report.get("frame_age_samples_ms", [])
    ]
    all_frame_ages_within_window = [
        latency
        for report in reports
        for latency in report.get("frame_age_within_window_samples_ms", [])
    ]
    for report in reports:
        report.pop("latencies_ms", None)
        report.pop("latencies_within_window_ms", None)
        report.pop("schedule_lateness_samples_ms", None)
        report.pop("stale_drop_age_samples_ms", None)
        report.pop("frame_age_samples_ms", None)
        report.pop("frame_age_within_window_samples_ms", None)
        report.pop("rtdetr_latencies_ms", None)
        report.pop("ppe_substitution_latencies_ms", None)
    achieved_rates = [report["achieved_fps"] for report in reports]
    effective_rates = [report["effective_fps"] for report in reports]
    service_ratios = [report["service_ratio"] for report in reports]
    effective_requests = len(all_latencies) + sum(
        report["substituted_requests"] for report in reports
    )
    effective_requests_within_window = sum(
        report["effective_successes_within_window"] for report in reports
    )
    benchmark_wall_elapsed = benchmark_finished - benchmark_start
    result = {
        "cameras": args.cameras,
        "target_fps": args.fps,
        "camera_fps_profile": camera_fps,
        "duration_seconds": args.duration,
        "benchmark_wall_elapsed_seconds": round(
            benchmark_wall_elapsed,
            3,
        ),
        "drain_elapsed_seconds": round(
            max(0.0, benchmark_finished - (benchmark_start + args.duration)),
            3,
        ),
        "maximum_drain_seconds": args.maximum_drain_seconds,
        "stale_after_ms": args.stale_after_ms or None,
        "specialist_duty_target": args.specialist_duty,
        "specialist_mode": args.specialist_mode,
        "specialist_requests": sum(report["specialist_requests"] for report in reports),
        "substitute_cameras": sorted(substitution_cameras),
        "substitute_duty_target": args.substitute_duty,
        "substitute_sequence_offset": args.substitute_sequence_offset,
        "substitution_source": args.substitution_source,
        "substitution_attempts": sum(
            report["substitution_attempts"] for report in reports
        ),
        "substituted_requests": sum(
            report["substituted_requests"] for report in reports
        ),
        "phone_context_duty_target": args.phone_context_duty,
        "phone_probe_width": args.phone_probe_width,
        "avoid_phone_specialist_overlap": args.avoid_phone_specialist_overlap,
        "admission_timeout_seconds": args.admission_timeout,
        "transport": args.transport,
        "jpeg_quality": args.jpeg_quality if args.transport == "jpeg" else None,
        "phase_mode": args.phase_mode,
        "phase_remainder_weight": args.phase_remainder_weight,
        "phase_remainder_hint": args.phase_remainder_hint,
        "edge_primary_batch": (
            edge_model_manager.remote_primary_batch_stats()
            if edge_model_manager is not None
            else None
        ),
        "edge_specialist_batch": (
            edge_model_manager.remote_specialist_batch_stats()
            if edge_model_manager is not None
            else None
        ),
        "edge_ppe_batch": (
            edge_model_manager.remote_ppe_batch_stats()
            if edge_model_manager is not None
            else None
        ),
        "edge_rtdetr_phone_batch": (
            edge_model_manager.remote_rtdetr_phone_batch_stats()
            if edge_model_manager is not None
            else None
        ),
        "requests": len(all_latencies),
        "requests_within_window": sum(
            report["successes_within_window"] for report in reports
        ),
        "effective_requests": effective_requests,
        "effective_requests_within_window": effective_requests_within_window,
        "aggregate_effective_fps": round(
            effective_requests_within_window / args.duration,
            3,
        ),
        "drained_effective_fps": round(
            effective_requests / max(benchmark_wall_elapsed, 1e-9),
            3,
        ),
        "completions_after_deadline": sum(
            report["completions_after_deadline"] for report in reports
        ),
        "abandoned_after_drain_deadline": sum(
            report["abandoned_after_drain_deadline"] for report in reports
        ),
        "overloads": sum(report["overloads"] for report in reports),
        "failures": sum(report["failures"] for report in reports),
        "stale_before_submit_drops": sum(
            report["stale_before_submit_drops"] for report in reports
        ),
        "minimum_camera_fps": min(report["achieved_fps"] for report in reports),
        "minimum_effective_camera_fps": min(
            report["effective_fps"] for report in reports
        ),
        "fairness": {
            "achieved_fps_jain": _jain_index(achieved_rates),
            "effective_fps_jain": _jain_index(effective_rates),
            "demand_normalized_jain": _jain_index(service_ratios),
        },
        "schedule_lateness_ms": _distribution(all_schedule_lateness),
        "stale_drop_age_ms": _distribution(all_stale_drop_ages),
        "frame_age_ms": _distribution(all_frame_ages),
        "frame_age_within_window_ms": _distribution(
            all_frame_ages_within_window
        ),
        "latency_ms": _distribution(all_latencies),
        "latency_within_window_ms": _distribution(all_latencies_within_window),
        "rtdetr_latency_ms": _distribution(all_rtdetr_latencies),
        "ppe_substitution_latency_ms": _distribution(
            all_ppe_substitution_latencies
        ),
        "per_camera": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
