"""Lightweight person tracking and phase-aware RT-DETR phone admission."""

from __future__ import annotations

import math
import threading
from typing import Any

import inference_scheduler


def _bbox(detection: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values = detection.get("bbox")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


class PrimaryPersonTracker:
    """Track stable primary-detector people with bounded state and no frame copies."""

    def __init__(self, *, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._tracks: list[dict[str, Any]] = []
        self._next_id = 1

    def update(
        self,
        detections: list[dict[str, Any]],
        *,
        now: float,
        ttl_seconds: float,
    ) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if now - float(track["last_seen"]) <= ttl_seconds
        ]
        people = [
            box
            for detection in detections
            if detection.get("class") == "person"
            and detection.get("model_family") == "coco_primary"
            and (box := _bbox(detection)) is not None
        ]
        unmatched_tracks = set(range(len(self._tracks)))
        for person_box in people:
            best_index = None
            best_iou = self.iou_threshold
            for index in unmatched_tracks:
                overlap = _iou(person_box, self._tracks[index]["bbox"])
                if overlap >= best_iou:
                    best_index = index
                    best_iou = overlap
            if best_index is None:
                self._tracks.append(
                    {
                        "id": self._next_id,
                        "bbox": person_box,
                        "hits": 1,
                        "last_seen": now,
                    }
                )
                self._next_id += 1
                continue
            track = self._tracks[best_index]
            track.update(
                bbox=person_box,
                hits=int(track["hits"]) + 1,
                last_seen=now,
            )
            unmatched_tracks.remove(best_index)

    def has_stable_person(
        self,
        *,
        now: float,
        min_hits: int,
        ttl_seconds: float,
    ) -> bool:
        return any(
            int(track["hits"]) >= min_hits
            and now - float(track["last_seen"]) <= ttl_seconds
            for track in self._tracks
        )

    def clear(self) -> None:
        self._tracks.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(track) for track in self._tracks]


class RTDETRPhoneSubstitutionScheduler:
    """Admit one phase-aligned pair per device interval with round-robin fairness."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._contexts: dict[str, float] = {}
        self._active_pair: set[str] = set()
        self._consumed: set[str] = set()
        self._active_expires_at = 0.0
        self._next_pair_at: float | None = None
        self._cursor = 0
        self._counters = {
            "selected_pairs": 0,
            "selected_singletons": 0,
            "selected_frames": 0,
            "missed_pair_windows": 0,
        }

    @staticmethod
    def _settings(cfg: dict) -> tuple[bool, float, float]:
        global_config = cfg.get("global") or {}
        enabled = global_config.get("rtdetr_phone_substitution_enabled") is True
        try:
            target_fps = float(global_config.get("rtdetr_phone_target_fps", 1.0))
            ttl_seconds = float(
                global_config.get("rtdetr_phone_person_track_ttl_seconds", 1.0)
            )
        except (TypeError, ValueError):
            return False, 1.0, 1.0
        if not math.isfinite(target_fps) or not 0.1 <= target_fps <= 1.0:
            return False, 1.0, 1.0
        if not math.isfinite(ttl_seconds) or not 0.25 <= ttl_seconds <= 5.0:
            return False, 1.0, 1.0
        return enabled, target_fps, ttl_seconds

    def consider(
        self,
        camera_id: str,
        cfg: dict,
        *,
        now: float,
        stable_person: bool,
    ) -> bool:
        enabled, target_fps, ttl_seconds = self._settings(cfg)
        with self._lock:
            if not enabled:
                self._contexts.clear()
                self._active_pair.clear()
                self._consumed.clear()
                self._next_pair_at = None
                return False
            if stable_person:
                self._contexts[camera_id] = now
            else:
                self._contexts.pop(camera_id, None)
            self._contexts = {
                tracked_camera: seen_at
                for tracked_camera, seen_at in self._contexts.items()
                if now - seen_at <= ttl_seconds
            }
            interval = 2.0 / target_fps
            if self._next_pair_at is None:
                # Warm context for one full interval before the first selection.
                enabled_camera_count = sum(
                    len(group) for group in inference_scheduler.camera_phase_groups(cfg)
                )
                self._next_pair_at = now + (
                    1.0 / target_fps if enabled_camera_count <= 2 else interval
                )
                return False
            if self._active_pair and now > self._active_expires_at:
                if self._consumed != self._active_pair:
                    self._counters["missed_pair_windows"] += 1
                self._active_pair.clear()
                self._consumed.clear()
            if not self._active_pair and now >= self._next_pair_at:
                pairs: list[tuple[str, str]] = []
                eligible = set(self._contexts)
                phase_groups = inference_scheduler.camera_phase_groups(cfg)
                for group in phase_groups:
                    group_eligible = [camera for camera in group if camera in eligible]
                    pairs.extend(
                        (group_eligible[index], group_eligible[index + 1])
                        for index in range(0, len(group_eligible) - 1, 2)
                    )
                if not pairs:
                    enabled_camera_count = sum(len(group) for group in phase_groups)
                    if enabled_camera_count <= 2 and eligible:
                        singleton = sorted(eligible)[self._cursor % len(eligible)]
                        self._cursor += 1
                        self._active_pair = {singleton}
                        self._consumed.clear()
                        self._active_expires_at = now + 0.5
                        self._next_pair_at = now + 1.0 / target_fps
                        self._counters["selected_singletons"] += 1
                    else:
                        self._next_pair_at = now + interval
                        self._counters["missed_pair_windows"] += 1
                        return False
                else:
                    self._next_pair_at = now + interval
                    pair = pairs[self._cursor % len(pairs)]
                    self._cursor += 1
                    self._active_pair = set(pair)
                    self._consumed.clear()
                    self._active_expires_at = now + 0.5
                    self._counters["selected_pairs"] += 1
            if camera_id not in self._active_pair or camera_id in self._consumed:
                return False
            self._consumed.add(camera_id)
            self._counters["selected_frames"] += 1
            if self._consumed == self._active_pair:
                self._active_pair.clear()
                self._consumed.clear()
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tracked_contexts": len(self._contexts),
                "active_pair_size": len(self._active_pair),
                **self._counters,
            }

    def reset(self) -> None:
        with self._lock:
            self._contexts.clear()
            self._active_pair.clear()
            self._consumed.clear()
            self._active_expires_at = 0.0
            self._next_pair_at = None
            self._cursor = 0
            for key in self._counters:
                self._counters[key] = 0


SUBSTITUTION_SCHEDULER = RTDETRPhoneSubstitutionScheduler()
