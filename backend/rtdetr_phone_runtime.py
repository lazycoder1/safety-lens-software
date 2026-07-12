"""Optional fixed-batch RT-DETRv4 TensorRT phone-recall runtime."""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67
_SUPPORTED_BATCHES = (1, 2)
_ENGINE_ENV = {
    1: "SAFETYLENS_RTDETR_PHONE_BATCH1_TENSORRT_ENGINE",
    2: "SAFETYLENS_RTDETR_PHONE_BATCH2_TENSORRT_ENGINE",
}
_SHA_ENV = {
    1: "SAFETYLENS_RTDETR_PHONE_BATCH1_ENGINE_SHA256",
    2: "SAFETYLENS_RTDETR_PHONE_BATCH2_ENGINE_SHA256",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_BINDINGS = {
    "images": ((None, 3, 640, 640), np.dtype(np.float32), True),
    "orig_target_sizes": ((None, 2), np.dtype(np.int32), True),
    "scores": ((None, 300), np.dtype(np.float32), False),
    "labels": ((None, 300), np.dtype(np.int32), False),
    "boxes": ((None, 300, 4), np.dtype(np.float32), False),
}


class RTDETRPhoneRuntimeUnavailable(RuntimeError):
    """The optional engine is absent or failed its identity/runtime contract."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _class_aware_nms(
    records: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.7,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    ):
        if any(
            int(existing["class_id"]) == int(record["class_id"])
            and _bbox_iou(existing["bbox"], record["bbox"]) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(record)
    return kept


def records_from_outputs(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    person_conf: float,
    phone_conf: float,
) -> list[dict[str, Any]]:
    """Normalize person/phone queries into the existing COCO record contract."""
    records: list[dict[str, Any]] = []
    for raw_class_id, raw_bbox, raw_score in zip(labels, boxes, scores):
        class_id = int(raw_class_id)
        confidence = float(raw_score)
        threshold = (
            person_conf
            if class_id == PERSON_CLASS_ID
            else phone_conf
            if class_id == PHONE_CLASS_ID
            else None
        )
        if threshold is None or not math.isfinite(confidence) or confidence < threshold:
            continue
        coordinates = [float(value) for value in raw_bbox]
        if len(coordinates) != 4 or not all(
            math.isfinite(value) for value in coordinates
        ):
            continue
        x1, y1, x2, y2 = coordinates
        bbox = [
            min(frame_width - 1, max(0, int(round(x1)))),
            min(frame_height - 1, max(0, int(round(y1)))),
            min(frame_width, max(0, int(round(x2)))),
            min(frame_height, max(0, int(round(y2)))),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        records.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "bbox": bbox,
            }
        )
    return _class_aware_nms(records)


class _TensorRTRuntime:
    def __init__(self, engine_path: Path, batch_size: int) -> None:
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize the RT-DETR engine")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT could not create an RT-DETR execution context")

        bindings: list[Any] = []
        names: list[str] = []
        output_indices: list[int] = []
        contract: dict[str, tuple[tuple[int, ...], np.dtype, bool]] = {}
        for index in range(engine.num_bindings):
            name = engine.get_binding_name(index)
            shape = tuple(int(value) for value in engine.get_binding_shape(index))
            if any(value < 1 for value in shape):
                raise RuntimeError(f"RT-DETR requires fixed bindings: {name}={shape}")
            numpy_dtype = np.dtype(trt.nptype(engine.get_binding_dtype(index)))
            is_input = bool(engine.binding_is_input(index))
            contract[name] = (shape, numpy_dtype, is_input)
            torch_dtype = torch.from_numpy(np.empty((), dtype=numpy_dtype)).dtype
            bindings.append(torch.empty(shape, dtype=torch_dtype, device="cuda:0"))
            names.append(name)
            if not is_input:
                output_indices.append(index)

        if set(contract) != set(_EXPECTED_BINDINGS):
            raise RuntimeError(
                f"RT-DETR binding names do not match: {sorted(contract)}"
            )
        for name, (
            expected_shape,
            expected_dtype,
            expected_input,
        ) in _EXPECTED_BINDINGS.items():
            shape, dtype, is_input = contract[name]
            resolved_shape = (batch_size, *expected_shape[1:])
            if (
                shape != resolved_shape
                or dtype != expected_dtype
                or is_input != expected_input
            ):
                raise RuntimeError(
                    f"RT-DETR binding contract mismatch for {name}: "
                    f"shape={shape}, dtype={dtype}, input={is_input}"
                )

        self.batch_size = batch_size
        self._engine = engine
        self._runtime = runtime
        self._context = context
        self._bindings = bindings
        self._names = names
        self._name_to_index = {name: index for index, name in enumerate(names)}
        self._output_indices = output_indices

    def predict(
        self,
        frames: list[np.ndarray],
        *,
        person_conf: float,
        phone_conf: float,
    ) -> list[list[dict[str, Any]]]:
        if len(frames) != self.batch_size:
            raise ValueError(
                f"RT-DETR batch-{self.batch_size} requires {self.batch_size} frames"
            )
        images = np.stack(
            [
                cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)[..., ::-1]
                .transpose(2, 0, 1)
                .astype(np.float32)
                / 255.0
                for frame in frames
            ]
        )
        sizes = np.asarray(
            [[frame.shape[1], frame.shape[0]] for frame in frames],
            dtype=np.int32,
        )
        image_binding = self._bindings[self._name_to_index["images"]]
        size_binding = self._bindings[self._name_to_index["orig_target_sizes"]]
        image_binding.copy_(
            self._torch.from_numpy(np.ascontiguousarray(images)).to(
                device="cuda:0",
                dtype=image_binding.dtype,
            )
        )
        size_binding.copy_(
            self._torch.from_numpy(sizes).to(
                device="cuda:0",
                dtype=size_binding.dtype,
            )
        )
        addresses = [int(tensor.data_ptr()) for tensor in self._bindings]
        if not self._context.execute_v2(addresses):
            raise RuntimeError("RT-DETR TensorRT execution failed")
        self._torch.cuda.synchronize()
        outputs = {
            self._names[index]: self._bindings[index].cpu().numpy().copy()
            for index in self._output_indices
        }
        return [
            records_from_outputs(
                outputs["labels"][index],
                outputs["boxes"][index],
                outputs["scores"][index],
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                person_conf=person_conf,
                phone_conf=phone_conf,
            )
            for index, frame in enumerate(frames)
        ]


class RTDETRPhoneRuntimePool:
    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ
        self._slots = {
            batch_size: {
                "lock": threading.Lock(),
                "runtime": None,
                "engine_path": None,
                "engine_sha256": None,
                "backend": "unconfigured",
                "warmed": False,
                "error": None,
            }
            for batch_size in _SUPPORTED_BATCHES
        }

    def _configuration(self, batch_size: int) -> tuple[Path, str] | None:
        raw_path = str(self._environ.get(_ENGINE_ENV[batch_size], "")).strip()
        expected_sha = str(self._environ.get(_SHA_ENV[batch_size], "")).strip().lower()
        if not raw_path and not expected_sha:
            return None
        if not raw_path or not _SHA256_PATTERN.fullmatch(expected_sha):
            raise RTDETRPhoneRuntimeUnavailable(
                f"RT-DETR batch-{batch_size} requires an engine and exact SHA-256"
            )
        path = Path(raw_path)
        if path.suffix.lower() != ".engine" or not path.is_file():
            raise RTDETRPhoneRuntimeUnavailable(
                f"RT-DETR batch-{batch_size} engine is unavailable"
            )
        return path, expected_sha

    def _runtime(self, batch_size: int) -> _TensorRTRuntime:
        if batch_size not in self._slots:
            raise RTDETRPhoneRuntimeUnavailable(
                f"Unsupported RT-DETR phone batch size: {batch_size}"
            )
        slot = self._slots[batch_size]
        try:
            configuration = self._configuration(batch_size)
        except RTDETRPhoneRuntimeUnavailable as exc:
            slot.update(
                runtime=None,
                engine_path=None,
                engine_sha256=None,
                backend="failed",
                warmed=False,
                error=str(exc),
            )
            raise
        if configuration is None:
            slot.update(backend="unconfigured", warmed=False, error=None)
            raise RTDETRPhoneRuntimeUnavailable(
                f"RT-DETR batch-{batch_size} engine is not configured"
            )
        engine_path, expected_sha = configuration
        identity = (str(engine_path), expected_sha)
        if (
            slot.get("engine_path") == identity[0]
            and slot.get("engine_sha256") == identity[1]
        ):
            if slot.get("backend") == "failed":
                raise RTDETRPhoneRuntimeUnavailable(str(slot.get("error")))
            if slot.get("runtime") is not None:
                return slot["runtime"]

        slot.update(
            runtime=None,
            engine_path=identity[0],
            engine_sha256=identity[1],
            backend="loading",
            warmed=False,
            error=None,
        )
        actual_sha = _file_sha256(engine_path)
        if actual_sha != expected_sha:
            slot.update(backend="failed", error="RT-DETR engine SHA-256 mismatch")
            raise RTDETRPhoneRuntimeUnavailable(str(slot["error"]))
        try:
            runtime = _TensorRTRuntime(engine_path, batch_size)
        except Exception as exc:
            slot.update(backend="failed", error=f"RT-DETR load failed: {exc}")
            raise RTDETRPhoneRuntimeUnavailable(str(slot["error"])) from exc
        slot.update(runtime=runtime, backend="tensorrt", error=None)
        return runtime

    def predict(
        self,
        frames: list[np.ndarray],
        *,
        person_conf: float,
        phone_conf: float,
    ) -> list[list[dict[str, Any]]]:
        batch_size = len(frames)
        if batch_size not in self._slots:
            raise RTDETRPhoneRuntimeUnavailable(
                f"Unsupported RT-DETR phone batch size: {batch_size}"
            )
        if not 0.0 <= person_conf <= 1.0 or not 0.0 <= phone_conf <= 1.0:
            raise ValueError("RT-DETR confidence thresholds must be between 0 and 1")
        if any(
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            for frame in frames
        ):
            raise ValueError("RT-DETR frames must be non-empty BGR arrays")
        slot = self._slots[batch_size]
        with slot["lock"]:
            runtime = self._runtime(batch_size)
            try:
                results = runtime.predict(
                    frames,
                    person_conf=person_conf,
                    phone_conf=phone_conf,
                )
            except Exception as exc:
                slot.update(
                    backend="failed",
                    warmed=False,
                    error=f"RT-DETR inference failed: {exc}",
                )
                raise
            slot["warmed"] = True
            return results

    def warm_configured(self) -> None:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        for batch_size in _SUPPORTED_BATCHES:
            try:
                if self._configuration(batch_size) is not None:
                    self.predict(
                        [dummy] * batch_size,
                        person_conf=0.3,
                        phone_conf=0.15,
                    )
            except RTDETRPhoneRuntimeUnavailable as exc:
                slot = self._slots[batch_size]
                if slot.get("backend") == "unconfigured":
                    slot.update(backend="failed", error=str(exc))
                continue
            except Exception:
                # Status retains the bounded failure reason; the base model
                # server remains available when this optional route is bad.
                continue

    def status(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for batch_size, slot in self._slots.items():
            with slot["lock"]:
                configured = bool(
                    str(self._environ.get(_ENGINE_ENV[batch_size], "")).strip()
                    or str(self._environ.get(_SHA_ENV[batch_size], "")).strip()
                )
                result[str(batch_size)] = {
                    "configured": configured,
                    "backend": slot.get("backend"),
                    "warmed": bool(slot.get("warmed")),
                    "engine": Path(slot["engine_path"]).name
                    if slot.get("engine_path")
                    else None,
                    "error": slot.get("error"),
                }
        return result


RUNTIME_POOL = RTDETRPhoneRuntimePool()
