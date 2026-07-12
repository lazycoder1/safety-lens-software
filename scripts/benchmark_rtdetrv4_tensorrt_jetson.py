#!/usr/bin/env python3
"""Benchmark a fixed-batch RT-DETRv4 TensorRT engine on Jetson frames."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bbox_coordinates(
    detection: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    bbox = detection.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coordinates):
        return None
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _phone_matches_person(phone: dict[str, Any], person: dict[str, Any]) -> bool:
    phone_bbox = _bbox_coordinates(phone)
    person_bbox = _bbox_coordinates(person)
    if phone_bbox is None or person_bbox is None:
        return False
    phone_x1, phone_y1, phone_x2, phone_y2 = phone_bbox
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1
    phone_center_x = (phone_x1 + phone_x2) / 2.0
    phone_center_y = (phone_y1 + phone_y2) / 2.0
    if not (
        person_x1 - person_width * 0.25
        <= phone_center_x
        <= person_x2 + person_width * 0.25
        and person_y1 - person_height * 0.10
        <= phone_center_y
        <= person_y2 + person_height * 0.10
    ):
        return False
    phone_area = (phone_x2 - phone_x1) * (phone_y2 - phone_y1)
    person_area = person_width * person_height
    if phone_area / person_area > 0.08:
        return False
    return phone_center_y <= person_y1 + person_height * 0.85


class TensorRTModel:
    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        self._bindings: list[torch.Tensor] = []
        self._names: list[str] = []
        self._input_indices: list[int] = []
        self._output_indices: list[int] = []
        for index in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(index)
            shape = tuple(self._engine.get_binding_shape(index))
            if any(dimension < 1 for dimension in shape):
                raise RuntimeError(
                    f"Only fixed-shape engines are supported: {name}={shape}"
                )
            numpy_dtype = trt.nptype(self._engine.get_binding_dtype(index))
            torch_dtype = torch.from_numpy(np.empty((), dtype=numpy_dtype)).dtype
            self._bindings.append(
                torch.empty(shape, dtype=torch_dtype, device="cuda:0")
            )
            self._names.append(name)
            target = (
                self._input_indices
                if self._engine.binding_is_input(index)
                else self._output_indices
            )
            target.append(index)
        self._name_to_index = {name: index for index, name in enumerate(self._names)}
        missing = {"images", "orig_target_sizes"} - self._name_to_index.keys()
        if missing:
            raise RuntimeError(f"TensorRT engine is missing inputs: {sorted(missing)}")

    @property
    def batch_size(self) -> int:
        return int(self._bindings[self._name_to_index["images"]].shape[0])

    @property
    def binding_contract(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "input": index in self._input_indices,
            }
            for index, (name, tensor) in enumerate(zip(self._names, self._bindings))
        ]

    def run(self, images, sizes, *, copy_outputs: bool) -> dict[str, Any]:
        self._bindings[self._name_to_index["images"]].copy_(images)
        self._bindings[self._name_to_index["orig_target_sizes"]].copy_(sizes)
        addresses = [int(tensor.data_ptr()) for tensor in self._bindings]
        if not self._context.execute_v2(addresses):
            raise RuntimeError("TensorRT execution failed")
        if not copy_outputs:
            return {}
        self._torch.cuda.synchronize()
        return {
            self._names[index]: self._bindings[index].cpu().numpy().copy()
            for index in self._output_indices
        }


def _prepare_group(model: TensorRTModel, frames: list[np.ndarray], start: int):
    import torch

    selected = [
        frames[(start + offset) % len(frames)] for offset in range(model.batch_size)
    ]
    images = np.stack(
        [
            cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)[..., ::-1]
            .transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
            for frame in selected
        ]
    )
    sizes = np.asarray([[frame.shape[1], frame.shape[0]] for frame in selected])
    image_binding = model._bindings[model._name_to_index["images"]]
    size_binding = model._bindings[model._name_to_index["orig_target_sizes"]]
    return (
        torch.from_numpy(np.ascontiguousarray(images)).to(
            device="cuda:0", dtype=image_binding.dtype
        ),
        torch.from_numpy(sizes).to(device="cuda:0", dtype=size_binding.dtype),
    )


def _summarize_outputs(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    person_conf: float,
    phone_conf: float,
) -> dict[str, Any]:
    detections = [
        {
            "class_id": int(class_id),
            "confidence": round(float(confidence), 4),
            "bbox": [round(float(value), 1) for value in bbox],
        }
        for class_id, bbox, confidence in zip(labels, boxes, scores)
        if (int(class_id) == PERSON_CLASS_ID and float(confidence) >= person_conf)
        or (int(class_id) == PHONE_CLASS_ID and float(confidence) >= phone_conf)
    ]
    people = [item for item in detections if item["class_id"] == PERSON_CLASS_ID]
    phones = [item for item in detections if item["class_id"] == PHONE_CLASS_ID]
    actionable = [
        phone
        for phone in phones
        if any(_phone_matches_person(phone, person) for person in people)
    ]
    return {
        "person_count": len(people),
        "phone_count": len(phones),
        "actionable_phone_count": len(actionable),
        "phone_scores": [item["confidence"] for item in phones],
        "actionable_phone_scores": [item["confidence"] for item in actionable],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--person-conf", type=float, default=0.30)
    parser.add_argument("--phone-conf", type=float, default=0.15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")

    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    frames = []
    for path in args.frames:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error(f"could not decode frame: {path}")
        frames.append(frame)

    model = TensorRTModel(args.engine)
    groups = [
        _prepare_group(model, frames, index * model.batch_size)
        for index in range(max(args.warmups, min(args.repeats, len(frames))))
    ]
    for index in range(args.warmups):
        images, sizes = groups[index % len(groups)]
        model.run(images, sizes, copy_outputs=False)
    torch.cuda.synchronize()

    model_latencies_ms = []
    total_started = time.perf_counter()
    for index in range(args.repeats):
        images, sizes = groups[index % len(groups)]
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        model.run(images, sizes, copy_outputs=False)
        finished.record()
        finished.synchronize()
        model_latencies_ms.append(float(started.elapsed_time(finished)))
    model_duration = time.perf_counter() - total_started

    end_to_end_latencies_ms = []
    end_to_end_started = time.perf_counter()
    for index in range(args.repeats):
        started = time.perf_counter()
        images, sizes = _prepare_group(model, frames, index * model.batch_size)
        model.run(images, sizes, copy_outputs=True)
        end_to_end_latencies_ms.append((time.perf_counter() - started) * 1000.0)
    end_to_end_duration = time.perf_counter() - end_to_end_started

    evaluation = []
    for start in range(0, len(frames), model.batch_size):
        images, sizes = _prepare_group(model, frames, start)
        outputs = model.run(images, sizes, copy_outputs=True)
        for offset in range(min(model.batch_size, len(frames) - start)):
            evaluation.append(
                {
                    "frame": args.frames[start + offset].name,
                    **_summarize_outputs(
                        outputs["labels"][offset],
                        outputs["boxes"][offset],
                        outputs["scores"][offset],
                        person_conf=args.person_conf,
                        phone_conf=args.phone_conf,
                    ),
                }
            )

    frames_processed = args.repeats * model.batch_size
    report = {
        "engine": args.engine.name,
        "batch_size": model.batch_size,
        "binding_contract": model.binding_contract,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "frames_processed": frames_processed,
        "model_only": {
            "throughput_fps": round(frames_processed / model_duration, 3),
            "group_latency_ms": {
                "median": round(statistics.median(model_latencies_ms), 3),
                "p95": round(_percentile(model_latencies_ms, 0.95), 3),
                "maximum": round(max(model_latencies_ms), 3),
            },
        },
        "end_to_end": {
            "throughput_fps": round(frames_processed / end_to_end_duration, 3),
            "group_latency_ms": {
                "median": round(statistics.median(end_to_end_latencies_ms), 3),
                "p95": round(_percentile(end_to_end_latencies_ms, 0.95), 3),
                "maximum": round(max(end_to_end_latencies_ms), 3),
            },
        },
        "evaluation": evaluation,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
