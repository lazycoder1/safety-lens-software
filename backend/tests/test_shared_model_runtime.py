import threading

import numpy as np

import model_manager


def test_model_keys_with_the_same_asset_share_one_runtime():
    ppe_runtime = model_manager._MODEL_RUNTIMES["ppe_specialist"]
    long_tail_runtime = model_manager._MODEL_RUNTIMES["yoloe_long_tail"]

    assert ppe_runtime is long_tail_runtime
    assert ppe_runtime["lock"] is long_tail_runtime["lock"]
    assert ppe_runtime is not model_manager._MODEL_RUNTIMES["coco_primary"]


def test_shared_open_vocab_runtime_switches_default_prompts(monkeypatch):
    class FakeHandle:
        def __init__(self):
            self.devices = []
            self.predict_calls = 0

        def to(self, device):
            self.devices.append(device)

        def predict(self, *_args, **_kwargs):
            self.predict_calls += 1
            return []

    handle = FakeHandle()
    runtime = {
        "handle": handle,
        "lock": threading.Lock(),
        "loaded_path": "/tmp/shared-yoloe.pt",
        "current_classes": [],
        "class_embeddings": {},
        "warmed": True,
    }
    class_updates = []

    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "ppe_specialist", runtime)
    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "yoloe_long_tail", runtime)
    monkeypatch.setattr(
        model_manager,
        "_apply_open_vocab_classes",
        lambda active_runtime, _handle, classes, *, device: (
            class_updates.append(list(classes)),
            active_runtime.update(current_classes=list(classes)),
        ),
    )

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    model_manager.predict(
        "ppe_specialist",
        frame,
        conf=0.35,
        device="cpu",
        imgsz=640,
        classes=None,
    )
    model_manager.predict(
        "yoloe_long_tail",
        frame,
        conf=0.35,
        device="cpu",
        imgsz=640,
        classes=None,
    )

    assert class_updates == [
        list(model_manager.ALL_PPE_PROMPT_TERMS),
        list(model_manager._DEFAULT_LONG_TAIL_PROMPTS),
    ]
    assert handle.devices == []
    assert handle.predict_calls == 2


def test_cached_prompt_switch_does_not_move_the_model():
    class FakeHandle:
        def __init__(self):
            self.set_calls = []

        def set_classes(self, classes, embeddings):
            self.set_calls.append((list(classes), embeddings))

        def to(self, _device):
            raise AssertionError("cached prompt switch moved the model")

    embeddings = object()
    handle = FakeHandle()
    classes = ["fire", "smoke"]
    runtime = {
        "current_classes": ["person"],
        "class_embeddings": {tuple(classes): embeddings},
    }

    model_manager._apply_open_vocab_classes(runtime, handle, classes, device="cuda")

    assert handle.set_calls == [(classes, embeddings)]
    assert runtime["current_classes"] == classes


def test_uncached_prompt_switch_populates_cache_and_moves_once(monkeypatch):
    class FakeHandle:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)

    embeddings = object()
    handle = FakeHandle()
    classes = ["hard hat", "safety vest"]
    runtime = {"current_classes": [], "class_embeddings": {}}
    monkeypatch.setattr(
        model_manager,
        "_set_open_vocab_classes",
        lambda _handle, requested_classes: embeddings,
    )

    model_manager._apply_open_vocab_classes(runtime, handle, classes, device="cuda")

    assert runtime["class_embeddings"][tuple(classes)] is embeddings
    assert runtime["current_classes"] == classes
    assert handle.devices == ["cuda"]


def test_prompt_embedding_cache_is_bounded(monkeypatch):
    class FakeHandle:
        def to(self, _device):
            return None

    runtime = {"current_classes": [], "class_embeddings": {}}
    monkeypatch.setattr(
        model_manager,
        "_set_open_vocab_classes",
        lambda _handle, _classes: object(),
    )

    for index in range(model_manager._MAX_CLASS_EMBEDDING_CACHE_ENTRIES + 1):
        model_manager._apply_open_vocab_classes(
            runtime,
            FakeHandle(),
            [f"class-{index}"],
            device="cuda",
        )

    assert len(runtime["class_embeddings"]) == model_manager._MAX_CLASS_EMBEDDING_CACHE_ENTRIES
    assert ("class-0",) not in runtime["class_embeddings"]
    assert (f"class-{model_manager._MAX_CLASS_EMBEDDING_CACHE_ENTRIES}",) in runtime["class_embeddings"]
