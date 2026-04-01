#!/usr/bin/env python3
"""
Benchmark qwen3-vl:8b and qwen3.5:35b via ollama API on video frames.
Measures: latency per frame, detection quality, token throughput.
"""
import base64
import json
import time
import sys
import random
import statistics
from pathlib import Path

import requests

OLLAMA_BASE = "http://localhost:11434"
FRAMES_DIR = Path(__file__).parent.parent / "dataset" / "frames"
RESULTS_DIR = Path(__file__).parent.parent / "docs" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FRAMES = 20  # frames to test per model

DETECTION_PROMPT = """Analyze this industrial workplace image. List all objects you can detect from these categories:
- person / worker
- hard hat / helmet (PPE)
- safety vest / high-vis jacket
- welding equipment / welding sparks
- machinery / lathe / industrial equipment
- electrical cables / electrical panels
- power tools / cutting tools

For each detected object, respond in JSON format:
{"detections": [{"class": "...", "confidence": "high/medium/low", "location": "brief description"}]}

If nothing from these categories is visible, return {"detections": []}"""


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def check_model(model: str) -> bool:
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/show",
            json={"name": model},
            timeout=10
        )
        return r.status_code == 200
    except Exception:
        return False


def query_vlm(model: str, image_path: Path) -> dict:
    img_b64 = image_to_base64(image_path)

    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": DETECTION_PROMPT,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                }
            },
            timeout=300
        )
        elapsed = time.time() - start

        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}", "latency_ms": round(elapsed * 1000)}

        data = r.json()
        response_text = data.get("response", "")
        tokens = data.get("eval_count", 0)
        prompt_tokens = data.get("prompt_eval_count", 0)

        # Parse detections from response
        detections = []
        try:
            # Find JSON in response
            text = response_text.strip()
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                parsed = json.loads(text[start_idx:end_idx])
                detections = parsed.get("detections", [])
        except Exception:
            pass

        return {
            "latency_ms": round(elapsed * 1000),
            "tokens_generated": tokens,
            "prompt_tokens": prompt_tokens,
            "tokens_per_second": round(tokens / elapsed, 1) if elapsed > 0 else 0,
            "detections": detections,
            "num_detections": len(detections),
            "raw_response": response_text[:500],
        }

    except requests.Timeout:
        return {"error": "timeout", "latency_ms": 300000}
    except Exception as e:
        return {"error": str(e), "latency_ms": 0}


def benchmark_model(model: str, frames: list[Path]) -> dict:
    print(f"\n{'='*50}")
    print(f"Benchmarking: {model}")
    print(f"Frames: {len(frames)}")
    print(f"{'='*50}")

    if not check_model(model):
        print(f"ERROR: Model '{model}' not available in ollama. Pull it first.")
        return {"error": f"model not found: {model}"}

    results = []
    for i, frame in enumerate(frames):
        print(f"  [{i+1}/{len(frames)}] {frame.name}...", end=" ", flush=True)
        result = query_vlm(model, frame)
        result["frame"] = frame.name
        results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"{result['latency_ms']}ms, {result['num_detections']} detections")

    # Aggregate stats (exclude errors)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"model": model, "error": "all requests failed", "raw": results}

    latencies = [r["latency_ms"] for r in valid]
    detections = [r["num_detections"] for r in valid]
    tps = [r["tokens_per_second"] for r in valid]

    summary = {
        "model": model,
        "frames_tested": len(frames),
        "successful": len(valid),
        "failed": len(results) - len(valid),
        "latency_ms": {
            "mean": round(statistics.mean(latencies)),
            "median": round(statistics.median(latencies)),
            "min": min(latencies),
            "max": max(latencies),
            "stdev": round(statistics.stdev(latencies)) if len(latencies) > 1 else 0,
        },
        "tokens_per_second": {
            "mean": round(statistics.mean(tps), 1),
            "min": round(min(tps), 1),
            "max": round(max(tps), 1),
        },
        "detections_per_frame": {
            "mean": round(statistics.mean(detections), 2),
            "total": sum(detections),
        },
        "fps_equivalent": round(1000 / statistics.mean(latencies), 3) if latencies else 0,
        "raw": results,
    }

    return summary


def main():
    # Get sample frames
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        print(f"No frames found in {FRAMES_DIR}. Run 01_extract_frames.py first.")
        sys.exit(1)

    random.seed(42)
    sample = random.sample(frames, min(SAMPLE_FRAMES, len(frames)))
    print(f"Selected {len(sample)} frames for benchmarking")

    models = [
        ("qwen3-vl:8b", "qwen3vl_results.json"),
        ("qwen3.5:35b", "qwen35_results.json"),
    ]

    all_results = {}
    for model_name, out_file in models:
        result = benchmark_model(model_name, sample)
        all_results[model_name] = result

        out_path = RESULTS_DIR / out_file
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved: {out_path}")

        # Summary printout
        if "error" not in result:
            print(f"\n--- {model_name} Summary ---")
            print(f"  Mean latency: {result['latency_ms']['mean']}ms")
            print(f"  Median latency: {result['latency_ms']['median']}ms")
            print(f"  Tokens/sec: {result['tokens_per_second']['mean']}")
            print(f"  Avg detections/frame: {result['detections_per_frame']['mean']}")
            print(f"  Effective FPS: {result['fps_equivalent']:.3f}")

    # Save combined
    combined_path = RESULTS_DIR / "vlm_benchmark_combined.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
