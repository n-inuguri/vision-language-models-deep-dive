"""
Resumable background runner: doc_type x model x fixed 100-image sample.

Usage:
    uv run python poc/run_local_comparison.py                  # everything
    uv run python poc/run_local_comparison.py --doc-type pan    # one doc type
    uv run python poc/run_local_comparison.py --n 5             # smoke test

Writes one JSON line per (doc_type, model, image) to
poc/real_data_cache/{doc_type}__{model}.jsonl (model tag ':' replaced with '_'
for a filesystem-safe name). That file contains real extracted PII and is
gitignored -- never commit it.

Resumable by construction: on startup, already-cached (doc_type, model, image)
triples found in existing cache files are skipped, so Ctrl-C / a crashed
Ollama call / a laptop sleep only costs the in-flight call, not prior progress.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extraction_lib import MODELS, extract_fields, sample_images, FIELD_NAMES

CACHE_DIR = Path(__file__).parent / "real_data_cache"


def cache_path(doc_type, model_name):
    safe_model = model_name.replace(":", "_")
    return CACHE_DIR / f"{doc_type}__{safe_model}.jsonl"


def load_done_images(doc_type, model_name):
    path = cache_path(doc_type, model_name)
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["image"])
    return done


GT_DIR = Path(__file__).parent / "manual_ground_truth_cache"


def run(doc_types, n):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for doc_type in doc_types:
        # For aadhaar only: force already-labeled ground-truth images into the
        # sample, so manual labeling stays joinable with these extraction
        # results. Never done for pan -- would invalidate its cached progress.
        force_include = None
        if doc_type == "aadhaar":
            gt_path = GT_DIR / "aadhaar.json"
            if gt_path.exists():
                force_include = list(json.loads(gt_path.read_text(encoding="utf-8")).keys())

        images = sample_images(doc_type, n=n, force_include=force_include)
        print(f"\n=== {doc_type}: {len(images)} images x {len(MODELS)} models ===")

        for model_name in MODELS:
            done = load_done_images(doc_type, model_name)
            path = cache_path(doc_type, model_name)
            todo = [img for img in images if img.name not in done]

            print(f"[{doc_type}/{model_name}] {len(done)} already cached, {len(todo)} to run")
            if not todo:
                continue

            latencies = []
            with open(path, "a", encoding="utf-8") as f:
                for i, image_path in enumerate(todo, 1):
                    try:
                        fields, latency = extract_fields(image_path, model_name, doc_type)
                    except Exception as e:
                        print(f"  [{i}/{len(todo)}] {image_path.name}: ERROR {e}")
                        continue
                    latencies.append(latency)
                    record = {
                        "image": image_path.name,
                        "model": model_name,
                        "doc_type": doc_type,
                        "fields": fields,
                        "latency": latency,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    if i % 10 == 0 or i == len(todo):
                        avg = sum(latencies) / len(latencies)
                        print(f"  [{doc_type}/{model_name}] {i}/{len(todo)} done, avg latency {avg:.2f}s")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-type", choices=["pan", "aadhaar"], default=None,
                         help="Run only this doc type (default: both, pan first)")
    parser.add_argument("--n", type=int, default=100, help="Sample size per doc type")
    args = parser.parse_args()

    doc_types = [args.doc_type] if args.doc_type else ["pan", "aadhaar"]
    t0 = time.time()
    run(doc_types, args.n)
    print(f"Total wall time: {(time.time() - t0)/60:.1f} min")
