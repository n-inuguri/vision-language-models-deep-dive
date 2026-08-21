"""
Resumable runner: for each dataset's retrieved questions (from retrieve.py's
output), call generate_answer() across the configured answer models and
cache results. Runs in the MAIN project .venv (not .venv-colpali) -- it never
touches colpali_engine, only Ollama and OpenRouter.

Usage:
    uv run python visual_rag/run_visual_rag.py --dataset chartqa               # local models only
    uv run python visual_rag/run_visual_rag.py --dataset chartqa --cloud       # + cloud models
    uv run python visual_rag/run_visual_rag.py --dataset chartqa --n 5         # smoke test

Writes one JSON line per (question, model) to
visual_rag/rag_cache/{dataset}__{model}.jsonl (model tag ':'/'/' replaced with
'_' for a filesystem-safe name).

Resumable by construction: on startup, already-cached question_ids found in
an existing cache file are skipped, so Ctrl-C / a crashed API call / a laptop
sleep only costs the in-flight call, not prior progress. Mirrors
poc/run_local_comparison.py's resumability pattern.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def load_done_question_ids(dataset, model_name):
    path = rag_lib.rag_cache_path(dataset, model_name)
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["question_id"])
    return done


def run(dataset, models, n=None, seed=42):
    retrieval_path = rag_lib.retrieval_cache_path(dataset)
    if not retrieval_path.exists():
        raise SystemExit(
            f"No retrieval results for {dataset!r} at {retrieval_path} -- run "
            f".venv-colpali/bin/python visual_rag/retrieve.py --dataset {dataset} first."
        )
    retrieved = json.loads(retrieval_path.read_text())
    question_ids = list(retrieved)
    if n:
        question_ids = rag_lib.sample_questions(question_ids, n, seed=seed)

    print(f"\n=== {dataset}: {len(question_ids)} questions x {len(models)} models ===")

    for model_name in models:
        done = load_done_question_ids(dataset, model_name)
        path = rag_lib.rag_cache_path(dataset, model_name)
        todo = [qid for qid in question_ids if qid not in done]

        print(f"[{dataset}/{model_name}] {len(done)} already cached, {len(todo)} to run")
        if not todo:
            continue

        latencies = []
        with open(path, "a", encoding="utf-8") as f:
            for i, qid in enumerate(todo, 1):
                entry = retrieved[qid]
                try:
                    answer, latency, usage = rag_lib.generate_answer(
                        entry["top_k_image_paths"], entry["question"], model_name
                    )
                except Exception as e:
                    print(f"  [{i}/{len(todo)}] {qid}: ERROR {e}")
                    continue
                latencies.append(latency)
                record = {
                    "question_id": qid,
                    "dataset": dataset,
                    "model": model_name,
                    "question": entry["question"],
                    "retrieved_doc_ids": entry["top_k_doc_ids"],
                    "answer": answer,
                    "latency": latency,
                    "usage": usage,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    avg = sum(latencies) / len(latencies)
                    print(f"  [{dataset}/{model_name}] {i}/{len(todo)} done, avg latency {avg:.2f}s")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(rag_lib.DATASETS))
    parser.add_argument("--cloud", action="store_true", help="Also run the OpenRouter cloud models")
    parser.add_argument("--n", type=int, default=None, help="Subsample of questions (default: all retrieved)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_models = [m for m in rag_lib.ANSWER_MODELS if rag_lib.MODEL_PROVIDER[m] == "ollama"]
    cloud_models = [m for m in rag_lib.ANSWER_MODELS if rag_lib.MODEL_PROVIDER[m] == "openrouter"]
    models = local_models + cloud_models if args.cloud else local_models

    if args.cloud and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set (needed for --cloud). "
            "Copy .env.example to .env and fill it in."
        )

    t0 = time.time()
    run(args.dataset, models, n=args.n, seed=args.seed)
    print(f"Total wall time: {(time.time() - t0)/60:.1f} min")
