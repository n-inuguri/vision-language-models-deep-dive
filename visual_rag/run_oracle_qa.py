"""
Experiment #1: no retrieval. For each question, the model is handed the
question's own known document directly (rag_lib.images_for_question) rather
than whatever ColPali would retrieve -- pure generation, no embeddings
involved. This is the ceiling measurement: how good is a model's reading of
a document when retrieval isn't a factor at all?

Contrast with run_visual_rag.py (Experiment #2): retrieve the best-matching
image(s) from the whole corpus first, then generate from *those*. Same
question sets, same generate_answer() dispatch, same resumable-JSONL
pattern -- the only difference is where the image(s) come from.

Runs in the MAIN venv (no colpali_engine involved at all -- doesn't need
.venv-colpali or a built index).

Usage:
    uv run python visual_rag/run_oracle_qa.py --dataset pan
    uv run python visual_rag/run_oracle_qa.py --dataset pan --cloud --n 15

Writes one JSON line per (question, model) to
visual_rag/oracle_cache/{dataset}__{model}.jsonl.
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


def load_questions(dataset):
    """Loads the dataset's Experiment #1 (no-retrieval) question set --
    see rag_lib.qa_path for the manual/dataset_provided and Exp #1 vs #2
    file routing."""
    path = rag_lib.qa_path(dataset, experiment="oracle")
    if not path.exists():
        raise SystemExit(f"No QA file at {path} -- author/prepare it first.")
    return json.loads(path.read_text())


def load_done_question_ids(dataset, model_name):
    path = rag_lib.oracle_cache_path(dataset, model_name)
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
    manifest = rag_lib.load_manifest(dataset)
    if not manifest:
        raise SystemExit(f"No page manifest for {dataset!r} -- build it first.")
    questions = load_questions(dataset)
    if n:
        questions = rag_lib.sample_questions(questions, n, seed=seed)
    by_qid = {q["question_id"]: q for q in questions}

    print(f"\n=== {dataset} (Experiment #1, no retrieval): {len(questions)} questions x {len(models)} models ===")

    for model_name in models:
        done = load_done_question_ids(dataset, model_name)
        path = rag_lib.oracle_cache_path(dataset, model_name)
        todo = [q for q in questions if q["question_id"] not in done]

        print(f"[{dataset}/{model_name}] {len(done)} already cached, {len(todo)} to run")
        if not todo:
            continue

        latencies = []
        with open(path, "a", encoding="utf-8") as f:
            for i, q in enumerate(todo, 1):
                qid = q["question_id"]
                image_paths = rag_lib.images_for_question(dataset, manifest, q)
                try:
                    answer, latency, usage = rag_lib.generate_answer(
                        image_paths, q["question"], model_name
                    )
                except Exception as e:
                    print(f"  [{i}/{len(todo)}] {qid}: ERROR {e}")
                    continue
                latencies.append(latency)
                record = {
                    "question_id": qid,
                    "dataset": dataset,
                    "model": model_name,
                    "question": q["question"],
                    "doc_id": q["doc_id"],
                    "image_paths": image_paths,
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
    parser.add_argument("--n", type=int, default=None, help="Subsample of questions (default: all)")
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
