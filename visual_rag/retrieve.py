"""
Load a persisted ColQwen2 index (built by build_index.py) and run retrieval
(.search()-equivalent, via late-interaction MaxSim scoring) for a dataset's
sampled questions.

Runs under .venv-colpali, same as build_index.py (including the
transformers==5.8.0 pin documented there -- required for correct, non-corrupted
ColQwen2 embeddings):

    .venv-colpali/bin/python visual_rag/retrieve.py --dataset chartqa --top-k 5

Writes visual_rag/retrieval_cache/{dataset}.json:
    {question_id: {"top_k_doc_ids": [...], "top_k_image_paths": [...], "scores": [...]}}

run_visual_rag.py (main venv) reads this file to know which page(s) to hand
each answer-generation model for a given question -- retrieval and answer
generation are fully decoupled into separate script invocations so each can
run in the environment it actually needs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402


def _load_model():
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ColQwen2.from_pretrained(
        rag_lib.COLPALI_MODEL_NAME,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(rag_lib.COLPALI_MODEL_NAME)
    return model, processor


def load_questions(dataset):
    """Loads the dataset's Experiment #2 (retrieve-then-generate) question
    set via rag_lib.qa_path -- see there for the manual/dataset_provided and
    Exp #1 vs #2 file routing."""
    qa_path = rag_lib.qa_path(dataset, experiment="retrieval")
    if not qa_path.exists():
        raise SystemExit(f"No QA file at {qa_path} -- run prepare_{dataset}.py first.")
    return json.loads(qa_path.read_text())


def retrieve(dataset, top_k=5, n=None, seed=42):
    import torch

    manifest = rag_lib.load_manifest(dataset)
    if not manifest:
        raise SystemExit(f"No page manifest for {dataset!r} -- build it first.")

    embeddings_path = rag_lib.index_dir(dataset) / "embeddings.pt"
    if not embeddings_path.exists():
        raise SystemExit(f"No index for {dataset!r} -- run build_index.py first.")

    doc_embeddings = torch.load(embeddings_path)
    questions = load_questions(dataset)
    if n:
        questions = rag_lib.sample_questions(questions, n, seed=seed)

    model, processor = _load_model()

    results = {}
    for q in questions:
        qid = q["question_id"]
        batch = processor.process_queries([q["question"]]).to(model.device)
        with torch.no_grad():
            query_embedding = model(**batch)
        scores = processor.score_multi_vector(query_embedding.to("cpu"), doc_embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_k].tolist()
        results[qid] = {
            "question": q["question"],
            "top_k_doc_ids": [manifest[i]["doc_id"] for i in top_indices],
            "top_k_image_paths": [manifest[i]["image_path"] for i in top_indices],
            "scores": [scores[i].item() for i in top_indices],
        }
        print(f"  {qid}: top doc = {results[qid]['top_k_doc_ids'][0]} "
              f"(score {results[qid]['scores'][0]:.3f})")

    out_path = rag_lib.retrieval_cache_path(dataset)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{dataset}] retrieved for {len(results)} questions -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(rag_lib.DATASETS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n", type=int, default=None, help="Subsample of questions (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    retrieve(args.dataset, top_k=args.top_k, n=args.n, seed=args.seed)
