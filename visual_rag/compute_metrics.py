"""
Consolidates every dataset's results into one summary, covering BOTH
experiments side by side:
  - Experiment #1 (oracle_cache/, run_oracle_qa.py): no retrieval, the
    model is handed the question's own document directly. A ceiling --
    isolates generation quality from retrieval quality.
  - Experiment #2 (rag_cache/, run_visual_rag.py): retrieve the
    best-matching image(s) via ColPali first, then generate from those.

Same question sets, same accuracy/empty-rate/latency/cost metrics for both --
only the source of the image(s) handed to the model differs. recall@k
(visual_rag/retrieval_cache/) only applies to Experiment #2, since Experiment
#1 has no retrieval step.

Runs in the MAIN venv.

Usage:
    uv run python visual_rag/compute_metrics.py

Writes visual_rag/visual_rag_summary.json and prints a readable table.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402


def load_questions(dataset, experiment):
    """experiment: 'oracle' (Exp #1) or 'retrieval' (Exp #2) -- see
    rag_lib.qa_path for why these can be different files (currently just
    rental_agreements, whose Exp #1 questions are too document-specific to
    reuse for Exp #2's retrieval)."""
    path = rag_lib.qa_path(dataset, experiment=experiment)
    if not path.exists():
        return {}
    return {q["question_id"]: q for q in json.loads(path.read_text())}


def compute_recall(dataset, questions):
    """recall@1/@5, model-agnostic (retrieval doesn't depend on the answering
    model). MMLongBench-Doc uses reference_page_ids (page-level ground truth,
    since doc_id alone is ambiguous across a multi-page document) and only
    over its answerable subset -- unanswerable questions have no ground-truth
    page to recall."""
    retrieval_path = rag_lib.retrieval_cache_path(dataset)
    if not retrieval_path.exists() or not questions:
        return None
    retrieved = json.loads(retrieval_path.read_text())

    hits1 = hits5 = n = 0
    for qid, q in questions.items():
        if qid not in retrieved:
            continue
        if dataset == "mmlongbench_doc":
            if q.get("unanswerable"):
                continue
            refs = set(q.get("reference_page_ids") or [])
            if not refs:
                continue
            top1 = {Path(retrieved[qid]["top_k_image_paths"][0]).stem}
            top5 = {Path(p).stem for p in retrieved[qid]["top_k_image_paths"]}
            n += 1
            hits1 += bool(refs & top1)
            hits5 += bool(refs & top5)
        else:
            true_doc = q.get("doc_id")
            top = retrieved[qid]["top_k_doc_ids"]
            n += 1
            hits1 += top[0] == true_doc
            hits5 += true_doc in top

    if n == 0:
        return None
    return {"n": n, "recall_at_1": hits1 / n, "recall_at_5": hits5 / n}


def compute_answer_metrics(dataset, model, questions, cache_path_fn):
    path = cache_path_fn(dataset, model)
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        return None

    n = len(records)
    empty = sum(1 for r in records if not r.get("answer"))
    latencies = sorted(r["latency"] for r in records if r.get("latency") is not None)
    costs = [r["usage"]["cost_usd"] for r in records if r.get("usage") and r["usage"].get("cost_usd") is not None]

    correct = 0
    scored = 0
    abstain_correct = abstain_total = 0
    for r in records:
        answer = r.get("answer")
        if not answer:
            continue  # empty answers can't be scored either way -- tracked separately
        q = questions.get(r["question_id"])
        if not q:
            continue
        if dataset == "mmlongbench_doc" and q.get("unanswerable"):
            abstain_total += 1
            abstain_correct += rag_lib.abstention_correct(answer)
            continue
        scored += 1
        if dataset == "chartqa":
            correct += rag_lib.relaxed_match(answer, q["reference_answer"])
        else:
            correct += rag_lib.token_overlap_proxy(answer, q["reference_answer"])

    out = {
        "n": n,
        "empty_answer_rate": empty / n,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
    }
    if scored:
        out["accuracy_proxy"] = correct / scored
        out["n_scored"] = scored
    if abstain_total:
        out["abstention_accuracy"] = abstain_correct / abstain_total
        out["n_unanswerable"] = abstain_total
    if costs:
        out["total_cost_usd"] = sum(costs)
        out["avg_cost_usd"] = sum(costs) / len(costs)
    return out


def _print_models(models):
    for model, m in models.items():
        parts = [f"n={m['n']}", f"empty={m['empty_answer_rate']:.0%}"]
        if "accuracy_proxy" in m:
            parts.append(f"accuracy~{m['accuracy_proxy']:.0%} (n={m['n_scored']})")
        if "abstention_accuracy" in m:
            parts.append(f"abstention_acc={m['abstention_accuracy']:.0%} (n={m['n_unanswerable']})")
        if m["avg_latency_s"] is not None:
            parts.append(f"latency avg/median={m['avg_latency_s']:.1f}s/{m['median_latency_s']:.1f}s")
        if "total_cost_usd" in m:
            parts.append(f"cost=${m['total_cost_usd']:.3f}")
        print(f"    {model:<32} " + "  ".join(parts))


def main():
    summary = {}
    for dataset in rag_lib.DATASETS:
        oracle_questions = load_questions(dataset, experiment="oracle")
        retrieval_questions = load_questions(dataset, experiment="retrieval")
        if not oracle_questions and not retrieval_questions:
            continue
        recall = compute_recall(dataset, retrieval_questions)

        oracle_models, retrieval_models = {}, {}
        for model in rag_lib.ANSWER_MODELS:
            m1 = compute_answer_metrics(dataset, model, oracle_questions, rag_lib.oracle_cache_path)
            if m1:
                oracle_models[model] = m1
            m2 = compute_answer_metrics(dataset, model, retrieval_questions, rag_lib.rag_cache_path)
            if m2:
                retrieval_models[model] = m2

        if recall or oracle_models or retrieval_models:
            summary[dataset] = {
                "retrieval": recall,
                "experiment_1_no_retrieval": oracle_models,
                "experiment_2_retrieve_then_generate": retrieval_models,
            }

    out_path = rag_lib.VISUAL_RAG_DIR / "visual_rag_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    for dataset, d in summary.items():
        print(f"\n=== {dataset} ===")
        if d["retrieval"]:
            r = d["retrieval"]
            print(f"  Experiment #2 retrieval quality: recall@1={r['recall_at_1']:.2f}  recall@5={r['recall_at_5']:.2f}  (n={r['n']})")
        if d["experiment_1_no_retrieval"]:
            print("  Experiment #1 (no retrieval -- own document handed directly):")
            _print_models(d["experiment_1_no_retrieval"])
        if d["experiment_2_retrieve_then_generate"]:
            print("  Experiment #2 (retrieve, then generate):")
            _print_models(d["experiment_2_retrieve_then_generate"])

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
