"""
Prepares MMLongBench-Doc (HF yubo2333/MMLongBench-Doc) for the Visual RAG
pipeline: renders a page-count-bounded subsample of its 135 long PDFs (avg
~47.5 pages/doc -- rendering all of them would mean thousands of pages), then
maps each question's evidence_pages (1-indexed, from the benchmark's own
ground truth) onto that manifest's page images.

This is the one dataset in the mix built specifically for cross-page
reasoning and abstention: evidence_pages can span several pages, and ~20% of
questions are marked "Not answerable" (the model should say so, not guess).
Both are preserved as explicit fields on each question, not merged away.

Runs in the MAIN venv (needs pyarrow to read the parquet, pymupdf to render).

Usage:
    uv run python visual_rag/prepare_mmlongbench_doc.py

Writes:
    data/MMLongBench_Doc_rendered/*.png
    visual_rag/page_manifests/mmlongbench_doc.json
    visual_rag/qa_cache/mmlongbench_doc.json
"""

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

PARQUET_PATH = rag_lib.DATA_DIR / "MMLongBench_Doc" / "data" / "train-00000-of-00001.parquet"
DOCUMENTS_DIR = rag_lib.DATASETS["mmlongbench_doc"]["raw_dir"]
RENDERED_DIR = rag_lib.DATASETS["mmlongbench_doc"]["rendered_dir"]


def _select_docs(doc_ids_with_qa, page_budget, seed):
    """Greedily fills a page budget from a shuffled doc order, via fitz page
    counts, so the rendered corpus stays comparable in scale to the other
    datasets (~100 pages) despite MMLongBench-Doc's long documents."""
    import fitz

    rng = __import__("random").Random(seed)
    shuffled = list(doc_ids_with_qa)
    rng.shuffle(shuffled)

    selected, total_pages = [], 0
    for doc_id in shuffled:
        pdf_path = DOCUMENTS_DIR / doc_id
        if not pdf_path.exists():
            continue  # doc_id in the parquet but PDF missing from documents/
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        doc.close()
        if total_pages + n_pages > page_budget and selected:
            continue  # skip docs that would blow the budget, keep filling with smaller ones
        selected.append(pdf_path)
        total_pages += n_pages
        if total_pages >= page_budget:
            break
    return selected


def prepare(page_budget=150, n_questions=60, seed=42):
    import pyarrow.parquet as pq

    if not PARQUET_PATH.exists():
        raise SystemExit(f"No parquet at {PARQUET_PATH} -- download MMLongBench-Doc first.")
    rows = pq.read_table(PARQUET_PATH).to_pylist()

    unique_doc_ids = sorted(set(r["doc_id"] for r in rows))
    selected_pdfs = _select_docs(unique_doc_ids, page_budget, seed)
    selected_doc_ids = {p.name for p in selected_pdfs}

    manifest = rag_lib.build_page_manifest_from_pdfs(selected_pdfs, RENDERED_DIR)
    rag_lib.save_manifest("mmlongbench_doc", manifest)

    # doc_id (parquet) includes ".pdf"; page manifest's doc_id is pdf_path.stem
    # (rag_lib.build_page_manifest_from_pdfs strips the extension) -- and
    # page_num there is 0-indexed vs. evidence_pages' 1-indexed page numbers.
    page_ids_by_doc_page = {
        (page["doc_id"], page["page_num"]): Path(page["image_path"]).stem
        for page in manifest
    }

    candidates = []
    for i, row in enumerate(rows):
        if row["doc_id"] not in selected_doc_ids:
            continue
        doc_stem = Path(row["doc_id"]).stem
        evidence_pages = ast.literal_eval(row["evidence_pages"])
        reference_page_ids = [
            page_ids_by_doc_page[(doc_stem, p - 1)]
            for p in evidence_pages
            if (doc_stem, p - 1) in page_ids_by_doc_page
        ]
        candidates.append({
            "question_id": f"mmlongbench_doc__{doc_stem}__{i}",
            "question": row["question"],
            "doc_id": doc_stem,
            "reference_answer": row["answer"],
            "unanswerable": row["answer"] == "Not answerable",
            "reference_page_ids": reference_page_ids,
            "n_evidence_pages": len(evidence_pages),
            "evidence_sources": ast.literal_eval(row["evidence_sources"]),
            "answer_format": row["answer_format"],
            "doc_type": row["doc_type"],
        })

    questions = rag_lib.sample_questions(candidates, n_questions, seed=seed)
    qa_path = rag_lib.qa_cache_path("mmlongbench_doc")
    qa_path.write_text(json.dumps(questions, indent=2))

    n_unanswerable = sum(1 for q in questions if q["unanswerable"])
    n_cross_page = sum(1 for q in questions if q["n_evidence_pages"] > 1)
    print(f"[mmlongbench_doc] corpus: {len(manifest)} pages across {len(selected_pdfs)} documents")
    print(f"[mmlongbench_doc] questions: {len(questions)} (from {len(candidates)} candidates)")
    print(f"  -- {n_unanswerable} unanswerable, {n_cross_page} multi-page-evidence")
    print(f"  -> {rag_lib.manifest_path('mmlongbench_doc')}")
    print(f"  -> {qa_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-budget", type=int, default=150)
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(page_budget=args.page_budget, n_questions=args.n_questions, seed=args.seed)
