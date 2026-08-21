"""
Prepares SPIQA (HF google/spiqa, test-A split) for the Visual RAG pipeline.

Unlike the other datasets, SPIQA already ships one cropped image per
figure/table (no PDF rendering, no page-ization needed) -- each question's
ground truth is a specific figure/table image within its paper, and the
retrieval corpus is a pool of figures across several papers, several of
which come from the same paper (a genuine "which figure, out of many similar
ones" retrieval test, unlike ChartQA's one-chart-per-doc setup).

Runs in the MAIN venv (only needs the stdlib zipfile module; no colpali_engine).

Usage:
    uv run python visual_rag/prepare_spiqa.py

Writes:
    data/SPIQA/test-A/images/<paper_id>/*.png   (unzipped once, ~120MB)
    visual_rag/page_manifests/spiqa.json
    visual_rag/qa_cache/spiqa.json
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

TESTA_DIR = rag_lib.DATA_DIR / "SPIQA" / "test-A"
QA_JSON = TESTA_DIR / "SPIQA_testA.json"
IMAGES_ZIP = TESTA_DIR / "SPIQA_testA_Images.zip"
IMAGES_DIR = rag_lib.DATASETS["spiqa"]["raw_dir"]


def _ensure_images_extracted():
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        return
    if not IMAGES_ZIP.exists():
        raise SystemExit(f"No images zip at {IMAGES_ZIP} -- download SPIQA test-A first.")
    print(f"Extracting {IMAGES_ZIP.name} -> {IMAGES_DIR} ...")
    with zipfile.ZipFile(IMAGES_ZIP) as zf:
        zf.extractall(IMAGES_DIR.parent)
    # The zip's top-level folder is "SPIQA_testA_Images/<paper_id>/..." --
    # flatten that one level so IMAGES_DIR == .../images/<paper_id>/...
    extracted = IMAGES_DIR.parent / "SPIQA_testA_Images"
    if extracted.exists():
        extracted.rename(IMAGES_DIR)


def prepare(n_papers=15, n_questions=60, seed=42):
    if not QA_JSON.exists():
        raise SystemExit(f"No QA json at {QA_JSON} -- download SPIQA test-A first.")
    _ensure_images_extracted()

    papers = json.loads(QA_JSON.read_text())
    paper_ids = rag_lib.sample_questions(sorted(papers), n_papers, seed=seed)

    manifest_images = []
    candidates = []
    for paper_id in paper_ids:
        paper = papers[paper_id]
        paper_dir = IMAGES_DIR / paper_id
        for fig_name in paper["all_figures"]:
            fig_path = paper_dir / fig_name
            if fig_path.exists():
                manifest_images.append(fig_path)
        for i, qa in enumerate(paper["qa"]):
            ref_path = paper_dir / qa["reference"]
            if not ref_path.exists():
                continue  # reference figure missing from the extracted zip; skip
            candidates.append({
                "question_id": f"spiqa__{paper_id}__{i}",
                "question": qa["question"],
                "doc_id": Path(qa["reference"]).stem,
                "reference_answer": qa["answer"],
            })

    manifest = rag_lib.build_page_manifest_from_images(manifest_images)
    rag_lib.save_manifest("spiqa", manifest)

    questions = rag_lib.sample_questions(candidates, n_questions, seed=seed)
    qa_path = rag_lib.qa_cache_path("spiqa")
    qa_path.write_text(json.dumps(questions, indent=2))

    print(f"[spiqa] corpus: {len(manifest)} figures across {len(paper_ids)} papers")
    print(f"[spiqa] questions: {len(questions)} (from {len(candidates)} candidates)")
    print(f"  -> {rag_lib.manifest_path('spiqa')}")
    print(f"  -> {qa_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-papers", type=int, default=15)
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(n_papers=args.n_papers, n_questions=args.n_questions, seed=args.seed)
