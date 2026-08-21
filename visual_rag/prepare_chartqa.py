"""
Prepares ChartQA (HF HuggingFaceM4/ChartQA) for the Visual RAG pipeline.

Unlike every other dataset here, ChartQA's images aren't files on disk --
they're PNG bytes embedded directly inside the HF parquet files
(data/ChartQA/data/*.parquet, one row per (chart, question) pair, many
questions sharing the same chart). This script decodes a sampled pool of
unique chart images to raw_dir (rag_lib.DATASETS["chartqa"]["raw_dir"]) so the
rest of the pipeline (build_page_manifest_from_images, ColQwen2 indexing) can
treat it exactly like every other images_exist dataset.

Runs in the MAIN venv (needs pyarrow to read the parquet; no colpali_engine).

Usage:
    uv run python visual_rag/prepare_chartqa.py

Writes:
    data/ChartQA/extracted_images/*.png        (a sampled corpus of unique charts)
    visual_rag/page_manifests/chartqa.json
    visual_rag/qa_cache/chartqa.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

TEST_PARQUET = rag_lib.DATA_DIR / "ChartQA" / "data" / "test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"
RAW_DIR = rag_lib.DATASETS["chartqa"]["raw_dir"]


def prepare(corpus_size=100, n_questions=60, seed=42):
    import pyarrow.parquet as pq

    if not TEST_PARQUET.exists():
        raise SystemExit(f"No parquet at {TEST_PARQUET} -- download ChartQA first.")
    rows = pq.read_table(TEST_PARQUET).to_pylist()

    # Many rows share the same chart image (avg ~1.7 questions/chart in the
    # test split) -- dedupe by content hash so each unique chart becomes one
    # page, same as build_page_manifest_from_images expects one image per page.
    by_hash = {}
    for row in rows:
        h = hashlib.md5(row["image"]["bytes"]).hexdigest()
        by_hash.setdefault(h, {"image_bytes": row["image"]["bytes"], "rows": []})
        by_hash[h]["rows"].append(row)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pool_hashes = rag_lib.sample_questions(sorted(by_hash), corpus_size, seed=seed)

    manifest_images = []
    for h in pool_hashes:
        image_path = RAW_DIR / f"{h}.png"
        if not image_path.exists():
            image_path.write_bytes(by_hash[h]["image_bytes"])
        manifest_images.append(image_path)

    manifest = rag_lib.build_page_manifest_from_images(manifest_images)
    rag_lib.save_manifest("chartqa", manifest)

    # human_or_machine == 0 is the human-authored subset -- natural-language
    # questions, closer to how a real user would actually ask, vs. 1's
    # template-generated ones.
    candidates = [
        {
            "question_id": f"chartqa__{h}__{i}",
            "question": row["query"],
            "doc_id": h,
            "reference_answer": row["label"][0],
        }
        for h in pool_hashes
        for i, row in enumerate(by_hash[h]["rows"])
        if row["human_or_machine"] == 0
    ]
    questions = rag_lib.sample_questions(candidates, n_questions, seed=seed)

    qa_path = rag_lib.qa_cache_path("chartqa")
    qa_path.write_text(json.dumps(questions, indent=2))

    print(f"[chartqa] corpus: {len(manifest)} unique charts (from {len(by_hash)} available)")
    print(f"[chartqa] questions: {len(questions)} human-authored (from {len(candidates)} candidates in the pool)")
    print(f"  -> {rag_lib.manifest_path('chartqa')}")
    print(f"  -> {qa_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=100)
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(corpus_size=args.corpus_size, n_questions=args.n_questions, seed=args.seed)
