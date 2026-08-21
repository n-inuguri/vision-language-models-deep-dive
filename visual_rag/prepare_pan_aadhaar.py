"""
Prepares the PAN/Aadhaar datasets for the Visual RAG pipeline.

Ground truth lives in visual_rag/expanded_ground_truth/{pan,aadhaar}.json
(gitignored, real PII), built by combining/matching multiple sources -- see
each file's provenance below. This supersedes the original Week 3-only
scale (15 pan / 15 aadhaar docs, from poc/manual_ground_truth_cache/); those
files are untouched, kept as the Week 3 POC's own record.
  - PAN (120 docs): a teammate's transcriptions from the dataset's own
    bounding-box labels (data/PAN_Kaggle/GroundTruth/{train,test}.txt),
    matched to actual downloaded images by stable filename prefix (Roboflow
    re-exports images under a different .rf.<hash> suffix each time, so
    matching must ignore that suffix).
  - Aadhaar (18 docs = 15 original + 3 new): a colleague's dataset
    (data/Adhaar_Kaggle/GroundTruth/) had 65 more candidate records, but 17
    of the 20 "complete" ones are explicitly unreviewed GLM-OCR model
    prefill ("Review required before approval" in their own notes, despite
    a review_status of "approved") -- excluded as not real ground truth.
    Only 3 are genuinely human-reviewed (no such note) and are included.
    Those 3 use `address` where the original 15 use `gender` (different
    source schemas) -- both fields are kept as a union field taxonomy below;
    each document just contributes whichever fields it actually has.

Runs in the MAIN venv only (imports poc/extraction_lib.py for sample_images,
which imports ollama at module level) -- never invoke this under
.venv-colpali.

Unlike rental_agreements/ChartQA/SPIQA/MMLongBench-Doc, there's no natural
single "question" per document here -- the ground truth is a set of fields
per card. This script turns each labeled (image, field) pair into one
retrieval question ("What is the PAN number shown on this document?"), with
the manually-verified field value as the reference answer -- so retrieval and
answer-generation get evaluated exactly like the other datasets.

The retrieval CORPUS is a larger sampled pool, not just the labeled images --
per the shared design principle (index_dir()/retrieve.py), retrieval must
search the whole corpus, not just the document a question came from, or
recall@k is meaningless. All labeled images are force-included in that pool
(poc/extraction_lib.sample_images' force_include) so every question's
ground-truth image is guaranteed to be retrievable. At this larger PAN scale
(120 labeled docs) force_include is now used for PAN too -- this changes
PAN's sampled pool versus the earlier n=100 unforced sample, which is
expected: this supersedes the earlier, smaller-scale PAN result.

Usage:
    uv run python visual_rag/prepare_pan_aadhaar.py --doc-type pan
    uv run python visual_rag/prepare_pan_aadhaar.py --doc-type aadhaar

Writes:
    visual_rag/page_manifests/{pan,aadhaar}.json   (paths only, not PII itself)
    visual_rag/manual_qa_cache/{pan,aadhaar}.json  (real PII -- gitignored)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "poc"))
from extraction_lib import sample_images  # noqa: E402

# Union of both Aadhaar ground-truth sources' schemas (gender from the
# original 15, address from the 3 newly-reviewed) -- a document simply
# contributes whichever fields it actually has (see prepare()'s `if not
# value` skip below), so a non-uniform schema across documents is fine.
FIELD_NAMES = {
    "pan": ["name", "father_name", "dob", "pan_number"],
    "aadhaar": ["name", "dob", "gender", "address", "aadhaar_number"],
}

# Every PAN/Aadhaar card shares the same visual template -- a generic
# per-field question ("What is the full name shown on this document?") gives
# the retriever nothing to distinguish one card from another, so retrieval
# would be meaningless (confirmed by a smoke test: it wasn't finding the
# right card at all). Each question is instead anchored on a DIFFERENT
# field's value from the same card ("...of the person whose PAN number is
# X?"), giving the retriever a genuine, unique detail to match against --
# closer to how a real user would actually ask ("find Acme Corp's invoice").
ANCHOR_FIELD = {
    "pan": {"name": "pan_number", "father_name": "pan_number", "dob": "pan_number", "pan_number": "name"},
    "aadhaar": {
        "name": "aadhaar_number", "dob": "aadhaar_number", "gender": "aadhaar_number",
        "address": "aadhaar_number", "aadhaar_number": "name",
    },
}
ANCHOR_LABEL = {"pan_number": "PAN number", "aadhaar_number": "Aadhaar number"}
TARGET_LABEL = {
    "name": "full name", "father_name": "father's name", "dob": "date of birth",
    "gender": "gender", "address": "address", "pan_number": "PAN number",
    "aadhaar_number": "12-digit Aadhaar number",
}


def build_question(field, anchor_field, anchor_value):
    if anchor_field == "name":
        anchor_desc = f"the person named {anchor_value}"
    else:
        anchor_desc = f"the person whose {ANCHOR_LABEL[anchor_field]} is {anchor_value}"
    return f"What is the {TARGET_LABEL[field]} of {anchor_desc}?"


GT_DIR = Path(__file__).resolve().parent / "expanded_ground_truth"

# poc/extraction_lib.sample_images() only globs DOC_DIRS["pan"] (train/images)
# -- at this larger PAN scale, 17/120 labeled docs live in valid/images
# instead, which sample_images can't see at all. force_include-ing their bare
# filenames would silently drop those 17 with no error (they'd still get
# QUESTIONS below, just unretrievable ones). So PAN's pool is assembled here
# directly, spanning both directories, instead of going through sample_images.
_PAN_IMAGE_DIRS = [
    Path(__file__).resolve().parent.parent / "data" / "PAN_Kaggle" / "train" / "images",
    Path(__file__).resolve().parent.parent / "data" / "PAN_Kaggle" / "valid" / "images",
]


def _pan_pool(ground_truth, corpus_size):
    by_name = {}
    for d in _PAN_IMAGE_DIRS:
        for f in d.glob("*.jpg"):
            by_name.setdefault(f.name, f)

    forced = [by_name[name] for name in ground_truth if name in by_name]
    missing = [name for name in ground_truth if name not in by_name]
    if missing:
        raise SystemExit(f"{len(missing)} ground-truth PAN images not found on disk: {missing[:5]}...")

    import random
    remaining = [f for f in by_name.values() if f.name not in ground_truth]
    fill_n = max(0, corpus_size - len(forced))
    filler = random.Random(42).sample(remaining, min(fill_n, len(remaining)))
    return forced + filler


def prepare(doc_type, corpus_size=100):
    gt_path = GT_DIR / f"{doc_type}.json"
    if not gt_path.exists():
        raise SystemExit(f"No ground truth at {gt_path} -- build it first (see this file's docstring).")
    ground_truth = json.loads(gt_path.read_text())

    if doc_type == "pan":
        pool = _pan_pool(ground_truth, corpus_size)
    else:
        # aadhaar: sample_images already handles force_include correctly
        # (single directory, dedup against augmented duplicates).
        pool = sample_images(doc_type, n=corpus_size, force_include=list(ground_truth.keys()))

    manifest = rag_lib.build_page_manifest_from_images(pool)
    rag_lib.save_manifest(doc_type, manifest)

    questions = []
    for image_name, fields in ground_truth.items():
        for field in FIELD_NAMES[doc_type]:
            value = fields.get(field)
            anchor_field = ANCHOR_FIELD[doc_type][field]
            anchor_value = fields.get(anchor_field)
            if not value or not anchor_value:
                continue  # can't pose a well-anchored question without both values
            questions.append({
                "question_id": f"{doc_type}__{Path(image_name).stem}__{field}",
                "question": build_question(field, anchor_field, anchor_value),
                "doc_id": Path(image_name).stem,
                "field": field,
                "reference_answer": value,
            })

    qa_path = rag_lib.MANUAL_QA_DIR / f"{doc_type}.json"
    qa_path.write_text(json.dumps(questions, indent=2))

    print(f"[{doc_type}] manifest: {len(manifest)} pages (pool includes all "
          f"{len(ground_truth)} labeled images)")
    print(f"[{doc_type}] questions: {len(questions)} (from {len(ground_truth)} "
          f"labeled documents x {len(FIELD_NAMES[doc_type])} fields)")
    print(f"  -> {rag_lib.manifest_path(doc_type)}")
    print(f"  -> {qa_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-type", choices=["pan", "aadhaar"], required=True)
    parser.add_argument("--corpus-size", type=int, default=100)
    args = parser.parse_args()
    prepare(args.doc_type, corpus_size=args.corpus_size)
