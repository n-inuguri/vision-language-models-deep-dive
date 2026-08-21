"""
Shared Visual RAG library (Week 4). Mirrors the extraction_lib.py/eval_utils.py
pattern from earlier phases: one module of shared config/helpers reused by
build_index.py, retrieve.py, run_visual_rag.py, prepare_rental_agreements.py,
label_ground_truth_rental.py, and the report notebook.

Must stay importable from BOTH environments used in this phase:
- the main project .venv (ollama, openai/OpenRouter, pymupdf, python-dotenv)
- .venv-colpali (colpali-engine only, used by build_index.py/retrieve.py)
So nothing at module level imports ollama/openai/fitz -- those are lazy
imports inside the functions that actually need them.
"""

import json
import os
import random
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
VISUAL_RAG_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = VISUAL_RAG_DIR / "page_manifests"
INDEX_DIR = VISUAL_RAG_DIR / "indexes"
RETRIEVAL_CACHE_DIR = VISUAL_RAG_DIR / "retrieval_cache"
RAG_CACHE_DIR = VISUAL_RAG_DIR / "rag_cache"
# Experiment #1 (no retrieval -- the question's own document is handed to the
# VLM directly) results, kept separate from RAG_CACHE_DIR's Experiment #2
# (retrieve, then generate) results so the two don't collide/overwrite.
ORACLE_CACHE_DIR = VISUAL_RAG_DIR / "oracle_cache"
MANUAL_QA_DIR = VISUAL_RAG_DIR / "manual_qa_cache"
# Dataset-provided QA (ChartQA/SPIQA/MMLongBench-Doc), derived from public
# benchmark ground truth -- unlike MANUAL_QA_DIR, not PII-adjacent, safe to commit.
QA_CACHE_DIR = VISUAL_RAG_DIR / "qa_cache"

for _dir in (MANIFEST_DIR, INDEX_DIR, RETRIEVAL_CACHE_DIR, RAG_CACHE_DIR, ORACLE_CACHE_DIR,
             MANUAL_QA_DIR, QA_CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

COLPALI_MODEL_NAME = "vidore/colqwen2-v1.0"

# --- Dataset registry --------------------------------------------------
# raw_dir/rendered_dir below reflect each dataset's actual on-disk layout,
# confirmed by inspection once downloaded (see prepare_{chartqa,spiqa,
# mmlongbench_doc}.py for the per-dataset extraction/render logic that
# populates these paths).
DATASETS = {
    "rental_agreements": {
        "page_source": "render_text",
        "raw_dir": DATA_DIR / "rental_agreements_raw",
        "rendered_dir": DATA_DIR / "rental_agreements_rendered",
        "qa_source": "manual",
    },
    "chartqa": {
        # Images are embedded as PNG bytes inside the HF parquet files
        # (data/ChartQA/data/*.parquet), not on-disk files -- prepare_chartqa.py
        # extracts a sampled pool to raw_dir below before this is "images_exist".
        "page_source": "images_exist",
        "raw_dir": DATA_DIR / "ChartQA" / "extracted_images",
        "qa_source": "dataset_provided",
    },
    "spiqa": {
        # SPIQA already ships one cropped image per figure/table (no PDF
        # rendering needed) -- prepare_spiqa.py just unzips test-A's images.
        "page_source": "images_exist",
        "raw_dir": DATA_DIR / "SPIQA" / "test-A" / "images",
        "qa_source": "dataset_provided",
    },
    "mmlongbench_doc": {
        "page_source": "render_pdf",
        "raw_dir": DATA_DIR / "MMLongBench_Doc" / "documents",
        "rendered_dir": DATA_DIR / "MMLongBench_Doc_rendered",
        "qa_source": "dataset_provided",
    },
    # PAN/Aadhaar: reuse the real, manually-labeled images/ground truth from
    # the Week 3 POC (poc/manual_ground_truth_cache/{pan,aadhaar}.json) rather
    # than re-authoring anything. Built via prepare_pan_aadhaar.py, not a
    # render step -- images already exist. Real PII; see .gitignore.
    "pan": {
        "page_source": "images_exist",
        "raw_dir": DATA_DIR / "PAN_Kaggle" / "train" / "images",
        "qa_source": "manual",
    },
    "aadhaar": {
        "page_source": "images_exist",
        "raw_dir": DATA_DIR / "Adhaar_Kaggle" / "train" / "images",
        "qa_source": "manual",
    },
}


def manifest_path(dataset):
    return MANIFEST_DIR / f"{dataset}.json"


def index_dir(dataset):
    return INDEX_DIR / dataset


def retrieval_cache_path(dataset):
    return RETRIEVAL_CACHE_DIR / f"{dataset}.json"


def rag_cache_path(dataset, model_name):
    safe_model = model_name.replace(":", "_").replace("/", "_")
    return RAG_CACHE_DIR / f"{dataset}__{safe_model}.jsonl"


def oracle_cache_path(dataset, model_name):
    safe_model = model_name.replace(":", "_").replace("/", "_")
    return ORACLE_CACHE_DIR / f"{dataset}__{safe_model}.jsonl"


def images_for_question(dataset, manifest, question):
    """The image(s) to hand the VLM directly for Experiment #1 (no
    retrieval) -- the question's own document, not whatever ColPali would
    retrieve. Most datasets are one-page-per-document, so this is just that
    one page. MMLongBench-Doc is multi-page: prefer the question's own
    reference_page_ids (the exact evidence page(s), a true "oracle" -- not
    just "the right document" but "the right page") when available; an
    unanswerable question has no evidence page, so falls back to every
    rendered page of that document, since that's the fairest way to let the
    model discover there's nothing to find."""
    doc_id = question["doc_id"]
    doc_pages = sorted(
        (p for p in manifest if p["doc_id"] == doc_id),
        key=lambda p: p["page_num"],
    )
    ref_page_ids = question.get("reference_page_ids")
    if ref_page_ids:
        by_stem = {Path(p["image_path"]).stem: p["image_path"] for p in doc_pages}
        matched = [by_stem[s] for s in ref_page_ids if s in by_stem]
        if matched:
            return matched
    return [p["image_path"] for p in doc_pages]


def qa_cache_path(dataset):
    return QA_CACHE_DIR / f"{dataset}.json"


def qa_path(dataset, experiment="oracle"):
    """Path to a dataset's question set. experiment is "oracle" (Experiment
    #1, no retrieval) or "retrieval" (Experiment #2). Datasets with
    qa_source != "manual" (chartqa/spiqa/mmlongbench_doc) always use
    qa_cache_path -- same questions serve both experiments there. For manual
    datasets (rental_agreements/pan/aadhaar), both experiments share one
    file UNLESS a "{dataset}_exp2.json" exists in MANUAL_QA_DIR, in which
    case Experiment #2 uses that instead -- needed when a dataset's Exp #1
    questions are too document-specific to give retrieval any signal (see
    rental_agreements: authored per-document, recall@1=0.04 when reused
    as-is for Exp #2)."""
    config = DATASETS[dataset]
    if config["qa_source"] != "manual":
        return qa_cache_path(dataset)
    if experiment == "retrieval":
        exp2_path = MANUAL_QA_DIR / f"{dataset}_exp2.json"
        if exp2_path.exists():
            return exp2_path
    return MANUAL_QA_DIR / f"{dataset}.json"


# --- Page manifest -------------------------------------------------------
def build_page_manifest_from_images(image_paths):
    """One page per image, doc_id = image stem. For datasets that are already
    single images (e.g. ChartQA)."""
    return [
        {"doc_id": p.stem, "page_num": 0, "image_path": str(p), "tags": {}}
        for p in sorted(Path(p) for p in image_paths)
    ]


def build_page_manifest_from_pdfs(pdf_paths, output_dir, dpi=150):
    """Renders each PDF's pages to PNG via PyMuPDF (main venv only), returns a
    page manifest. Skips a page's render if its PNG already exists, so this is
    safely re-runnable without re-rendering everything."""
    import fitz  # PyMuPDF

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pdf_path in sorted(Path(p) for p in pdf_paths):
        doc_id = pdf_path.stem
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            image_path = output_dir / f"{doc_id}__p{page_num:03d}.png"
            if not image_path.exists():
                pix = doc[page_num].get_pixmap(dpi=dpi)
                pix.save(image_path)
            manifest.append({
                "doc_id": doc_id, "page_num": page_num,
                "image_path": str(image_path), "tags": {},
            })
        doc.close()
    return manifest


def save_manifest(dataset, manifest):
    manifest_path(dataset).write_text(json.dumps(manifest, indent=2))


def load_manifest(dataset):
    path = manifest_path(dataset)
    if not path.exists():
        return None
    return json.loads(path.read_text())


# --- Question sampling -----------------------------------------------------
def sample_questions(questions, n, seed=42):
    """Fixed, reproducible sample of question dicts -- mirrors
    poc/extraction_lib.py's sample_images() pattern (random.Random(seed))."""
    rng = random.Random(seed)
    return rng.sample(questions, min(n, len(questions)))


# --- Answer-generation dispatch ---------------------------------------------
ANSWER_MODELS = [
    "qwen2.5vl:7b", "qwen2.5vl:3b", "llama3.2-vision:11b", "minicpm-v", "llava-phi3",
    "google/gemini-3.1-pro-preview", "qwen/qwen2.5-vl-72b-instruct",
]

MODEL_PROVIDER = {
    "qwen2.5vl:7b": "ollama",
    "qwen2.5vl:3b": "ollama",
    "llama3.2-vision:11b": "ollama",
    "minicpm-v": "ollama",
    "llava-phi3": "ollama",
    "google/gemini-3.1-pro-preview": "openrouter",
    "qwen/qwen2.5-vl-72b-instruct": "openrouter",
}

# Point-in-time snapshot (August 2026, from openrouter.ai) -- re-check the
# live model pages before trusting cost numbers on a much later run.
PRICING = {
    "google/gemini-3.1-pro-preview": {"input_per_1m": 2.00, "output_per_1m": 12.00},
    "qwen/qwen2.5-vl-72b-instruct": {"input_per_1m": 0.25, "output_per_1m": 0.75},
}

# Some local models reject a multi-image request outright (confirmed:
# llama3.2-vision:11b errors "this model only supports one image" against
# retrieve.py's top-5), so cap how many of the top-k retrieved images
# generate_answer() actually sends for those models specifically.
MAX_IMAGES = {
    "llama3.2-vision:11b": 1,
}


def _image_to_data_url(image_path):
    import base64
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    return f"data:image/{mime};base64,{data}"


def generate_answer(image_paths, question, model_name):
    """Runs one answer-generation call, local (Ollama) or cloud (OpenRouter).

    Returns (answer_text_or_None, latency_seconds, usage_or_None). usage is
    {"input_tokens":.., "output_tokens":.., "cost_usd":..} for OpenRouter
    models, None for Ollama (no token-cost concept applies locally).
    """
    provider = MODEL_PROVIDER[model_name]
    image_paths = [str(p) for p in image_paths][:MAX_IMAGES.get(model_name)]
    t0 = time.time()

    if provider == "ollama":
        import ollama  # lazy: keeps this module importable from .venv-colpali too
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": question, "images": image_paths}],
        )
        return response.message.content, time.time() - t0, None

    if provider == "openrouter":
        import openai  # lazy: same reason
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        content = [{"type": "text", "text": question}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(p)}})
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
        )
        latency = time.time() - t0
        answer = response.choices[0].message.content
        usage = None
        if response.usage:
            pricing = PRICING.get(model_name)
            cost = None
            if pricing:
                cost = (
                    response.usage.prompt_tokens / 1e6 * pricing["input_per_1m"]
                    + response.usage.completion_tokens / 1e6 * pricing["output_per_1m"]
                )
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cost_usd": cost,
            }
        return answer, latency, usage

    raise ValueError(f"Unknown provider for model {model_name!r}")


# --- Correctness helpers -----------------------------------------------
_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
              "to", "and", "or", "it", "this", "that", "as", "at", "by", "for"}


def normalize(s):
    return " ".join(str(s).strip().upper().split())


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def relaxed_match(predicted, reference, numeric_tolerance=0.05):
    """ChartQA-style relaxed accuracy, adapted for free-form VLM answers
    (a full sentence, not a bare extracted value -- unlike Week 3's
    schema-constrained JSON extraction this was originally written for).
    Numeric references: any number mentioned anywhere in the predicted text
    within relative tolerance counts as a match. String references: the
    reference's normalized text appearing anywhere in the predicted text
    counts as a match (containment, not exact-whole-string equality)."""
    try:
        ref_num = float(str(reference).replace(",", ""))
        for match in _NUMBER_RE.findall(str(predicted)):
            pred_num = float(match.replace(",", ""))
            if ref_num == 0:
                if pred_num == 0:
                    return True
            elif abs(pred_num - ref_num) / abs(ref_num) <= numeric_tolerance:
                return True
        return False
    except (TypeError, ValueError):
        return normalize(reference) in normalize(predicted)


def token_overlap_proxy(predicted, reference, threshold=0.6):
    """Free-form-answer correctness proxy for SPIQA/MMLongBench-Doc/rental
    agreements: does the reference answer's key terms show up in the
    generated answer? An automatic proxy only -- pair with a small
    manually-verified subsample, same honesty pattern as
    poc/model_comparison_summary.json's cross_model_agreement (proxy) +
    manual_ground_truth_accuracy (the real check)."""
    ref_tokens = set(re.findall(r"\w+", reference.lower())) - _STOPWORDS
    if not ref_tokens:
        return False
    pred_lower = predicted.lower()
    hits = sum(1 for tok in ref_tokens if tok in pred_lower)
    return hits / len(ref_tokens) >= threshold


_ABSTENTION_PHRASES = [
    "not answerable", "cannot be determined", "cannot determine", "can't determine",
    "no information", "not possible to answer", "unable to answer", "does not provide",
    "does not contain", "not mentioned", "not stated", "insufficient information",
    "cannot find", "no such", "does not appear", "not available in",
]


def abstention_correct(predicted):
    """Heuristic proxy for MMLongBench-Doc's unanswerable subset: did the model
    correctly decline rather than guess? Substring match against common
    refusal phrasing -- a proxy, not a semantic check, same honesty caveat as
    token_overlap_proxy above."""
    pred_lower = (predicted or "").lower()
    return any(phrase in pred_lower for phrase in _ABSTENTION_PHRASES)
