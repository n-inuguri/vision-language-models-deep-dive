"""
Shared extraction library for the multi-model, multi-document-type local
comparison (Week 3 Phase 2). Mirrors the eval_utils.py pattern from Week 2 --
one module of shared config/helpers reused by the runner script, the manual
labeling script, and the report notebook.

Everything here is model/prompt/schema plumbing only. It never persists real
extracted values itself -- callers are responsible for writing results only
to the gitignored caches (poc/real_data_cache/, poc/manual_ground_truth_cache/).
"""

import json
import random
import re
import time
from pathlib import Path

import ollama

MODELS = ["qwen2.5vl:7b", "qwen2.5vl:3b", "llama3.2-vision:11b", "minicpm-v", "llava-phi3"]

DOC_DIRS = {
    "pan": Path(__file__).parent.parent / "data" / "PAN_Kaggle" / "train" / "images",
    "aadhaar": Path(__file__).parent.parent / "data" / "Adhaar_Kaggle" / "train" / "images",
}

DOC_SCHEMAS = {
    "pan": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The PAN card holder's full name"},
            "father_name": {"type": "string", "description": "The holder's father's name"},
            "dob": {"type": "string", "description": "Date of birth as printed on the card"},
            "pan_number": {"type": "string", "description": "The 10-character PAN number"},
        },
        "required": ["name", "father_name", "dob", "pan_number"],
    },
    "aadhaar": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The Aadhaar card holder's full name"},
            "dob": {"type": "string", "description": "Date of birth as printed on the card (or year of birth if only that is shown)"},
            "gender": {"type": "string", "description": "Gender as printed on the card (e.g. Male/Female/Transgender)"},
            "aadhaar_number": {"type": "string", "description": "The 12-digit Aadhaar number, digits only"},
        },
        "required": ["name", "dob", "gender", "aadhaar_number"],
    },
}

EXTRACTION_PROMPTS = {
    "pan": (
        "This is a scanned Indian PAN (Permanent Account Number) card. "
        "Extract the holder name, father name, date of birth, and PAN number exactly as printed. "
        "If a field is not legible, use an empty string for it."
    ),
    "aadhaar": (
        "This is a scanned Indian Aadhaar card. "
        "Extract the holder name, date of birth, gender, and the 12-digit Aadhaar number exactly as printed. "
        "For the Aadhaar number, return digits only (no spaces). "
        "If a field is not legible, use an empty string for it."
    ),
}

FIELD_NAMES = {
    "pan": ["name", "father_name", "dob", "pan_number"],
    "aadhaar": ["name", "dob", "gender", "aadhaar_number"],
}

PAN_NUMBER_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
AADHAAR_NUMBER_RE = re.compile(r"^\d{12}$")
DATE_FORMAT_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}$")
GENDER_RE = re.compile(r"^(male|female|transgender|other|m|f|t)$", re.IGNORECASE)


def _looks_like_name(s):
    return bool(s) and bool(re.match(r"^[A-Za-z .]+$", s.strip()))


def normalize(s):
    return " ".join(str(s).strip().upper().split())


def field_is_well_formed(doc_type, field, value):
    """Format-validity check only -- says nothing about whether the value is correct."""
    value = (value or "").strip()
    if field in ("name", "father_name"):
        return _looks_like_name(value)
    if field == "dob":
        return bool(DATE_FORMAT_RE.match(value))
    if field == "pan_number":
        return bool(PAN_NUMBER_RE.match(value.replace(" ", "").upper()))
    if field == "aadhaar_number":
        return bool(AADHAAR_NUMBER_RE.match(value.replace(" ", "")))
    if field == "gender":
        return bool(GENDER_RE.match(value))
    return bool(value)


SOURCE_STEM_RE = re.compile(r"(_jpg)?\.rf\.[0-9a-f]+\.jpg$")


def _source_stem(filename):
    """Roboflow exports multiple augmented copies of the same source scan as
    '<stem>_jpg.rf.<hash>.jpg' / '<stem>.rf.<hash>.jpg' -- this recovers <stem>,
    which identifies the underlying document regardless of which augmented
    copy a given file is."""
    return SOURCE_STEM_RE.sub("", filename)


def sample_images(doc_type, n=100, seed=42, force_include=None):
    """Fixed, reproducible sample of real images for a doc type -- every model
    sees the exact same images, so results are comparable across models.

    Aadhaar-specific dedup: data/Adhaar_Kaggle turned out to be extremely
    augmented -- 1852 files but only 217 distinct underlying documents, some
    duplicated up to 17x (confirmed by grouping filenames via _source_stem).
    A naive sample would silently over-represent a handful of people. PAN's
    duplication is negligible (~2%, 88/1458 files) and PAN's fixed 100-sample
    is already partially run through several models -- deduping it now would
    invalidate that cached progress for a marginal benefit, so PAN sampling is
    left as plain random sampling over all files.

    force_include: filenames that must be in the returned sample regardless of
    the random draw -- used only for Aadhaar, to guarantee already-labeled
    manual ground-truth images stay part of the set that actually gets run
    through the models. NEVER pass this for 'pan': PAN's sample is already
    partially cached, and adding force_include changes which *other* files get
    randomly selected (removing items from the population shifts the rest of
    random.sample's draw), which would invalidate that cached progress.
    """
    images = sorted(DOC_DIRS[doc_type].glob("*.jpg"))

    if doc_type == "aadhaar":
        seen_stems = set()
        deduped = []
        for img in images:
            stem = _source_stem(img.name)
            if stem not in seen_stems:
                seen_stems.add(stem)
                deduped.append(img)
        images = deduped

    force_include = set(force_include or [])
    forced = [img for img in images if img.name in force_include]
    remaining = [img for img in images if img.name not in force_include]

    rng = random.Random(seed)
    fill_n = max(0, n - len(forced))
    sampled = rng.sample(remaining, min(fill_n, len(remaining)))
    return forced + sampled


def extract_fields(image_path, model_name, doc_type):
    """Runs one Ollama vision-model extraction call. Returns (fields_or_None, latency_seconds)."""
    t0 = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPTS[doc_type],
            "images": [str(image_path)],
        }],
        format=DOC_SCHEMAS[doc_type],
    )
    latency = time.time() - t0
    try:
        fields = json.loads(response.message.content)
    except json.JSONDecodeError:
        fields = None
    return fields, latency
