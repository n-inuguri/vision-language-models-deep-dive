# VLM Deep Dive: From Scratch to Retrieval-Augmented Visual QA

A four-part technical exploration of Vision-Language Models (VLMs), progressing from
building one from scratch to a full retrieval-augmented visual question-answering system.

- **Part I — Minimal VLM from scratch** (`minimal_vlm/`): builds an image-captioning VLM in
  PyTorch from first principles, then rebuilds it in stages toward a LLaVA-style architecture
  (frozen pretrained vision encoder + LoRA-fine-tuned language model), tracking corpus BLEU-4
  across each stage.
- **Part II — Structured field extraction** (`poc/`): applies pretrained local VLMs (via
  Ollama) to extract structured fields from scanned identity documents.
- **Part III — Single-document visual QA** (`visual_rag/`): free-form question answering over
  one known document image/PDF, across six document types.
- **Part IV — Retrieval-augmented visual QA** (`visual_rag/`): adds a ColPali-style page-image
  retrieval step (via ColQwen2) — a question must first find the right document out of a
  corpus, then a VLM answers using the retrieved page image.

No PII, client data, or other sensitive material is included in this repository. Datasets
containing real personal data (identity documents, rental agreements) must be sourced and
placed locally by the user as described below, and are never committed.

## Repository structure

```
minimal_vlm/     Part I — from-scratch VLM notebooks + shared model/eval code
poc/             Part II — local VLM field-extraction pipeline (Ollama)
visual_rag/      Parts III & IV — retrieval (ColQwen2) + answer generation (Ollama/OpenRouter)
figures/         Report figure-generation scripts (matplotlib), output PNGs
data/            Datasets (gitignored — see "Datasets" below; you populate this locally)
```

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

**Main environment** (everything except Part IV's retrieval indexing/search):

```bash
uv sync
```

**Retrieval environment** (`colpali-engine` needs a `torch`/`transformers` combination that
conflicts with the main project's pins, so it lives in a second, isolated venv — used only by
`visual_rag/build_index.py` and `visual_rag/retrieve.py`):

```bash
uv venv .venv-colpali --python 3.11
uv pip install --python .venv-colpali/bin/python colpali-engine==0.3.17 pillow "transformers==5.8.0"
```

> `transformers>=5.9` silently corrupts ColQwen2's weight loading for this checkpoint (missing
> keys fall back to random init instead of raising an error) — `5.8.0` is a deliberate pin, not
> arbitrary. Verify a clean load (no `MISSING`/`UNEXPECTED` keys reported) if you ever bump it.

**Local model serving**: [Ollama](https://ollama.com) running locally, with the models you
intend to use pulled (e.g. `ollama pull qwen2.5vl:7b`).

**Cloud models (optional)**: copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` if you
want to run the cloud-model comparisons (routed through [OpenRouter](https://openrouter.ai),
one API for every cloud VLM used here).

## Datasets

None of these are included in the repo (see `.gitignore`). Download each one and place it at
the exact path shown — some need the archive extracted as-is, others need a small manual
reorganization (noted below).

| Dataset | Used by | Source | Place at |
|---|---|---|---|
| Flickr8k | Part I | [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) | `data/flickr8k/Images/`, `data/flickr8k/captions.txt` |
| PAN card (entity detection) | Parts II–IV | [Kaggle](https://www.kaggle.com/datasets/nagendra048/pan-card-dataset) | `data/PAN_Kaggle/train/images/` **and** `data/PAN_Kaggle/valid/images/` |
| Aadhaar card (entity detection) | Parts II–IV | [Kaggle](https://www.kaggle.com/datasets/nagendra048/aadhar-dataset) | `data/Adhaar_Kaggle/train/images/` (note: "Adhaar" spelling in the path) |
| Rental agreements (`.docx`) | Parts III–IV | [Kaggle](https://www.kaggle.com/datasets/rbdbhardwaj123/rentalagreementsdocs) | `data/rental_agreements_raw/` (flat, no subfolders) |
| ChartQA | Parts III–IV | [HuggingFace](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | `data/ChartQA/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet` (filename must match exactly) |
| SPIQA | Parts III–IV | [HuggingFace](https://huggingface.co/datasets/google/spiqa) | `data/SPIQA/test-A/SPIQA_testA.json`, `data/SPIQA/test-A/SPIQA_testA_Images.zip` (left zipped — `prepare_spiqa.py` extracts it) |
| MMLongBench-Doc | Parts III–IV | [HuggingFace](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc) | `data/MMLongBench_Doc/data/train-00000-of-00001.parquet`, plus each document's PDF in `data/MMLongBench_Doc/documents/` (filenames must match the parquet's `doc_id` column) |

Notes:
- PAN specifically needs **both** `train/images` and `valid/images` populated — the extraction
  pipeline pools images from both and will fail if `valid/` is missing.
- SPIQA's zip is extracted and flattened automatically by `visual_rag/prepare_spiqa.py` — don't
  unzip it yourself.
- MMLongBench-Doc's PDFs are not bundled in the HuggingFace parquet and must be added manually;
  any `doc_id` without a matching PDF in `documents/` is silently skipped.

## Running it

**Part I** (`minimal_vlm/`): work through `minimal_vlm_from_scratch.ipynb`, then
`minimal_vlm_llava_style.ipynb`, then `minimal_vlm_llava_stage2_lora.ipynb`, in order — each
stage builds on the previous one's saved outputs. `compare_results.ipynb` aggregates the three
stages' generated captions and BLEU-4 scores side by side.

**Part II** (`poc/`): `run_local_comparison.py` runs the local-model extraction sweep over the
sampled PAN/Aadhaar images (resumable, writes to `poc/local_results.json`);
`model_comparison_report.ipynb` aggregates it into a summary table.

**Part III & IV** (`visual_rag/`), per dataset:

```bash
# 1. Build the page manifest + question set (main venv) -- see prepare_<dataset>.py for each
uv run python visual_rag/prepare_<dataset>.py

# 2. Build the ColQwen2 retrieval index (isolated venv)
.venv-colpali/bin/python visual_rag/build_index.py --dataset <dataset>

# 3. Retrieve top-k pages per question (isolated venv)
.venv-colpali/bin/python visual_rag/retrieve.py --dataset <dataset>

# 4. Generate answers -- Part III (oracle, no retrieval) and/or Part IV (retrieved) (main venv)
uv run python visual_rag/run_oracle_qa.py --dataset <dataset>
uv run python visual_rag/run_visual_rag.py --dataset <dataset> [--cloud]

# 5. Aggregate metrics
uv run python visual_rag/compute_metrics.py
```

Interactive demos: `uv run python visual_rag/ask_image_ui.py` (single-image Q&A, Part III) and
`uv run python visual_rag/ask_rag_ui.py` (retrieve-then-generate, Part IV) — both Gradio apps.
