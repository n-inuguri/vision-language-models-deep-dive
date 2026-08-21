"""
Shared evaluation utilities for comparing the from-scratch, LLaVA-stage-1,
and LLaVA-stage-2 (LoRA) MiniVLM notebooks on an *identical* set of images,
using metrics that are actually comparable across models with different
vocabularies -- which teacher-forced loss/perplexity is not.

Why not just compare teacher-forced loss/perplexity across models?
--------------------------------------------------------------------
Two independent problems:

1. Different vocabularies -> different random-guessing floors.
   ln(vocab_size) is ~8.56 for the from-scratch notebook's ~5,240-word
   vocabulary vs. ~10.8 for SmolLM2's ~49K-token vocabulary. A loss of 2.45
   means something different against each floor.

2. A frozen, already-fluent LLM's loss is dominated by its own language
   prior, not by whether the caption is actually grounded in the image.
   SmolLM2 already assigns high probability to plausible English sentences
   before any image conditioning exists -- so teacher-forced loss can look
   "fine" even when the visual grounding is weak, especially early in
   projector-only training. Teacher-forced loss tells you "how surprised
   was the model by this exact reference sentence", not "did the model
   correctly describe this image."

What this module does instead
------------------------------
Runs *free generation* (the model's own greedy captions, not the
teacher-forced reference) on a fixed set of validation images, and scores
those generated captions against all known reference captions for each
image using corpus BLEU-4. BLEU-4 doesn't care about a model's internal
vocabulary size, so it's directly comparable across all three notebooks.
"""

import json
from collections import defaultdict
from pathlib import Path

# Shared results file all three notebooks write into -- since each notebook
# runs in its own kernel/process, this file (not any in-memory variable) is
# the only thing that can actually hold all three results at once.
COMPARISON_RESULTS_FILE = "comparison_results.json"


# Same validation indices used across all three notebooks. Because every
# notebook builds its dataset from the same captions.txt (same row order)
# and splits it with the same `generator=Generator().manual_seed(42)`,
# `val_set[i]` refers to the same underlying image in every notebook --
# this list is what actually pins that down explicitly and makes it
# impossible to accidentally drift.
FIXED_EVAL_INDICES = list(range(10))


def build_reference_lookup(captions_file):
    """Flickr8k has ~5 captions per image. BLEU is far more meaningful
    scored against *all* of an image's references than against just the
    one caption that happened to land in a given dataset row.

    Returns: dict[image_filename] -> list[str] (all captions for that image)
    """
    refs = defaultdict(list)
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines and lines[0].lower().startswith("image"):
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        image_name, caption = line.split(",", 1)
        refs[image_name.strip()].append(caption.strip())
    return refs


def get_image_name(val_subset, idx):
    """Both CaptionDataset implementations (from-scratch and LLaVA-style)
    store `self.samples = [(image_filename, caption), ...]` in the same
    shape, so this helper works unmodified for either notebook.

    val_subset is a torch.utils.data.Subset (what random_split returns).
    """
    underlying_dataset = val_subset.dataset
    real_idx = val_subset.indices[idx]
    image_name, _caption = underlying_dataset.samples[real_idx]
    return image_name


def corpus_bleu4(hypotheses, references_list):
    """Corpus-level BLEU-4 with simple whitespace/lowercase tokenization --
    deliberately dependency-light (no nltk data download required, unlike
    nltk's word_tokenize which needs the 'punkt' download).

    hypotheses:      list[str]       -- one generated caption per image
    references_list: list[list[str]] -- one list of reference captions per image
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    hyp_tok = [h.lower().split() for h in hypotheses]
    ref_tok = [[r.lower().split() for r in refs] for refs in references_list]
    # method1 smoothing avoids a score of 0 when a short caption is missing
    # just one higher-order n-gram -- appropriate for our short captions.
    smoothing = SmoothingFunction().method1
    return corpus_bleu(ref_tok, hyp_tok, smoothing_function=smoothing)


def evaluate_fixed_set(val_set, generate_and_decode_fn, reference_lookup,
                        indices=FIXED_EVAL_INDICES, verbose=True):
    """
    Runs free-generation captioning on a fixed set of validation examples
    and scores it with corpus BLEU-4 against each image's full reference set.

    generate_and_decode_fn(val_set, idx) -> str
        A callable you provide (differs per notebook, since the from-scratch
        model and the LLaVA-style model have different .generate()/decode
        signatures -- see the usage examples at the bottom of this file).
        Should return the model's freely-generated caption as plain text,
        for the example at position `idx` in val_set.

    Returns a dict with the BLEU-4 score plus per-image detail, so you can
    both quote a single headline number and show qualitative examples.
    """
    hypotheses, references_list, image_names = [], [], []
    for idx in indices:
        image_name = get_image_name(val_set, idx)
        caption = generate_and_decode_fn(val_set, idx)
        hypotheses.append(caption)
        references_list.append(reference_lookup[image_name])
        image_names.append(image_name)

    bleu = corpus_bleu4(hypotheses, references_list)

    if verbose:
        print(f"Corpus BLEU-4 over {len(indices)} fixed images: {bleu:.4f}\n")
        for name, hyp, refs in zip(image_names, hypotheses, references_list):
            print(f"  [{name}]")
            print(f"    generated: {hyp}")
            print(f"    reference: {refs[0]}")
        print()

    return {
        "bleu4": bleu,
        "image_names": image_names,
        "generated_captions": hypotheses,
        "reference_captions": references_list,
    }


def save_result(stage_name, results, path=COMPARISON_RESULTS_FILE):
    """Persist one notebook's evaluate_fixed_set() result into the shared
    comparison file, keyed by stage_name. Safe to call from any of the three
    notebooks in any order/kernel -- it merges into whatever's already on disk
    rather than overwriting other stages' entries.
    """
    path = Path(path)
    all_results = json.loads(path.read_text()) if path.exists() else {}
    all_results[stage_name] = {
        "bleu4": results["bleu4"],
        "image_names": results["image_names"],
        "generated_captions": results["generated_captions"],
    }
    path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved '{stage_name}' result ({results['bleu4']:.4f} BLEU-4) to {path}")


def load_comparison(path=COMPARISON_RESULTS_FILE):
    """Load whatever stage results have been saved so far. Returns {} if the
    file doesn't exist yet (i.e. no notebook has called save_result() yet)."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def print_comparison_table(path=COMPARISON_RESULTS_FILE):
    """The actual "where do I see the comparison" answer: prints every saved
    stage's BLEU-4 side by side, in the order they were saved."""
    all_results = load_comparison(path)
    if not all_results:
        print(f"No results saved yet at {path} -- call save_result(...) from each notebook first.")
        return
    print(f"{'Stage':<32} {'BLEU-4':>8}")
    print("-" * 42)
    for stage_name, r in all_results.items():
        print(f"{stage_name:<32} {r['bleu4']:>8.4f}")


# ---------------------------------------------------------------------------
# Usage examples (copy the relevant one into each notebook -- not run here,
# since it depends on that notebook's own model/tokenizer objects)
# ---------------------------------------------------------------------------

USAGE_EXAMPLE_FROM_SCRATCH = '''
# --- In minimal_vlm_from_scratch.ipynb, add a cell after Section 11 ---
from eval_utils import evaluate_fixed_set, build_reference_lookup

reference_lookup = build_reference_lookup(CAPTIONS_FILE)

def generate_and_decode(val_set, idx):
    image_tensor, _gt_token_ids = val_set.dataset[val_set.indices[idx]]
    generated_ids = model.generate(
        image_tensor.to(device), bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        max_new_tokens=MAX_TEXT_LEN,
    )
    return tokenizer.decode(generated_ids.tolist())

results = evaluate_fixed_set(val_set, generate_and_decode, reference_lookup)
'''

USAGE_EXAMPLE_LLAVA_STYLE = '''
# --- In minimal_vlm_llava_style.ipynb (or the stage-2 LoRA notebook), add a cell after Section 12 ---
from eval_utils import evaluate_fixed_set, build_reference_lookup

reference_lookup = build_reference_lookup(CAPTIONS_FILE)

def generate_and_decode(val_set, idx):
    pixel_values, _input_ids, _attention_mask = val_set.dataset[val_set.indices[idx]]
    generated_ids = model.generate(pixel_values.unsqueeze(0).to(device), max_new_tokens=MAX_TEXT_LEN)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

results = evaluate_fixed_set(val_set, generate_and_decode, reference_lookup)
'''
