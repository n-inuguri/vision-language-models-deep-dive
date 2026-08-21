"""
Interactive CLI for manually transcribing ground truth on a small real-image
subsample, so we have one real (not proxy) accuracy number to validate the
cross-model agreement metric against.

Deliberately a script, not a notebook: opens each image in Preview rather than
displaying it inline, so a real document image is never embedded in a saved
.ipynb file's output.

Usage:
    uv run python poc/label_ground_truth.py --doc-type pan --n 15
    uv run python poc/label_ground_truth.py --doc-type aadhaar --n 15

Saves to poc/manual_ground_truth_cache/{doc_type}.json (gitignored, real PII,
never committed). Safe to stop and resume -- already-labeled images are
skipped and shown images are drawn from the same fixed sample() used by
run_local_comparison.py, so this ground truth can be joined against the
per-model extraction cache later.

To reject an image (bad scan quality, you're unsure of a field, whatever the
reason), type `skip` as the answer to the first field -- it's recorded in
poc/manual_ground_truth_cache/{doc_type}_skipped.json (also gitignored) and
never shown again, and the script moves on to the next unlabeled image from
the same fixed 100-image pool.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extraction_lib import FIELD_NAMES, sample_images

GT_DIR = Path(__file__).parent / "manual_ground_truth_cache"


def load_existing(doc_type):
    path = GT_DIR / f"{doc_type}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save(doc_type, ground_truth):
    GT_DIR.mkdir(parents=True, exist_ok=True)
    path = GT_DIR / f"{doc_type}.json"
    path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")


def load_skipped(doc_type):
    path = GT_DIR / f"{doc_type}_skipped.json"
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set()


def save_skipped(doc_type, skipped):
    GT_DIR.mkdir(parents=True, exist_ok=True)
    path = GT_DIR / f"{doc_type}_skipped.json"
    path.write_text(json.dumps(sorted(skipped), indent=2), encoding="utf-8")


def label(doc_type, n):
    fields = FIELD_NAMES[doc_type]
    ground_truth = load_existing(doc_type)
    skipped = load_skipped(doc_type)

    # force_include only for aadhaar: guarantees already-labeled images stay in the
    # pool that run_local_comparison.py will actually process. Never do this for
    # pan -- it would shift which *other* files get sampled and invalidate the
    # already-cached PAN extraction progress (see sample_images' docstring).
    force_include = list(ground_truth.keys()) if doc_type == "aadhaar" else None
    pool = sample_images(doc_type, n=100, force_include=force_include)  # full 100-image pool, not just the first n

    # Candidates: everything in the fixed pool not already labeled or skipped.
    # This is what makes "delete a bad entry and get a *different* image" work --
    # the pool has 100 candidates to draw from, not just a fixed first-n window.
    candidates = [img for img in pool if img.name not in ground_truth and img.name not in skipped]
    still_needed = max(0, n - len(ground_truth))
    todo = candidates[:still_needed]

    print(f"{doc_type}: {len(ground_truth)} already labeled, {len(skipped)} skipped, "
          f"{len(todo)} to go (target {n})\n")

    for i, image_path in enumerate(todo, 1):
        subprocess.run(["open", str(image_path)])
        print(f"\n[{i}/{len(todo)}] {image_path.name} (opened in Preview)")
        entry = {}
        try:
            first_field = fields[0]
            first_value = input(f"  {first_field} (or 'skip' to reject this image): ").strip()
            if first_value.lower() == "skip":
                skipped.add(image_path.name)
                save_skipped(doc_type, skipped)
                print("  Skipped -- won't be shown again.")
                continue
            entry[first_field] = first_value
            for field in fields[1:]:
                entry[field] = input(f"  {field}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nStopping early -- saving progress so far.")
            break
        ground_truth[image_path.name] = entry
        save(doc_type, ground_truth)  # save after every image, not just at the end

    print(f"\nSaved {len(ground_truth)} labeled images to {GT_DIR / f'{doc_type}.json'}")
    if len(ground_truth) < n and len(todo) < still_needed:
        print(f"Note: pool exhausted -- only {len(candidates)} unlabeled/unskipped "
              f"candidates were available in the fixed 100-image sample.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-type", choices=["pan", "aadhaar"], required=True)
    parser.add_argument("--n", type=int, default=15, help="How many images to label")
    args = parser.parse_args()
    label(args.doc_type, args.n)
