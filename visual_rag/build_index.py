"""
Build/persist a ColQwen2 retrieval index for one dataset's page manifest.

Runs under .venv-colpali (NOT the main project .venv) -- colpali-engine's
torch/transformers pins conflict with the main project's. Invoke as:

    .venv-colpali/bin/python visual_rag/build_index.py --dataset chartqa

Skips (re-)embedding if the persisted index's recorded image list already
matches the current page manifest -- embedding a bounded subsample is a
one-off job, not a multi-hour sweep, so this dataset-level resumability is
enough (unlike poc/run_local_comparison.py's per-item resumability, which
was needed for an actual multi-hour run).

REQUIRED transformers pin: colpali-engine==0.3.17's vidore/colqwen2-v1.0
checkpoint uses transformers's older flat Qwen2VL weight names
(model.embed_tokens/model.norm). transformers>=5.9 restructured these under
a nested language_model.* submodule; from_pretrained() doesn't error on the
mismatch, it silently random-inits the (missing) token-embedding table and
final layer norm and continues, producing near-flat, content-insensitive
retrieval scores with no warning beyond an easily-missed load report.
Confirmed clean load + real query/doc score separation at transformers==5.8.0
(5.9-5.14 all reproduce the bug; 5.10.0 is additionally a yanked PyPI
release). Re-pin if this ever needs bumping:
    uv pip install --python .venv-colpali/bin/python "transformers==5.8.0"
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402


def _load_model():
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {rag_lib.COLPALI_MODEL_NAME} on {device} ...")
    model = ColQwen2.from_pretrained(
        rag_lib.COLPALI_MODEL_NAME,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(rag_lib.COLPALI_MODEL_NAME)
    return model, processor


def build_index(dataset, rebuild=False):
    import torch
    from PIL import Image

    manifest = rag_lib.load_manifest(dataset)
    if not manifest:
        raise SystemExit(
            f"No page manifest for {dataset!r} -- build it first "
            f"(see rag_lib.build_page_manifest_from_images/_pdfs, run in the main venv)."
        )

    out_dir = rag_lib.index_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "manifest_snapshot.json"
    embeddings_path = out_dir / "embeddings.pt"

    image_paths = [page["image_path"] for page in manifest]
    if not rebuild and meta_path.exists() and embeddings_path.exists():
        prior = json.loads(meta_path.read_text())
        if prior.get("image_paths") == image_paths:
            print(f"[{dataset}] index already up to date ({len(image_paths)} pages) -- no-op. "
                  f"Pass --rebuild to force.")
            return

    model, processor = _load_model()

    print(f"[{dataset}] embedding {len(image_paths)} pages ...")
    t0 = time.time()
    all_embeddings = []
    batch_size = 4
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch = processor.process_images(images).to(model.device)
        with torch.no_grad():
            batch_embeddings = model(**batch)
        all_embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
        print(f"  {min(i + batch_size, len(image_paths))}/{len(image_paths)} "
              f"({time.time() - t0:.1f}s elapsed)")

    torch.save(all_embeddings, embeddings_path)
    meta_path.write_text(json.dumps({"image_paths": image_paths, "dataset": dataset}, indent=2))
    print(f"[{dataset}] index built: {len(all_embeddings)} pages in {time.time() - t0:.1f}s "
          f"-> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(rag_lib.DATASETS))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    build_index(args.dataset, rebuild=args.rebuild)
