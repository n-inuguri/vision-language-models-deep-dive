"""
Idea #2: have the same 5 pretrained VLMs from the Week 3 POC caption the
identical fixed 10-image Flickr8k set used to evaluate the Week 2 minimal-VLM
stages, and score them with the same corpus BLEU-4 harness (eval_utils.py) --
extending the from-scratch -> Stage 1 -> Stage 2 ladder with 5 more rows, all
directly comparable since every stage/model is scored on the same 10 images
against the same reference captions.

Not a privacy-sensitive run (Flickr8k is generic public photos, no PII
concerns) -- results save straight into the shared comparison_results.json.
"""
import sys
import time

import ollama

sys.path.insert(0, ".")
from eval_utils import build_reference_lookup, corpus_bleu4, save_result

IMAGE_DIR = "../data/flickr8k/Images"
CAPTIONS_FILE = "../data/flickr8k/captions.txt"
CAPTION_PROMPT = "Write a single, short caption describing this image, in plain English."

MODELS = ["qwen2.5vl:7b", "qwen2.5vl:3b", "llama3.2-vision:11b", "minicpm-v", "llava-phi3"]

# Same 10 images every Week 2 notebook was scored on (from comparison_results.json).
FIXED_IMAGE_NAMES = [
    "1459032057_97e73ed6ab.jpg", "1178705300_c224d9a4f1.jpg", "2206594874_5e0087c6b7.jpg",
    "3376809186_4e26d880b7.jpg", "783994497_4f6885454d.jpg", "2455528149_6c3477fd33.jpg",
    "1294578091_2ad02fea91.jpg", "483841513_e660391880.jpg", "3270047169_2ed289a9af.jpg",
    "3711611500_ea47b58b6f.jpg",
]


def caption_image(model_name, image_path):
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [str(image_path)]}],
    )
    return response.message.content.strip()


def main():
    reference_lookup = build_reference_lookup(CAPTIONS_FILE)

    for model_name in MODELS:
        hypotheses = []
        t0 = time.time()
        for image_name in FIXED_IMAGE_NAMES:
            image_path = f"{IMAGE_DIR}/{image_name}"
            caption = caption_image(model_name, image_path)
            hypotheses.append(caption)
        elapsed = time.time() - t0

        references_list = [reference_lookup[name] for name in FIXED_IMAGE_NAMES]
        bleu = corpus_bleu4(hypotheses, references_list)

        results = {"bleu4": bleu, "image_names": FIXED_IMAGE_NAMES, "generated_captions": hypotheses}
        save_result(model_name, results)
        print(f"  ({elapsed:.1f}s total, {elapsed/len(FIXED_IMAGE_NAMES):.1f}s/image)\n")


if __name__ == "__main__":
    main()
