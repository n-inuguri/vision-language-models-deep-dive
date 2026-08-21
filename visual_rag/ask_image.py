"""
Ad-hoc single-image VLM Q&A: no dataset, no manifest, no retrieval -- just
(image_path, question, model) -> answer. This is the primitive every other
visual_rag script is built on (rag_lib.generate_answer()), exposed directly
for live demos and one-off spot checks. If the answer isn't in the image,
the model is expected to say so -- no special handling needed for that,
it's just what a VLM does when asked something the image doesn't show.

Single-image only for now: fine for PAN/Aadhaar/ChartQA/SPIQA (always
one image per document) and for individual pages of rental_agreements/
mmlongbench_doc, but this script does not bundle multiple pages of a
multi-page document into one call.

Interactive mode (recommended for live use):
    uv run python visual_rag/ask_image.py

Non-interactive / scriptable mode (skips all prompts):
    uv run python visual_rag/ask_image.py --image <path> --question "..." --model qwen2.5vl:7b

Runs in the MAIN venv.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

CURATED_PATH = Path(__file__).resolve().parent / "curated_demo_qa.json"


def load_curated():
    if not CURATED_PATH.exists():
        return []
    return json.loads(CURATED_PATH.read_text())


def prompt_choice(label, options, formatter=str):
    print(f"\n{label}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {formatter(opt)}")
    while True:
        raw = input(f"Enter 1-{len(options)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice, try again.")


def open_image(image_path):
    try:
        subprocess.run(["open", str(image_path)], check=False)
    except FileNotFoundError:
        print(f"  (couldn't auto-open the image -- view it manually at {image_path})")


def pick_image_and_question():
    curated = load_curated()
    mode_options = ["Pick from curated demo set", "Enter a custom image path"]
    mode = prompt_choice("How do you want to pick an image?", mode_options)

    if mode == mode_options[0] and curated:
        item = prompt_choice(
            "Curated examples:",
            curated,
            formatter=lambda it: f"[{it['dataset']}] {it['question']}",
        )
        print(f"  Image: {item['image_path']}")
        open_image(item["image_path"])
        if item.get("reference_answer") is not None:
            print(f"  (reference answer, for your own checking: {item['reference_answer']!r})")
        else:
            print("  (this one is intentionally unanswerable from the image)")
        question = input(f"Question [{item['question']}]: ").strip() or item["question"]
        return item["image_path"], question

    while True:
        image_path = input("Image path: ").strip().strip('"').strip("'")
        if Path(image_path).exists():
            break
        print(f"  No file at {image_path!r}, try again.")
    open_image(image_path)
    question = input("Question: ").strip()
    return image_path, question


def pick_model():
    def fmt(m):
        tag = "local" if rag_lib.MODEL_PROVIDER[m] == "ollama" else "cloud"
        return f"{m}  [{tag}]"
    return prompt_choice("Which model?", rag_lib.ANSWER_MODELS, formatter=fmt)


def ask(image_path, question, model_name):
    answer, latency, usage = rag_lib.generate_answer([image_path], question, model_name)
    return answer, latency, usage


def print_result(answer, latency, usage):
    print(f"\nAnswer: {answer}")
    print(f"Latency: {latency:.2f}s")
    if usage:
        print(f"Cost: ${usage['cost_usd']:.5f}  (in={usage['input_tokens']} out={usage['output_tokens']})")


def interactive():
    print("=== ask_image.py -- single-image VLM Q&A, no retrieval ===")
    image_path, question = pick_image_and_question()
    while True:
        model_name = pick_model()
        try:
            answer, latency, usage = ask(image_path, question, model_name)
            print_result(answer, latency, usage)
        except Exception as e:
            print(f"\nERROR: {e}")

        again_options = [
            "Ask another question on this image",
            "Try a different model on this same question",
            "Pick a new image",
            "Quit",
        ]
        choice = prompt_choice("What next?", again_options)
        if choice == again_options[0]:
            question = input("Question: ").strip()
        elif choice == again_options[1]:
            continue
        elif choice == again_options[2]:
            image_path, question = pick_image_and_question()
        else:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to an image (skips interactive picking)")
    parser.add_argument("--question", help="Question text (skips interactive picking)")
    parser.add_argument("--model", choices=rag_lib.ANSWER_MODELS, help="Model name (skips interactive picking)")
    args = parser.parse_args()

    if args.image and args.question and args.model:
        answer, latency, usage = ask(args.image, args.question, args.model)
        print_result(answer, latency, usage)
    else:
        interactive()
