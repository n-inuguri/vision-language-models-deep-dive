"""
Browser UI for ask_image.py: pick a curated example or upload your own
image, see it rendered in the browser, type/edit a question, pick a model,
get an answer. Same underlying primitive as ask_image.py (no dataset, no
retrieval, no manifest) -- rag_lib.generate_answer() -- just wrapped in a
Gradio front end for live demos instead of a terminal prompt loop.

Runs in the MAIN venv.

Usage:
    uv run python visual_rag/ask_image_ui.py

Opens at http://127.0.0.1:7860
"""

import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

CURATED_PATH = Path(__file__).resolve().parent / "curated_demo_qa.json"
CURATED = json.loads(CURATED_PATH.read_text()) if CURATED_PATH.exists() else []
CURATED_LABELS = [f"[{item['dataset']}] {item['question']}" for item in CURATED]

MODEL_CHOICES = [
    f"{m}  [{'local' if rag_lib.MODEL_PROVIDER[m] == 'ollama' else 'cloud'}]"
    for m in rag_lib.ANSWER_MODELS
]
LABEL_TO_MODEL = dict(zip(MODEL_CHOICES, rag_lib.ANSWER_MODELS))


def on_curated_select(label):
    if not label:
        return gr.update(), gr.update(), gr.update(value="")
    item = CURATED[CURATED_LABELS.index(label)]
    ref = (
        f"Reference answer (for your own checking): {item['reference_answer']!r}"
        if item.get("reference_answer") is not None
        else "This one is intentionally unanswerable from the image."
    )
    return item["image_path"], item["question"], ref


def ask(image_path, question, model_label):
    if not image_path:
        return "Pick or upload an image first.", "", ""
    if not question or not question.strip():
        return "Type a question first.", "", ""
    if not model_label:
        return "Pick a model first.", "", ""
    model_name = LABEL_TO_MODEL[model_label]
    try:
        answer, latency, usage = rag_lib.generate_answer([image_path], question, model_name)
    except Exception as e:
        return f"ERROR: {e}", "", ""
    latency_str = f"{latency:.2f}s"
    cost_str = (
        f"${usage['cost_usd']:.5f}  (in={usage['input_tokens']} out={usage['output_tokens']})"
        if usage
        else "local model, no cost"
    )
    return answer, latency_str, cost_str


with gr.Blocks(title="ask_image") as demo:
    gr.Markdown("# ask_image -- single-image VLM Q&A (no retrieval)")

    with gr.Row():
        with gr.Column(scale=1):
            curated_dropdown = gr.Dropdown(
                CURATED_LABELS, label="Curated example (optional)", value=None
            )
            reference_box = gr.Markdown("")
            image_box = gr.Image(label="Image (or upload your own)", type="filepath", height=400)
            question_box = gr.Textbox(label="Question", lines=2)
            model_dropdown = gr.Dropdown(MODEL_CHOICES, label="Model", value=MODEL_CHOICES[0])
            ask_button = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
            latency_box = gr.Textbox(label="Latency", interactive=False)
            cost_box = gr.Textbox(label="Cost", interactive=False)

    curated_dropdown.change(
        on_curated_select,
        inputs=curated_dropdown,
        outputs=[image_box, question_box, reference_box],
    )
    ask_button.click(
        ask,
        inputs=[image_box, question_box, model_dropdown],
        outputs=[answer_box, latency_box, cost_box],
    )

if __name__ == "__main__":
    demo.launch()
