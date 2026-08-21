"""
Browser UI for the full Experiment #2 pipeline: pick a dataset and model,
type a question, retrieve the best-matching page(s) live via ColPali (over
HTTP from retrieval_server.py, which must already be running), see the
retrieved thumbnails + filenames, then get the VLM's answer generated from
those retrieved images.

Requires retrieval_server.py running first (separate terminal, .venv-colpali):
    .venv-colpali/bin/python visual_rag/retrieval_server.py

Then, in the MAIN venv:
    uv run python visual_rag/ask_rag_ui.py

Opens at http://127.0.0.1:7861 (a different port from ask_image_ui.py's
7860, so both UIs can run side by side).
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

RETRIEVAL_SERVER = "http://127.0.0.1:8765"

MODEL_CHOICES = [
    f"{m}  [{'local' if rag_lib.MODEL_PROVIDER[m] == 'ollama' else 'cloud'}]"
    for m in rag_lib.ANSWER_MODELS
]
LABEL_TO_MODEL = dict(zip(MODEL_CHOICES, rag_lib.ANSWER_MODELS))


def retrieve(dataset, question, top_k=5):
    url = f"{RETRIEVAL_SERVER}/search?dataset={dataset}&question={urllib.parse.quote(question)}&top_k={top_k}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def ask(dataset, question, model_label):
    if not question or not question.strip():
        return [], "Type a question first.", "", ""
    if not model_label:
        return [], "Pick a model first.", "", ""
    model_name = LABEL_TO_MODEL[model_label]

    try:
        retrieved = retrieve(dataset, question)
    except urllib.error.URLError:
        return [], (
            f"ERROR: can't reach the retrieval server at {RETRIEVAL_SERVER}. "
            f"Start it first: .venv-colpali/bin/python visual_rag/retrieval_server.py"
        ), "", ""
    except Exception as e:
        return [], f"ERROR from retrieval server: {e}", "", ""

    image_paths = retrieved["top_k_image_paths"]
    gallery = [
        (path, f"{Path(path).name}  (score {score:.2f})")
        for path, score in zip(image_paths, retrieved["scores"])
    ]

    try:
        answer, latency, usage = rag_lib.generate_answer(image_paths, question, model_name)
    except Exception as e:
        return gallery, f"ERROR: {e}", "", ""

    latency_str = f"{latency:.2f}s"
    cost_str = (
        f"${usage['cost_usd']:.5f}  (in={usage['input_tokens']} out={usage['output_tokens']})"
        if usage
        else "local model, no cost"
    )
    return gallery, answer, latency_str, cost_str


with gr.Blocks(title="ask_rag") as demo:
    gr.Markdown("# ask_rag -- retrieve then generate (Experiment #2)")
    gr.Markdown(
        f"Requires `retrieval_server.py` running (`.venv-colpali`). "
        f"Talking to it at `{RETRIEVAL_SERVER}`."
    )

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dropdown = gr.Dropdown(
                list(rag_lib.DATASETS), label="Dataset", value=list(rag_lib.DATASETS)[0]
            )
            question_box = gr.Textbox(label="Question", lines=2)
            model_dropdown = gr.Dropdown(MODEL_CHOICES, label="Model", value=MODEL_CHOICES[0])
            ask_button = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
            latency_box = gr.Textbox(label="Latency", interactive=False)
            cost_box = gr.Textbox(label="Cost", interactive=False)

    gr.Markdown("### Retrieved pages")
    gallery_box = gr.Gallery(label="Top-k retrieved images", columns=5, height=250)

    ask_button.click(
        ask,
        inputs=[dataset_dropdown, question_box, model_dropdown],
        outputs=[gallery_box, answer_box, latency_box, cost_box],
    )

if __name__ == "__main__":
    demo.launch(server_port=7861)
