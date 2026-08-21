"""One-off script to generate the Part IV example figure embedded in
TECH_REPORT_EXTENDED.md: a full retrieve-then-generate example matching
what visual_rag/ask_rag_ui.py's live demo shows in one screen -- the
question, the retrieved-pages gallery with ColPali/ColQwen2 scores (the
indexing stage), and the final generated answer -- pulled from the real
caches in visual_rag/retrieval_cache/chartqa.json and
visual_rag/rag_cache/chartqa__qwen2.5vl_7b.jsonl. Not part of the pipeline;
run manually if the underlying example ever changes.

Uses ChartQA (public benchmark data, no PII) rather than PAN/Aadhaar
specifically to avoid the masking complexity: swapping masked images into
the actual retrieval corpus for a screenshot would risk altering what
ColQwen2 embeds and retrieves, for no real benefit here (the retrieval
mechanism is identical regardless of which dataset demonstrates it).

    uv run python figures/make_part4_example.py
"""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE = "#2a78d6"
GRID = "#e3e2dd"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

QUESTION_ID = "chartqa__7c4f367bed9a1a756b6a1aacefc0cc19__1"
retrieval = json.loads((REPO_ROOT / "visual_rag" / "retrieval_cache" / "chartqa.json").read_text())[QUESTION_ID]

rag_record = None
for line in (REPO_ROOT / "visual_rag" / "rag_cache" / "chartqa__qwen2.5vl_7b.jsonl").read_text().splitlines():
    rec = json.loads(line)
    if rec["question_id"] == QUESTION_ID:
        rag_record = rec
        break

QUESTION = retrieval["question"]
TOP_K_PATHS = retrieval["top_k_image_paths"]
TOP_K_DOC_IDS = retrieval["top_k_doc_ids"]
SCORES = retrieval["scores"]
CORRECT_DOC_ID = rag_record["retrieved_doc_ids"][0]  # top-1 happens to be the true document here
ANSWER = rag_record["answer"]
LATENCY = f"{rag_record['latency']:.2f}s"
COST = "local model, no cost" if not rag_record.get("usage") else str(rag_record["usage"])
MODEL = rag_record["model"]

fig = plt.figure(figsize=(11, 6.3))
gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1], hspace=0.3, top=0.96, bottom=0.08)

# --- Top: question/model on the left, answer/latency/cost on the right ---
gs_top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.3])
ax_q = fig.add_subplot(gs_top[0, 0])
ax_a = fig.add_subplot(gs_top[0, 1])
for ax in (ax_q, ax_a):
    ax.axis("off")

ax_q.set_title("ask_rag -- retrieve then generate (Part IV)", fontsize=10, color=TEXT_PRIMARY, loc="left")
ax_q.text(0.0, 0.85, "Dataset", fontsize=9.5, color=TEXT_SECONDARY, va="top")
ax_q.text(0.0, 0.75, "chartqa", fontsize=11, color=TEXT_PRIMARY, va="top")
ax_q.text(0.0, 0.58, "Question", fontsize=9.5, color=TEXT_SECONDARY, va="top")
ax_q.text(0.0, 0.48, "\n".join(textwrap.wrap(QUESTION, 38)), fontsize=11, color=TEXT_PRIMARY, va="top", linespacing=1.5)
ax_q.text(0.0, 0.18, "Model", fontsize=9.5, color=TEXT_SECONDARY, va="top")
ax_q.text(0.0, 0.08, MODEL, fontsize=11, color=TEXT_PRIMARY, va="top")

ax_a.set_title(" ", fontsize=10)
rows = [("Answer", "\n".join(textwrap.wrap(ANSWER, 60))), ("Latency", LATENCY), ("Cost", COST)]
y = 0.95
for label, value in rows:
    ax_a.text(0.0, y, label, fontsize=9.5, color=TEXT_SECONDARY, va="top", transform=ax_a.transAxes)
    n_lines = value.count("\n") + 1
    ax_a.text(0.0, y - 0.09, value, fontsize=10.5, color=TEXT_PRIMARY, va="top",
              transform=ax_a.transAxes, linespacing=1.4)
    y -= 0.09 + 0.115 * n_lines + 0.06

# --- Bottom: retrieved-pages gallery (the indexing stage) ---
bottom_top = gs[1].get_position(fig).y1
gs_bottom = gs[1].subgridspec(1, 5, wspace=0.15)
fig.text(0.02, bottom_top - 0.02, "Retrieved pages (top-5, ColPali/ColQwen2 scores)", fontsize=10, color=TEXT_PRIMARY, va="top")
for i, (path, doc_id, score) in enumerate(zip(TOP_K_PATHS, TOP_K_DOC_IDS, SCORES)):
    ax = fig.add_subplot(gs_bottom[0, i])
    img = Image.open(path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    is_correct = doc_id == CORRECT_DOC_ID
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLUE if is_correct else GRID)
        spine.set_linewidth(3 if is_correct else 1)
    label = f"#{i+1}  score {score:.2f}" + ("  (correct doc)" if is_correct else "")
    ax.set_xlabel(label, fontsize=8.5, color=BLUE if is_correct else TEXT_SECONDARY)

plt.savefig(REPO_ROOT / "figures" / "part4_example_ask_rag.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Wrote figures/part4_example_ask_rag.png")
