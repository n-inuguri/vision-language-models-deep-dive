"""One-off script to generate the Part III example figure embedded in
TECH_REPORT_EXTENDED.md: a single-image Q&A example matching what
visual_rag/ask_image_ui.py's live demo shows (image, question, model,
answer, latency, cost). Not part of the pipeline; run manually if the
underlying example ever changes.

Reuses the same masked PAN card from Part II's example (poc/masked_examples/
pan_example_masked.jpg -- run figures/make_part2_examples.py first). Uses a
plain question ("What is the father's name printed on this card?") rather
than the study's actual anchored-question wording (Section 6.2, which
anchors on the card's own PAN number, needed for Part IV's retrieval to have
distinguishing signal) -- for a screenshot alone there is no retrieval step
to serve, so the anchor would only add an unnecessary number to redact. The
evaluation itself is unchanged; this is illustration only. Answer/latency
below are a real, live call against the (unmasked) source image via
rag_lib.generate_answer, the same function ask_image_ui.py uses.

    uv run python figures/make_part3_example.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MASKED_PAN_IMAGE = REPO_ROOT / "poc" / "masked_examples" / "pan_example_masked.jpg"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Real, live call via rag_lib.generate_answer (same function ask_image_ui.py
# uses) against the unmasked source image -- masking is a report-display
# concern only, not something the model itself should ever be handed.
QUESTION_DISPLAY = "What is the father's name printed on this card?"
ANSWER_DISPLAY = 'The father\'s name printed on the card is "Vijay Kumar Chadda."'
MODEL = "qwen2.5vl:7b"
LATENCY = "6.28s"
COST = "local model, no cost"

fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})

img = Image.open(MASKED_PAN_IMAGE)
ax_img.imshow(img)
ax_img.axis("off")
ax_img.set_title("Image (numbers masked)", fontsize=10, color=TEXT_PRIMARY, loc="left")

ax_text.axis("off")
ax_text.set_title("ask_image -- single-image Q&A, no retrieval (Part III)", fontsize=10, color=TEXT_PRIMARY, loc="left")


def wrap(text, width=52):
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


rows = [
    ("Question", wrap(QUESTION_DISPLAY)),
    ("Model", MODEL),
    ("Answer", wrap(ANSWER_DISPLAY)),
    ("Latency", LATENCY),
    ("Cost", COST),
]
y = 0.90
for label, value in rows:
    ax_text.text(0.02, y, label, fontsize=9.5, color=TEXT_SECONDARY, va="top", ha="left", transform=ax_text.transAxes)
    n_lines = value.count("\n") + 1
    ax_text.text(0.02, y - 0.06, value, fontsize=11, color=TEXT_PRIMARY, va="top", ha="left",
                 transform=ax_text.transAxes, linespacing=1.5)
    y -= 0.06 + 0.09 * n_lines + 0.06

plt.tight_layout()
plt.savefig(REPO_ROOT / "figures" / "part3_example_ask_image.png", dpi=150)
plt.close(fig)

print("Wrote figures/part3_example_ask_image.png")
