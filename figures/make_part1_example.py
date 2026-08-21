"""One-off script to generate the Part I qualitative-example figure embedded
in TECH_REPORT_EXTENDED.md: one Flickr8k image, its ground-truth caption,
and the caption each of the three architecture stages actually generated
for it -- the same content minimal_vlm/compare_results.ipynb's Section 3
cell displays, reproduced here as a static image for the report. Not part
of the pipeline; run manually if the underlying example ever changes.

    uv run python figures/make_part1_example.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Real output from comparison_results.json / compare_results.ipynb Section 3,
# for the fixed 10-image evaluation set's 1178705300_c224d9a4f1.jpg.
IMAGE_NAME = "1178705300_c224d9a4f1.jpg"
IMAGE_PATH = REPO_ROOT / "data" / "flickr8k" / "Images" / IMAGE_NAME
GROUND_TRUTH = "a brown dog jumping into a pool after a bloe ball ."
CAPTIONS = [
    ("Ground truth", GROUND_TRUTH),
    ("Stage 0 (from scratch)", "a a boy boy is is jumping jumping into into a a pool pool . ."),
    ("Stage 1 (LLaVA, projector only)", "A dog jumps out of a pool ."),
    ("Stage 2 (LoRA)", "A dog jumps into a pool ."),
]

fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(9.5, 4), gridspec_kw={"width_ratios": [1, 1.3]})

img = Image.open(IMAGE_PATH)
ax_img.imshow(img)
ax_img.axis("off")
ax_img.set_title(IMAGE_NAME, fontsize=9, color=TEXT_SECONDARY, loc="left")

ax_text.axis("off")
ax_text.set_title("Generated captions, same image, all three stages", fontsize=10, color=TEXT_PRIMARY, loc="left")
y = 0.85
for label, caption in CAPTIONS:
    ax_text.text(0.0, y, label, fontsize=9.5, color=TEXT_SECONDARY, va="top", transform=ax_text.transAxes)
    ax_text.text(0.0, y - 0.09, f'"{caption}"', fontsize=11, color=TEXT_PRIMARY, va="top",
                 transform=ax_text.transAxes)
    y -= 0.24

plt.tight_layout()
plt.savefig(REPO_ROOT / "figures" / "part1_example_captions.png", dpi=150)
plt.close(fig)

print("Wrote figures/part1_example_captions.png")
