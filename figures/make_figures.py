"""One-off script to generate the three static figures embedded in
TECH_REPORT_EXTENDED.md. Not part of the pipeline; run manually if the
underlying numbers ever change.

    uv run python figures/make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axes(ax, hide_y_ticks=False):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if hide_y_ticks:
        ax.set_yticks([])


# --- Figure 1: Part I BLEU-4 progression ------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4))
stages = ["Stage 0\n(from scratch)", "Stage 1\n(frozen + projector)", "Stage 2\n(+ LoRA, greedy)", "Stage 2\n(+ LoRA, beam-4)"]
bleu = [0.0098, 0.2049, 0.4282, 0.4564]
bars = ax.bar(stages, bleu, color=BLUE, width=0.55, edgecolor="none")
for rect, v in zip(bars, bleu):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT_PRIMARY)
ax.set_ylabel("Corpus BLEU-4 (fixed 10-image set)")
ax.set_ylim(0, 0.52)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
style_axes(ax)
plt.tight_layout()
plt.savefig("figures/part1_bleu_progression.png", dpi=150)
plt.close(fig)

# --- Figure 2: Retrieval recall@1 / recall@5 across datasets -----------
datasets = ["Rental agreements\n(anchored)", "PAN\n(expanded)", "SPIQA", "ChartQA", "MMLongBench-Doc", "Aadhaar\n(expanded)"]
recall1 = [0.86, 0.70, 0.62, 0.60, 0.54, 0.04]
recall5 = [0.86, 0.84, 0.87, 0.65, 0.75, 0.15]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = range(len(datasets))
width = 0.36
b1 = ax.bar([i - width / 2 for i in x], recall1, width, label="recall@1", color=BLUE)
b2 = ax.bar([i + width / 2 for i in x], recall5, width, label="recall@5", color=ORANGE)
for rect, v in zip(list(b1) + list(b2), recall1 + recall5):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
ax.set_xticks(list(x))
ax.set_xticklabels(datasets, fontsize=9.5)
ax.set_ylabel("Recall")
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
style_axes(ax)
ax.legend(frameon=False, loc="upper right", ncol=2)
plt.tight_layout()
plt.savefig("figures/part4_retrieval_recall.png", dpi=150)
plt.close(fig)

# --- Figure 3: Oracle vs. retrieved accuracy, per dataset/model --------
rows = [
    ("ChartQA / qwen2.5vl:7b", 89, 48),
    ("ChartQA / Gemini 3.1 Pro", 87, 67),
    ("SPIQA / minicpm-v", 77, 67),
    ("SPIQA / Qwen2.5-VL-72B", 93, 87),
    ("Aadhaar / qwen2.5vl:7b", 22, 21),
    ("Aadhaar / qwen2.5vl:3b", 18, 34),
    ("Aadhaar / llama3.2-vision:11b", 13, 6),
    ("Aadhaar / minicpm-v", 32, 31),
    ("Aadhaar / llava-phi3", 18, 2),
    ("PAN / minicpm-v", 56, 49),
    ("PAN / qwen2.5vl:7b", 57, 43),
    ("PAN / llava-phi3", 11, 1),
    ("Rental agreements / Gemini 3.1 Pro", 93, 48),
    ("Rental agreements / llama3.2-vision:11b", 65, 9),
]
rows = rows[::-1]  # top-to-bottom in the figure matches top-to-bottom in the report table
labels = [r[0] for r in rows]
oracle = [r[1] for r in rows]
retrieved = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(8.5, 7.5))
y = range(len(labels))
height = 0.36
ax.barh([i + height / 2 for i in y], oracle, height, label="Oracle (Part III, correct document handed directly)", color=BLUE)
ax.barh([i - height / 2 for i in y], retrieved, height, label="Retrieved (Part IV, ColPali finds the document)", color=ORANGE)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(0, 100)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
style_axes(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=9)
plt.tight_layout()
plt.savefig("figures/part4_oracle_vs_retrieved.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Figure 4: Part III best local vs. best cloud accuracy, per dataset ---
datasets3 = ["Rental\nagreements", "ChartQA", "SPIQA", "MMLongBench-\nDoc", "PAN\n(expanded)", "Aadhaar\n(expanded)"]
best_local = [77, 89, 77, 48, 57, 32]
best_cloud = [93, 87, 93, 80, 100, 7]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = range(len(datasets3))
width = 0.36
b1 = ax.bar([i - width / 2 for i in x], best_local, width, label="Best local model", color=BLUE)
b2 = ax.bar([i + width / 2 for i in x], best_cloud, width, label="Best cloud model", color=ORANGE)
for rect, v in zip(list(b1) + list(b2), best_local + best_cloud):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v}%", ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
ax.set_xticks(list(x))
ax.set_xticklabels(datasets3, fontsize=9.5)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 108)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
style_axes(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5)
plt.tight_layout()
plt.savefig("figures/part3_local_vs_cloud.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Wrote figures/part1_bleu_progression.png, figures/part3_local_vs_cloud.png, figures/part4_retrieval_recall.png, figures/part4_oracle_vs_retrieved.png")
