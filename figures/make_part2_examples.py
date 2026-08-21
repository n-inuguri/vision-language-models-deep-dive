"""One-off script to generate the two Part II document-extraction example
figures embedded in TECH_REPORT_EXTENDED.md: a masked PAN/Aadhaar card image
next to the exact JSON the extraction pipeline returned for it. Not part of
the pipeline; run manually if the underlying example images or extraction
results ever change.

Masks the number-like fields (PAN number, DOB; Aadhaar number, DOB) directly
on the card using the dataset's own YOLO bounding-box labels, per
practice-head guidance: mask numbers, names are fine to show. The masked
source images are also saved to poc/masked_examples/, kept separate from the
data/PAN_Kaggle and data/Adhaar_Kaggle download folders. Masked fields are
never printed as their real value in the JSON either -- shown as a run of
'x' the same length as the real value, since printing the real value next to
a masked image would defeat the masking.

    uv run python figures/make_part2_examples.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
MASKED_DIR = REPO_ROOT / "poc" / "masked_examples"
MASKED_DIR.mkdir(exist_ok=True)

TEXT_PRIMARY = "#0b0b0b"
MASK_FILL = "#1a1a1a"

plt.rcParams.update({
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_yolo_boxes(label_path):
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = (float(p) for p in parts[1:5])
        boxes.append((cls, xc, yc, bw, bh))
    return boxes


def mask_image(image_path, label_path, mask_classes, out_path, extra_boxes=()):
    """extra_boxes: manually-identified pixel-space (x1, y1, x2, y2) regions to
    mask that aren't covered by a YOLO label -- specifically, the QR code both
    PAN and Aadhaar cards print, which re-encodes the same number/DOB fields
    in machine-readable form. Masking the printed number alone and leaving the
    QR code visible would just leak the same value back via any QR scanner."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for cls, xc, yc, bw, bh in load_yolo_boxes(label_path):
        if cls not in mask_classes:
            continue
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        pad = max(3, int(0.2 * (y2 - y1)))
        draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), fill=MASK_FILL)
    for x1, y1, x2, y2 in extra_boxes:
        draw.rectangle((x1, y1, x2, y2), fill=MASK_FILL)
    img.save(out_path)


def make_panel(masked_image_path, fields, mask_field_names, out_path):
    """Right-hand panel shows the actual JSON the extraction pipeline
    returns for this image, exactly as it appears in poc/real_data_cache/,
    except masked fields are replaced with 'x' * len(real_value) -- same
    shape as the real output, no real value leaked."""
    img = Image.open(masked_image_path)
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(9.5, 4), gridspec_kw={"width_ratios": [1.1, 1]})

    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("Input (numbers masked)", fontsize=10, color=TEXT_PRIMARY, loc="left")

    ax_text.axis("off")
    ax_text.set_title("Extracted output (qwen2.5vl:7b)", fontsize=10, color=TEXT_PRIMARY, loc="left")
    display_fields = {
        field: ("x" * len(value) if field in mask_field_names else value)
        for field, value in fields.items()
    }
    json_text = json.dumps(display_fields, indent=2, ensure_ascii=False)
    ax_text.text(0.02, 0.88, json_text, fontsize=10.5, color=TEXT_PRIMARY, family="monospace",
                 va="top", ha="left", transform=ax_text.transAxes, linespacing=1.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --- PAN example (real qwen2.5vl:7b output from poc/real_data_cache) ---
PAN_STEM = "1d685e4f-photo_4_2023-07-14_15-27-27_jpg.rf.e700bb6bc5bedb4e6bd640ff1ab291e6"
PAN_IMAGE = REPO_ROOT / "data" / "PAN_Kaggle" / "train" / "images" / f"{PAN_STEM}.jpg"
PAN_LABEL = REPO_ROOT / "data" / "PAN_Kaggle" / "train" / "labels" / f"{PAN_STEM}.txt"
PAN_MASK_CLASSES = {0, 3}  # data.yaml: 0=dob, 3=pan
PAN_QR_BOX = (46, 48, 104, 104)  # manually identified; not a YOLO-labeled class
PAN_MASKED_OUT = MASKED_DIR / "pan_example_masked.jpg"
PAN_FIELDS = {
    "name": "SAMRAT CHADDA",
    "father_name": "VIJAY KUMAR CHADDA",
    "dob": "15/02/1985",
    "pan_number": "BOHPC6420G",
}
PAN_MASK_FIELDS = {"dob", "pan_number"}

mask_image(PAN_IMAGE, PAN_LABEL, PAN_MASK_CLASSES, PAN_MASKED_OUT, extra_boxes=[PAN_QR_BOX])
make_panel(
    PAN_MASKED_OUT, PAN_FIELDS, PAN_MASK_FIELDS,
    REPO_ROOT / "figures" / "part2_pan_extraction_example.png",
)

# --- Aadhaar example (real qwen2.5vl:7b output; class mapping determined by
# visually inspecting each box crop, since data.yaml only names classes 0-4) ---
AADHAAR_STEM = "14171f7e3e906f9bb9cbee0e44b183ab-1-_jpg.rf.2e207c79621f8d1612c7130a3001e1c1"
AADHAAR_IMAGE = REPO_ROOT / "data" / "Adhaar_Kaggle" / "train" / "images" / f"{AADHAAR_STEM}.jpg"
AADHAAR_LABEL = REPO_ROOT / "data" / "Adhaar_Kaggle" / "train" / "labels" / f"{AADHAAR_STEM}.txt"
AADHAAR_MASK_CLASSES = {0, 1}  # 0=aadhaar_number, 1=dob (2=gender, 3=name stay visible)
AADHAAR_QR_BOX = (470, 160, 640, 480)  # manually identified; not a YOLO-labeled class
AADHAAR_MASKED_OUT = MASKED_DIR / "aadhaar_example_masked.jpg"
AADHAAR_FIELDS = {
    "name": "Renuka Vinod Sonar",
    "dob": "11/05/1996",
    "gender": "FEMALE",
    "aadhaar_number": "320692446504",
}
AADHAAR_MASK_FIELDS = {"dob", "aadhaar_number"}

mask_image(AADHAAR_IMAGE, AADHAAR_LABEL, AADHAAR_MASK_CLASSES, AADHAAR_MASKED_OUT, extra_boxes=[AADHAAR_QR_BOX])
make_panel(
    AADHAAR_MASKED_OUT, AADHAAR_FIELDS, AADHAAR_MASK_FIELDS,
    REPO_ROOT / "figures" / "part2_aadhaar_extraction_example.png",
)

print("Wrote poc/masked_examples/pan_example_masked.jpg, poc/masked_examples/aadhaar_example_masked.jpg")
print("Wrote figures/part2_pan_extraction_example.png, figures/part2_aadhaar_extraction_example.png")
