"""
Prepares the Rental Agreements dataset (Kaggle rbdbhardwaj123/rentalagreementsdocs)
for the Visual RAG pipeline: renders each plain-text agreement into a
scanned-page-style image (PIL, no PDF intermediate), and builds the page
manifest.

MANDATORY FIRST STEP -- the PII checkpoint (do this before --render):

    uv run python visual_rag/prepare_rental_agreements.py --inspect 10

Opens 10 raw text files so you can manually check for real names/addresses/
signatures before rendering or labeling anything at scale. Until that check
is done, data/rental_agreements_raw/ and data/rental_agreements_rendered/
stay gitignored by default (see .gitignore) -- same trust boundary as
PAN_Kaggle/Adhaar_Kaggle until proven otherwise.

Usage (after the checkpoint):
    uv run python visual_rag/prepare_rental_agreements.py --render

Expects raw .txt files already manually downloaded into
data/rental_agreements_raw/ (no download scripts exist anywhere in this repo
-- matches the existing convention for every other dataset).
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

RAW_DIR = rag_lib.DATASETS["rental_agreements"]["raw_dir"]
RENDERED_DIR = rag_lib.DATASETS["rental_agreements"]["rendered_dir"]

PAGE_SIZE = (1275, 1650)  # ~8.5x11in @ 150dpi
MARGIN = 80
FONT_SIZE = 18
LINE_SPACING = 6


def _get_font():
    for candidate in (
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, FONT_SIZE)
    return ImageFont.load_default()


def _wrap_text(text, font, max_width):
    avg_char_width = font.getlength("m") or 10
    chars_per_line = max(20, int(max_width / avg_char_width))
    wrapped = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(paragraph, width=chars_per_line) or [""])
    return wrapped


def render_pages(text, font):
    """Splits one document's text across as many page images as needed."""
    usable_width = PAGE_SIZE[0] - 2 * MARGIN
    usable_height = PAGE_SIZE[1] - 2 * MARGIN
    lines = _wrap_text(text, font, usable_width)
    line_height = FONT_SIZE + LINE_SPACING
    lines_per_page = max(1, usable_height // line_height)

    pages = []
    for i in range(0, len(lines), lines_per_page):
        page_lines = lines[i:i + lines_per_page]
        img = Image.new("RGB", PAGE_SIZE, "white")
        draw = ImageDraw.Draw(img)
        y = MARGIN
        for line in page_lines:
            draw.text((MARGIN, y), line, font=font, fill="black")
            y += line_height
        pages.append(img)
    return pages


def _extract_text(docx_path):
    """The dataset's files are .docx (Scribd-scraped rental agreement templates,
    named like '<scribd_id>-<title>.pdf.docx'), not plain .txt."""
    import docx
    document = docx.Document(docx_path)
    return "\n".join(p.text for p in document.paragraphs)


def inspect(n):
    files = sorted(RAW_DIR.glob("*.docx"))[:n]
    if not files:
        raise SystemExit(f"No .docx files found in {RAW_DIR} -- download the dataset there first.")
    print(f"Opening {len(files)} raw documents for manual inspection ...")
    for f in files:
        subprocess.run(["open", str(f)])
    print(
        "\nCheck each opened document for real names/addresses/signatures/phone numbers.\n"
        "If any real people's information appears, keep data/rental_agreements_raw/ and\n"
        "data/rental_agreements_rendered/ gitignored (already the default -- see .gitignore).\n"
        "If everything is clearly templated/synthetic, you can remove those two entries --\n"
        "but default to leaving them in unless you're confident."
    )


def render_all():
    files = sorted(RAW_DIR.glob("*.docx"))
    if not files:
        raise SystemExit(f"No .docx files found in {RAW_DIR} -- download the dataset there first.")
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    font = _get_font()

    manifest = []
    for f in files:
        doc_id = f.stem
        text = _extract_text(f)
        for page_num, page_img in enumerate(render_pages(text, font)):
            image_path = RENDERED_DIR / f"{doc_id}__p{page_num:03d}.png"
            if not image_path.exists():
                page_img.save(image_path)
            manifest.append({
                "doc_id": doc_id, "page_num": page_num,
                "image_path": str(image_path), "tags": {},
            })
    rag_lib.save_manifest("rental_agreements", manifest)
    print(f"Rendered {len(files)} documents into {len(manifest)} pages -> {RENDERED_DIR}")
    print(f"Manifest saved -> {rag_lib.manifest_path('rental_agreements')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", type=int, metavar="N", help="Open N raw docs for the PII checkpoint")
    group.add_argument("--render", action="store_true", help="Render all raw docs to page images + build manifest")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
    else:
        render_all()
