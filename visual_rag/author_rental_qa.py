"""
Interactive CLI for manually authoring retrieval questions on the rental
agreements dataset. Mirrors poc/label_ground_truth.py's pattern (opens each
document for you to actually read, saves after every answer, safe to stop
and resume), but for free-form Q&A instead of fixed-field extraction, since
rental agreements don't have a fixed schema the way PAN/Aadhaar do.

Deliberately a script, not a notebook: opens each document in its native
viewer (Word/Preview) rather than displaying it inline, so a real document
is never embedded in a saved .ipynb file's output.

Usage:
    uv run python visual_rag/author_rental_qa.py

For each of the 43 raw agreements, opens the .docx, then repeatedly prompts
for a question and its answer. Type 'done' as the question to move to the
next document, or 'skip' to skip a document with no questions at all.

Saves to visual_rag/manual_qa_cache/rental_agreements.json after every
answer (gitignored, real PII, never committed). Already-finished documents
(typed 'done'/'skip' on a prior run) are not reopened.

Pass --exp2 to author a SEPARATE question set for Experiment #2 (retrieve-
then-generate) instead of the default Experiment #1 (no-retrieval) set.
Exp #2 needs each question to carry a distinguishing detail from its own
document -- a party's name, the execution date, the address, a specific
figure -- used to identify that document while asking about a different
fact in it (the same anchoring idea as PAN/Aadhaar's "whose PAN number is
X" questions). A generic question ("What is the rent?") gives ColPali
nothing to retrieve on. E.g.:
  "What is the notice period in the agreement executed on Jan 10, 2011?"
  "What is the security deposit in the agreement between landlord
   Namashivayam and the tenant?"
Writes to rental_agreements_exp2.json, independent of the Exp #1 set --
existing Exp #1 questions are untouched either way.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

RAW_DIR = rag_lib.DATASETS["rental_agreements"]["raw_dir"]


def author(exp2=False):
    suffix = "_exp2" if exp2 else ""
    qa_path = rag_lib.MANUAL_QA_DIR / f"rental_agreements{suffix}.json"
    finished_path = rag_lib.MANUAL_QA_DIR / f"rental_agreements{suffix}_finished_docs.json"

    def load_questions():
        return json.loads(qa_path.read_text()) if qa_path.exists() else []

    def save_questions(questions):
        rag_lib.MANUAL_QA_DIR.mkdir(parents=True, exist_ok=True)
        qa_path.write_text(json.dumps(questions, indent=2))

    def load_finished():
        return set(json.loads(finished_path.read_text())) if finished_path.exists() else set()

    def save_finished(finished):
        finished_path.write_text(json.dumps(sorted(finished), indent=2))

    if exp2:
        print(
            "=== Experiment #2 mode: anchor each question on a distinguishing detail\n"
            "    from THIS document (a name/date/address/figure) while asking about a\n"
            "    different fact -- a generic question gives retrieval nothing to match on.\n"
            "    e.g. \"What is the security deposit in the agreement executed on <date>?\"\n"
        )

    files = sorted(RAW_DIR.glob("*.docx"))
    if not files:
        raise SystemExit(f"No .docx files found in {RAW_DIR}.")

    questions = load_questions()
    finished = load_finished()
    todo = [f for f in files if f.stem not in finished]

    print(f"{len(finished)}/{len(files)} documents already finished, {len(todo)} to go.\n")

    for i, doc_path in enumerate(todo, 1):
        doc_id = doc_path.stem
        existing_for_doc = [q for q in questions if q["doc_id"] == doc_id]
        subprocess.run(["open", str(doc_path)])
        print(f"\n[{i}/{len(todo)}] {doc_path.name} (opened) "
              f"-- {len(existing_for_doc)} question(s) so far for this doc")

        n = len(existing_for_doc)
        try:
            while True:
                q = input("  Question (or 'done' / 'skip'): ").strip()
                if q.lower() in ("done", "skip"):
                    break
                a = input("  Answer: ").strip()
                if not a:
                    print("  Empty answer, not saved -- try again or type 'done'.")
                    continue
                questions.append({
                    "question_id": f"rental_agreements__{doc_id}__{n}",
                    "question": q,
                    "doc_id": doc_id,
                    "reference_answer": a,
                })
                save_questions(questions)
                n += 1
        except (KeyboardInterrupt, EOFError):
            print("\nStopping early -- progress so far is saved.")
            break

        finished.add(doc_id)
        save_finished(finished)

    print(f"\n{len(questions)} total questions saved -> {qa_path}")
    print(f"{len(finished)}/{len(files)} documents finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp2", action="store_true",
        help="Author the Experiment #2 (retrieval-anchored) question set instead of Experiment #1's",
    )
    args = parser.parse_args()
    author(exp2=args.exp2)
