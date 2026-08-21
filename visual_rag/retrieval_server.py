"""
Tiny local HTTP server exposing live ColPali retrieval for ask_rag_ui.py.
Loads the ColQwen2 model once at startup, then loads/caches each dataset's
persisted embeddings (from build_index.py) on first use -- keeps everything
in memory so a query answers in ~1s instead of reloading the model per
request. Deliberately stdlib-only (http.server), no new dependency added to
the intentionally-minimal .venv-colpali environment.

Runs under .venv-colpali (needs colpali_engine/torch), same as retrieve.py:
    .venv-colpali/bin/python visual_rag/retrieval_server.py [--port 8765]

ask_rag_ui.py (main venv) talks to this over localhost -- retrieval and
answer-generation stay split across environments exactly like the batch
retrieve.py / run_visual_rag.py pair, just as long-lived services instead
of one-shot scripts.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_lib  # noqa: E402

_model = None
_processor = None
_manifests = {}
_embeddings = {}


def _load_model():
    global _model, _processor
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading ColQwen2 on {device}...")
    _model = ColQwen2.from_pretrained(
        rag_lib.COLPALI_MODEL_NAME,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    ).eval()
    _processor = ColQwen2Processor.from_pretrained(rag_lib.COLPALI_MODEL_NAME)
    print("Model loaded.")


def _get_dataset(dataset):
    if dataset not in _manifests:
        manifest = rag_lib.load_manifest(dataset)
        if not manifest:
            raise ValueError(f"No page manifest for {dataset!r} -- build it first.")
        embeddings_path = rag_lib.index_dir(dataset) / "embeddings.pt"
        if not embeddings_path.exists():
            raise ValueError(f"No index for {dataset!r} -- run build_index.py first.")
        import torch
        _manifests[dataset] = manifest
        _embeddings[dataset] = torch.load(embeddings_path)
        print(f"Cached index for {dataset!r} ({len(manifest)} pages).")
    return _manifests[dataset], _embeddings[dataset]


def search(dataset, question, top_k=5):
    import torch

    manifest, doc_embeddings = _get_dataset(dataset)
    batch = _processor.process_queries([question]).to(_model.device)
    with torch.no_grad():
        query_embedding = _model(**batch)
    scores = _processor.score_multi_vector(query_embedding.to("cpu"), doc_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    return {
        "top_k_doc_ids": [manifest[i]["doc_id"] for i in top_indices],
        "top_k_image_paths": [manifest[i]["image_path"] for i in top_indices],
        "scores": [scores[i].item() for i in top_indices],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[retrieval_server] {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/search":
            self.send_response(404)
            self.end_headers()
            return
        params = parse_qs(parsed.query)
        try:
            dataset = params["dataset"][0]
            question = params["question"][0]
            top_k = int(params.get("top_k", [5])[0])
            result = search(dataset, question, top_k=top_k)
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    _load_model()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Retrieval server listening on http://127.0.0.1:{args.port} (GET /search?dataset=..&question=..&top_k=5)")
    server.serve_forever()
