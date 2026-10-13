"""Embed canonical product descriptions as text; joint mode is opt-in for comparison.

Requires the official Qwen3-VL-Embedding repository on PYTHONPATH. The output
uses the same item ordering and filename convention as amazon_text2emb.py.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


def load_summaries(path, require_v2=True):
    summaries = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("summary_text"):
                if require_v2 and record.get("summary_version") != "product-attributes-v2":
                    raise ValueError("Old summary format: regenerate image-derived v2 summaries")
                if record["summary_text"] == "unknown product":
                    raise ValueError("Cannot embed an empty product description")
                summaries[str(record["item_id"])] = record["summary_text"]
    return summaries


def load_image(source, timeout):
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=timeout)
        response.raise_for_status()
        raw = io.BytesIO(response.content)
    else:
        raw = source
    with Image.open(raw) as image:
        return image.convert("RGB")


def build_inputs(items, summaries, max_images, image_timeout, require_images, embedding_mode="text"):
    """Yield item ID and official Qwen3VLEmbedder input dictionaries."""
    for item_id, item in sorted(items.items(), key=lambda pair: int(pair[0])):
        key = str(item_id)
        if key not in summaries:
            raise ValueError(f"Missing structured summary for item {key}")
        if embedding_mode == "text":
            yield int(item_id), {"text": summaries[key]}
            continue
        images = []
        for source in item.get("images", [])[:max_images]:
            try:
                images.append(load_image(str(source), image_timeout))
            except (OSError, requests.RequestException) as exc:
                if require_images:
                    raise RuntimeError(f"Cannot load image for item {key}: {source}") from exc
        if require_images and not images:
            raise ValueError(f"No usable image for item {key}")
        entry = {"text": summaries[key]}
        if images:
            entry["image"] = images if len(images) > 1 else images[0]
        yield int(item_id), entry


def gguf_embed(server_url, entries, media_marker):
    """Call llama.cpp /embedding with plain text or explicit image content."""
    vectors = []
    for entry in entries:
        images = entry.get("image", [])
        if not isinstance(images, list):
            images = [images]
        encoded = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        content = {
            "prompt_string": entry["text"] + ("\n" + " ".join([media_marker] * len(images)) if images else ""),
            "multimodal_data": encoded,
        } if images else entry["text"]
        response = requests.post(
            f"{server_url.rstrip('/')}/embedding",
            json={"content": content}, timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        vector = result[0]["embedding"] if isinstance(result, list) else result["embedding"]
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("llama.cpp returned an invalid embedding")
        vectors.append(vector)
    return np.stack(vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_mode", choices=("text", "joint"), default="text")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--plm_checkpoint", required=True)
    parser.add_argument("--backend", choices=("hf", "gguf"), default="hf")
    parser.add_argument("--server_url", default="http://127.0.0.1:8080")
    parser.add_argument("--mmproj", help="Vision projector GGUF required for image embeddings")
    parser.add_argument("--plm_name", default="qwen-attributes-text")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=1)
    parser.add_argument("--image_timeout", type=float, default=10.0)
    parser.add_argument("--allow_missing_images", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_images < 1:
        parser.error("--batch_size and --max_images must be positive")

    if args.backend == "gguf":
        if not Path(args.plm_checkpoint).is_file():
            parser.error(f"GGUF model file not found: {args.plm_checkpoint}")
        if args.embedding_mode == "joint" and (not args.mmproj or not Path(args.mmproj).is_file()):
            parser.error("GGUF image embedding requires an existing --mmproj vision projector file")
        try:
            models = requests.get(f"{args.server_url.rstrip('/')}/models", timeout=5).json()
        except requests.RequestException as exc:
            parser.error(f"llama.cpp server is unavailable: {exc}")
        if args.embedding_mode == "joint" and not any("multimodal" in model.get("capabilities", []) for model in models.get("models", [])):
            parser.error("llama.cpp server does not report multimodal capability; check --mmproj")
        props = requests.get(f"{args.server_url.rstrip('/')}/props", timeout=5).json()
        media_marker = props.get("media_marker")
        if args.embedding_mode == "joint" and (not props.get("modalities", {}).get("vision") or not media_marker):
            parser.error("llama.cpp server does not report a loaded vision projector")
    else:
        try:
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError as exc:
            raise SystemExit(
                "Install the official Qwen3-VL-Embedding repository and put its root on PYTHONPATH"
            ) from exc

    item_path = Path(args.root) / f"{args.dataset}.item.json"
    with item_path.open(encoding="utf-8") as stream:
        items = json.load(stream)
    summaries = load_summaries(args.summary_path, args.embedding_mode == "text")
    if args.limit > 0:
        items = dict(list(sorted(items.items(), key=lambda pair: int(pair[0])))[:args.limit])

    if args.backend == "hf":
        import torch
        model = Qwen3VLEmbedder(
            model_name_or_path=args.plm_checkpoint,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    vectors = []
    batch = []
    for _, entry in tqdm(
        build_inputs(items, summaries, args.max_images, args.image_timeout,
                     not args.allow_missing_images, args.embedding_mode),
        total=len(items), desc="Embedding products"
    ):
        batch.append(entry)
        if len(batch) == args.batch_size:
            vectors.append(gguf_embed(args.server_url, batch, media_marker) if args.backend == "gguf"
                           else model.process(batch).cpu().numpy())
            batch = []
    if batch:
        vectors.append(gguf_embed(args.server_url, batch, media_marker) if args.backend == "gguf"
                       else model.process(batch).cpu().numpy())
    if not vectors:
        raise ValueError("No items to embed")

    embeddings = np.concatenate(vectors, axis=0)
    output = Path(args.root) / f"{args.dataset}.emb-{args.plm_name}-td.npy"
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new --plm_name")
    np.save(output, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {output}")


if __name__ == "__main__":
    main()
