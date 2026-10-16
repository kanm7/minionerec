"""Embed product images and structured descriptions (mixed by default).

The summary JSONL comes from generate_structured_summaries.py. Each output
shard keeps item_id, parent_asin and extraction/embedding provenance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

try:
    from .amazon_summary_vl2emb import gguf_embed, load_image
except ImportError:
    from amazon_summary_vl2emb import gguf_embed, load_image


def load_summary_records(path, require_v2=True):
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("summary_text"):
                if require_v2 and record.get("summary_version") != "product-attributes-v2":
                    raise ValueError("Old summary format: regenerate image-derived v2 summaries")
                if record["summary_text"] == "unknown product":
                    raise ValueError("Cannot embed an empty product description")
                records[str(record["item_id"])] = record
    return records


def save_shard(output_dir, ids, asins, vectors, image_used, summary_image_used, canonical_ids, mode):
    target = output_dir / f"items_{ids[0]:07d}_{ids[-1] + 1:07d}.npz"
    temporary = output_dir / f"{target.name}.part"
    with temporary.open("wb") as stream:
        np.savez(stream, item_id=np.asarray(ids, dtype=np.int64),
                 parent_asin=np.asarray(asins),
                 embedding=np.concatenate(vectors),
                 image_used=np.asarray(image_used, dtype=np.bool_),
                 summary_image_used=np.asarray(summary_image_used, dtype=np.bool_),
                 canonical_summary_item_id=np.asarray(canonical_ids, dtype=np.int64),
                 embedding_mode=np.asarray(mode))
    temporary.replace(target)
    print(f"Saved {target} ({len(ids)} items)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--embedding_mode", choices=("text", "joint", "mixed"), default="mixed",
                        help="mixed: joint with readable images, text for explicitly image-less products")
    parser.add_argument("--verified_product_map", type=Path,
                        help='JSON {item_id: canonical_item_id}; externally verified same variant only')
    parser.add_argument("--server_url", default="http://127.0.0.1:8912")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--max_image_side", type=int, default=512)
    parser.add_argument("--image_timeout", type=float, default=20)
    parser.add_argument("--allow_missing_images", action="store_true")
    parser.add_argument("--allow_unembeddable", action="store_true",
                        help="Record catalog items marked text_usable=false as exceptions")
    args = parser.parse_args()
    if not args.model.is_file() or (args.embedding_mode in ("joint", "mixed") and (not args.mmproj or not args.mmproj.is_file())):
        parser.error("The local embedding GGUF and vision projector must exist")
    if args.start < 0 or args.shard_size < 1 or args.max_image_side < 1:
        parser.error("Invalid start, shard size, or image size")

    base = args.server_url.rstrip("/")
    models = requests.get(f"{base}/models", timeout=5).json()
    if args.embedding_mode in ("joint", "mixed") and not any("multimodal" in model.get("capabilities", []) for model in models.get("models", [])):
        parser.error("Embedding server lacks multimodal capability")
    props = requests.get(f"{base}/props", timeout=5).json()
    marker = props.get("media_marker")
    if args.embedding_mode in ("joint", "mixed") and (not marker or not props.get("modalities", {}).get("vision")):
        parser.error("Embedding server has no loaded vision projector")

    summaries = load_summary_records(args.summary_path)
    canonical_map = {}
    if args.verified_product_map:
        if args.embedding_mode != "text":
            parser.error("--verified_product_map is supported only for text embeddings")
        with args.verified_product_map.open(encoding="utf-8") as stream:
            canonical_map = json.load(stream)
        if not isinstance(canonical_map, dict):
            parser.error("Verified map must be an object of item IDs to canonical item IDs")
        canonical_map = {str(k): str(v) for k, v in canonical_map.items()}
        for key, canonical in canonical_map.items():
            if key not in summaries or canonical not in summaries:
                parser.error(f"Verified map references missing summary: {key} -> {canonical}")
            if canonical_map.get(canonical, canonical) != canonical:
                parser.error("Verified map must point directly to canonical items, without chains")
    vector_cache = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("items_*.npz")):
        parser.error("Output directory already contains shards; choose a fresh directory")
    ids, asins, vectors, image_used, summary_images, canonical_ids = [], [], [], [], [], []
    selected = 0
    modes = []
    exceptions = []
    counts = {"joint": 0, "text": 0}
    vector_dim = None
    with args.catalog.open(encoding="utf-8") as catalog:
        for line in tqdm(catalog, desc=f"Embedding products ({args.embedding_mode})"):
            item = json.loads(line)
            item_id = int(item["item_id"])
            if item_id < args.start:
                continue
            if args.limit > 0 and selected >= args.limit:
                break
            selected += 1
            if item.get("text_usable") is False:
                if not args.allow_unembeddable:
                    raise ValueError(f"No usable product text for item {item_id}")
                exceptions.append({"item_id": item_id, "parent_asin": item["parent_asin"],
                                   "error": "no usable product text"})
                (args.output_dir / "exceptions.json").write_text(json.dumps(exceptions, indent=2))
                continue
            summary_record = summaries.get(str(item_id))
            if not summary_record:
                raise ValueError(f"Missing structured summary for item {item_id}")
            if summary_record.get("parent_asin") != item["parent_asin"]:
                raise ValueError(f"Summary parent_asin mismatch for item {item_id}")
            image = None
            if args.embedding_mode in ("joint", "mixed"):
                for source in item.get("images", []):
                    try:
                        image = load_image(source, args.image_timeout)
                        image.thumbnail((args.max_image_side, args.max_image_side))
                        break
                    except (OSError, requests.RequestException):
                        continue
                if args.embedding_mode == "mixed" and image is None:
                    if item.get("images") or item.get("image_status") in ("available", "download_failed"):
                        raise ValueError(f"Unreadable/download-failed image for item {item_id}; refusing silent text fallback")
                elif image is None and not args.allow_missing_images:
                    raise ValueError(f"No usable image for item {item_id}")
            canonical_id = canonical_map.get(str(item_id), str(item_id))
            canonical_record = summaries[canonical_id]
            text = canonical_record["summary_text"]
            model_input = {"text": text}
            if image is not None:
                model_input["image"] = image
            if args.embedding_mode == "text":
                if text not in vector_cache:
                    vector_cache[text] = gguf_embed(base, [model_input], marker)
                vector = vector_cache[text]
            else:
                vector = gguf_embed(base, [model_input], marker)
            if vector_dim is None:
                vector_dim = vector.shape[1]
            if vector.shape != (1, vector_dim) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid/inconsistent vector shape for item {item_id}")
            actual_mode = "joint" if image is not None else "text"
            modes.append(actual_mode)
            counts[actual_mode] += 1
            ids.append(item_id)
            asins.append(item["parent_asin"])
            vectors.append(vector)
            image_used.append(image is not None)
            summary_images.append(canonical_record.get("summary_image_used", False))
            canonical_ids.append(int(canonical_id))
            if len(ids) == args.shard_size:
                save_shard(args.output_dir, ids, asins, vectors, image_used, summary_images, canonical_ids,
                           modes if args.embedding_mode == "mixed" else args.embedding_mode)
                ids, asins, vectors, image_used, summary_images, canonical_ids = [], [], [], [], [], []
                modes = []
    if ids:
        save_shard(args.output_dir, ids, asins, vectors, image_used, summary_images, canonical_ids,
                   modes if args.embedding_mode == "mixed" else args.embedding_mode)
    (args.output_dir / "exceptions.json").write_text(json.dumps(exceptions, indent=2))
    (args.output_dir / "embedding.stats.json").write_text(json.dumps({
        "requested_mode": args.embedding_mode, "selected_items": selected,
        "successful_items": sum(counts.values()), "input_counts": counts,
        "unembeddable_items": len(exceptions), "dimension": vector_dim,
        "model": str(args.model.resolve()), "mmproj": str(args.mmproj.resolve()) if args.mmproj else None,
        "max_image_side": args.max_image_side,
    }, indent=2))
    print(f"Processed {selected} products")


if __name__ == "__main__":
    main()
