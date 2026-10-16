"""Verify embedding shards and export SID inputs with explicit row/item mapping."""
import argparse
import json
from pathlib import Path

import numpy as np


def export(catalog_path, shard_dir, output_path):
    with Path(catalog_path).open(encoding="utf-8") as stream:
        catalog = {int(i["item_id"]): i for i in map(json.loads, stream)}
    records = {}
    dimension = None
    for path in sorted(Path(shard_dir).glob("items_*.npz")):
        with np.load(path, allow_pickle=False) as d:
            vectors = d["embedding"]
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid vectors: {path}")
            dimension = vectors.shape[1] if dimension is None else dimension
            if vectors.shape[1] != dimension or len(vectors) != len(d["item_id"]):
                raise ValueError(f"Inconsistent shape: {path}")
            modes = d["embedding_mode"]
            modes = np.repeat(modes.item(), len(vectors)) if modes.ndim == 0 else modes
            for pos, iid in enumerate(d["item_id"].tolist()):
                if iid in records or iid not in catalog:
                    raise ValueError(f"Duplicate/unexpected item ID: {iid}")
                item = catalog[iid]
                image_used = bool(d["image_used"][pos])
                if d["parent_asin"][pos] != item["parent_asin"]:
                    raise ValueError(f"ASIN mismatch: {iid}")
                if bool(item.get("images")) != image_used or modes[pos] != ("joint" if image_used else "text"):
                    raise ValueError(f"Incorrect image/input mode: {iid}")
                if bool(d["summary_image_used"][pos]) != image_used:
                    raise ValueError(f"Summary image provenance mismatch: {iid}")
                if int(d["canonical_summary_item_id"][pos]) != iid:
                    raise ValueError(f"Unexpected canonical ID remapping: {iid}")
                if np.linalg.norm(vectors[pos]) <= 0:
                    raise ValueError(f"Zero vector: {iid}")
                records[iid] = (vectors[pos].copy(), image_used, str(modes[pos]))
    exceptions = [dict(item_id=iid, parent_asin=item["parent_asin"], error="no usable product text")
                  for iid, item in catalog.items() if item.get("text_usable") is False]
    expected = set(catalog) - {x["item_id"] for x in exceptions}
    if set(records) != expected:
        raise ValueError(f"Embedding coverage mismatch: missing {sorted(expected-set(records))[:20]}")
    if not records: raise ValueError("No valid product embeddings")
    ids = np.array(sorted(records), dtype=np.int64)
    vectors = np.stack([records[int(i)][0] for i in ids]).astype(np.float32)
    output_path = Path(output_path)
    if output_path.exists(): raise ValueError(f"Output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vectors)
    np.save(str(output_path) + ".item_ids.npy", ids)
    norms = np.linalg.norm(vectors, axis=1)
    stats = {"catalog_items": len(catalog), "embedded_items": len(ids), "shape": list(vectors.shape),
             "joint_items": sum(x[1] for x in records.values()),
             "text_items": sum(not x[1] for x in records.values()), "unembeddable_items": len(exceptions),
             "exceptions": exceptions, "finite": True, "norm_min": float(norms.min()),
             "norm_mean": float(norms.mean()), "norm_max": float(norms.max()),
             "matrix": str(output_path.resolve()), "item_id_mapping": str(output_path.resolve())+".item_ids.npy",
             "row_ids_are_dense_catalog_ids": np.array_equal(ids, np.arange(len(catalog))),
             "input_mode": "mixed", "interactions_filtered_for_embedding_availability": False}
    output_path.with_suffix(".manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--shard_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args(); export(a.catalog,a.shard_dir,a.output)


if __name__ == "__main__": main()
