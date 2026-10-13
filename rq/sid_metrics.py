#!/usr/bin/env python3
"""Report tokenizer-level SID utilization, balance and optional purity metrics."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def normalized_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum(
        (value / total) * math.log(value / total) for value in counts.values()
    )
    return entropy / math.log(len(counts))


def prefix_metrics(rows, depth):
    counts = Counter(tuple(row[:depth]) for row in rows)
    total = len(rows)
    return {
        "depth": depth,
        "used_codes": len(counts),
        "singleton_rate": sum(value == 1 for value in counts.values())
        / max(len(counts), 1),
        "top_1pct_item_share": sum(
            sorted(counts.values(), reverse=True)[
                : max(1, math.ceil(len(counts) * 0.01))
            ]
        )
        / max(total, 1),
        "normalized_entropy": normalized_entropy(counts),
        "max_cluster_size": max(counts.values(), default=0),
    }


def category_purity(indices, item_meta, category_field):
    clusters = defaultdict(Counter)
    labeled = 0
    for item_id, sid in indices.items():
        label = item_meta.get(str(item_id), {}).get(category_field)
        if isinstance(label, list):
            label = label[0] if label else None
        if label:
            clusters[tuple(sid)][str(label)] += 1
            labeled += 1
    correct = sum(max(labels.values()) for labels in clusters.values())
    return correct / labeled if labeled else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", required=True)
    parser.add_argument("--codebook_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--item_file", default="")
    parser.add_argument("--category_field", default="categories")
    parser.add_argument("--output_file", default="")
    args = parser.parse_args()

    indices = json.loads(Path(args.index_file).read_text(encoding="utf-8"))
    rows = list(indices.values())
    if not rows:
        raise ValueError("index file is empty")
    depth = len(rows[0])
    if any(len(row) != depth for row in rows):
        raise ValueError("SID depth is not uniform")

    full_counts = Counter(tuple(row) for row in rows)
    result = {
        "items": len(rows),
        "depth": depth,
        "unique_sid_rate": len(full_counts) / len(rows),
        "collision_rate": (len(rows) - len(full_counts)) / len(rows),
        "prefix": [prefix_metrics(rows, level) for level in range(1, depth + 1)],
        "layers": [],
    }
    for layer in range(depth):
        counts = Counter(row[layer] for row in rows)
        capacity = (
            args.codebook_sizes[layer] if layer < len(args.codebook_sizes) else None
        )
        result["layers"].append(
            {
                "layer": layer + 1,
                "used_codes": len(counts),
                "capacity": capacity,
                "occupancy_rate": len(counts) / capacity if capacity else None,
                "normalized_entropy": normalized_entropy(counts),
                "max_code_item_share": max(counts.values()) / len(rows),
            }
        )
    if args.item_file:
        item_meta = json.loads(Path(args.item_file).read_text(encoding="utf-8"))
        result["full_sid_category_purity"] = category_purity(
            indices, item_meta, args.category_field
        )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_file:
        Path(args.output_file).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
