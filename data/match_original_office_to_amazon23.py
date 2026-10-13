"""Audit exact-title links from the bundled Office_Products items to Amazon23."""

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(title):
    return " ".join(html.unescape(str(title or "")).casefold().split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_items", type=Path, required=True)
    parser.add_argument("--amazon23_catalog", type=Path, required=True)
    parser.add_argument("--kcore_catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.original_items.read_text(encoding="utf-8"))
    title_to_ids = defaultdict(list)
    for item_id, item in original.items():
        title_to_ids[normalize(item.get("title"))].append(item_id)
    candidates = defaultdict(list)
    with args.amazon23_catalog.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            title = normalize(item.get("title"))
            if title in title_to_ids:
                candidates[title].append({
                    "parent_asin": item["parent_asin"],
                    "has_image": bool(item.get("images")),
                })
    kcore_asins = set()
    with args.kcore_catalog.open(encoding="utf-8") as stream:
        for line in stream:
            kcore_asins.add(json.loads(line)["parent_asin"])
    records = []
    for item_id, item in sorted(original.items(), key=lambda pair: int(pair[0])):
        matches = candidates.get(normalize(item.get("title")), [])
        records.append({
            "original_item_id": int(item_id),
            "title": item.get("title", ""),
            "exact_title_candidates": [
                {**match, "in_amazon23_kcore": match["parent_asin"] in kcore_asins}
                for match in matches
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    unique = [row for row in records if len(row["exact_title_candidates"]) == 1]
    ambiguous = [row for row in records if len(row["exact_title_candidates"]) > 1]
    missing = [row for row in records if not row["exact_title_candidates"]]
    print(f"original={len(records)} unique_exact={len(unique)} "
          f"ambiguous={len(ambiguous)} unmatched={len(missing)}")
    print(f"unique_exact_with_image={sum(row['exact_title_candidates'][0]['has_image'] for row in unique)} "
          f"unique_exact_in_kcore={sum(row['exact_title_candidates'][0]['in_amazon23_kcore'] for row in unique)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
