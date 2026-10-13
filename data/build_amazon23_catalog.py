"""Stream Amazon Reviews 2023 metadata into a compact parent_asin catalog."""

import argparse
import gzip
import json
from pathlib import Path


def image_urls(images):
    urls = []
    for image in images or []:
        if isinstance(image, dict):
            url = image.get("hi_res") or image.get("large") or image.get("thumb")
            if url:
                urls.append(url)
    return urls


def build_catalog(source, output, index_path, limit=-1):
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = with_images = 0
    opener = gzip.open if str(source).endswith(".gz") else open
    with opener(source, "rt", encoding="utf-8") as input_stream, \
            output.open("w", encoding="utf-8") as catalog, \
            index_path.open("w", encoding="utf-8") as index:
        for line in input_stream:
            meta = json.loads(line)
            asin = meta.get("parent_asin")
            if not asin or asin in seen:
                continue
            seen.add(asin)
            images = image_urls(meta.get("images"))
            item = {
                "item_id": count,
                "parent_asin": asin,
                "title": meta.get("title", ""),
                "description": meta.get("description", []),
                "features": meta.get("features", []),
                "categories": meta.get("categories", []),
                "brand": meta.get("store", ""),
                "details": meta.get("details", {}),
                "images": images,
            }
            catalog.write(json.dumps(item, ensure_ascii=False) + "\n")
            index.write(f"{asin}\t{count}\n")
            count += 1
            with_images += bool(images)
            if limit > 0 and count >= limit:
                break
    print(f"Catalog items={count}, with_images={with_images}, output={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_file", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--index_file", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    build_catalog(args.metadata_file, args.output_file, args.index_file, args.limit)
