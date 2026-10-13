"""Check item IDs, sequence references, split sizes, and 5-core user histories."""

import argparse
import collections
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    stats = json.loads((root / "subset.stats.json").read_text(encoding="utf-8"))
    catalog = [json.loads(line) for line in (root / "Office_Products.item.jsonl").open(encoding="utf-8")]
    assert len(catalog) == stats["items"]
    assert all(item["item_id"] == i and item["images"] for i, item in enumerate(catalog))
    mappings = [line.rstrip("\n").split("\t") for line in (root / "Office_Products.item2id").open(encoding="utf-8")]
    assert [(item["parent_asin"], str(i)) for i, item in enumerate(catalog)] == [tuple(row) for row in mappings]
    local_catalog = root / "Office_Products.local.item.jsonl"
    if local_catalog.exists():
        local_items = [json.loads(line) for line in local_catalog.open(encoding="utf-8")]
        assert len(local_items) == len(catalog)
        for original, local in zip(catalog, local_items):
            assert original["item_id"] == local["item_id"]
            assert original["parent_asin"] == local["parent_asin"]
            assert local["image_urls"] == original["images"]
            assert len(local["images"]) == 1
            image_path = Path(local["images"][0])
            assert image_path.is_file()
            with Image.open(image_path) as image:
                image.verify()
    histories = json.loads((root / "Office_Products.inter.json").read_text(encoding="utf-8"))
    assert len(histories) == stats["users"]
    counts = collections.Counter(i for items in histories.values() for i in items)
    assert len(counts) == stats["items"] and min(counts.values()) >= stats["min_count"]
    assert min(map(len, histories.values())) >= stats["min_count"]
    assert sum(map(len, histories.values())) == stats["interactions"]
    for split in ("train", "valid", "test"):
        with (root / f"Office_Products.{split}.inter").open(encoding="utf-8") as stream:
            assert next(stream).rstrip("\n") == "user_id:token\titem_id_list:token_seq\titem_id:token"
            n = 0
            for line in stream:
                user, history, target = line.rstrip("\n").split("\t")
                assert user in histories
                ids = [int(i) for i in history.split()]
                assert 1 <= len(ids) <= 10
                assert all(0 <= i < stats["items"] for i in ids)
                assert 0 <= int(target) < stats["items"]
                n += 1
            assert n == stats[split], (split, n, stats[split])
    print("PASS", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
