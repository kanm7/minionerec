"""Export the Amazon23 5-core Office_Products catalog with stable numeric IDs."""

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source_catalog", type=Path, required=True)
    parser.add_argument("--output_catalog", type=Path, required=True)
    parser.add_argument("--index_file", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    selected = {row[0] for row in connection.execute("SELECT DISTINCT asin FROM reviews")}
    connection.close()
    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)
    count = with_images = 0
    with args.source_catalog.open(encoding="utf-8") as source, \
            args.output_catalog.open("w", encoding="utf-8") as output, \
            args.index_file.open("w", encoding="utf-8") as index:
        for line in source:
            item = json.loads(line)
            asin = item["parent_asin"]
            if asin not in selected:
                continue
            item["item_id"] = count
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
            index.write(f"{asin}\t{count}\n")
            count += 1
            with_images += bool(item.get("images"))
    if count != len(selected):
        raise ValueError(f"Only {count}/{len(selected)} selected items had metadata")
    print(f"Exported items={count}, with_images={with_images}, without_images={count-with_images}")


if __name__ == "__main__":
    main()
