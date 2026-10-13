"""Verify the downloaded Amazon Reviews 2023 Office_Products gzip files."""

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "Amazon23" / "raw"


def verify(name, metadata=False):
    path = ROOT / name
    rows = 0
    with_images = 0
    with_usable_image = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            rows += 1
            if metadata:
                images = record.get("images", [])
                with_images += bool(images)
                with_usable_image += any(
                    isinstance(image, dict)
                    and any(image.get(key) for key in ("hi_res", "large", "thumb"))
                    for image in images
                )
    print(name, "rows=", rows, "with_images=", with_images,
          "with_usable_image=", with_usable_image,
          "compressed_bytes=", path.stat().st_size, flush=True)


if __name__ == "__main__":
    verify("meta_Office_Products.jsonl.gz", metadata=True)
    verify("Office_Products.jsonl.gz")
