"""Download one verified product image per item and create a local image catalog."""

import argparse
import concurrent.futures
import io
import json
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


def download(item, image_dir, timeout, attempts):
    item_id = int(item["item_id"])
    target = image_dir / f"{item_id:07d}.jpg"
    if target.exists():
        try:
            with Image.open(target) as image:
                image.verify()
            return item_id, str(target.resolve()), None, None
        except (OSError, ValueError):
            target.unlink()
    errors = []
    for url in item.get("images", []):
        for attempt in range(attempts):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as image:
                    image = image.convert("RGB")
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=92)
                temporary = target.with_suffix(".jpg.part")
                temporary.write_bytes(buffer.getvalue())
                temporary.replace(target)
                return item_id, str(target.resolve()), url, None
            except (requests.RequestException, OSError, ValueError) as exc:
                errors.append(f"{attempt + 1}: {str(exc)[:160]}")
    return item_id, None, None, "; ".join(errors[-3:]) or "no image URLs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output_catalog", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--allow_missing_images", action="store_true",
                        help="Retain products without image URLs; download failures still fail")
    parser.add_argument("--allow_unavailable_images", action="store_true",
                        help="Explicitly retain failed image downloads as text-only inputs, recording errors")
    args = parser.parse_args()
    items = [json.loads(line) for line in args.catalog.open(encoding="utf-8")]
    if args.limit > 0:
        items = items[:args.limit]
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item, args.image_dir, args.timeout, args.attempts)
                   for item in items]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Downloading product images"):
            item_id, path, url, error = future.result()
            results[item_id] = (path, url, error)
    failures = []
    with args.output_catalog.open("w", encoding="utf-8") as output:
        for item in items:
            item_id = int(item["item_id"])
            path, url, error = results[item_id]
            local = dict(item)
            local["image_urls"] = item.get("images", [])
            local["images"] = [path] if path else []
            local["image_status"] = ("available" if path else
                                     ("no_usable_image" if args.allow_unavailable_images else "download_failed")
                                     if item.get("images") else "no_image_urls")
            if error and item.get("images"):
                local["image_error"] = error
            if url:
                local["downloaded_image_url"] = url
            output.write(json.dumps(local, ensure_ascii=False) + "\n")
            if error:
                failures.append({"item_id": item_id, "parent_asin": item["parent_asin"],
                                 "error": error})
    failure_path = args.output_catalog.with_suffix(".failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    fatal = [row for row in failures if not args.allow_unavailable_images and
             (not args.allow_missing_images or results[row["item_id"]][2] != "no image URLs")]
    print(f"Downloaded/verified {len(items) - len(failures)}/{len(items)}; "
          f"catalog={args.output_catalog}; failures={failure_path}")
    if fatal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
