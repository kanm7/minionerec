"""Download the official Amazon Reviews 2023 Office_Products raw files.

Files are saved under data/Amazon23/raw. Partial downloads can be resumed.
"""

from pathlib import Path

import requests


BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
FILES = {
    "meta_Office_Products.jsonl.gz": "meta_categories",
    "Office_Products.jsonl.gz": "review_categories",
}
DEST = Path(__file__).resolve().parent / "Amazon23" / "raw"


def download(name, category):
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    partial = DEST / f"{name}.part"
    url = f"{BASE}/{category}/{name}"
    size = int(requests.head(url, timeout=30).headers["Content-Length"])
    if target.exists() and target.stat().st_size == size:
        print(f"Already complete: {target} ({size} bytes)", flush=True)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as output:
            downloaded = offset
            next_report = offset + 100_000_000
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"{name}: {downloaded}/{size} bytes", flush=True)
                        next_report += 100_000_000
    if partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download: {partial.stat().st_size}/{size}")
    partial.replace(target)
    print(f"Complete: {target} ({size} bytes)", flush=True)


if __name__ == "__main__":
    for filename, folder in FILES.items():
        download(filename, folder)
