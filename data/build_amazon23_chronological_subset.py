"""Select a timestamp prefix reaching N distinct products, without k-core.

Metadata and image availability never affect the cutoff or retained histories.
All source rows at the cutoff millisecond are retained, including duplicates.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_reviews(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                timestamp = row.get("timestamp")
                if isinstance(timestamp, bool) or timestamp is None:
                    raise ValueError("missing/invalid timestamp")
                timestamp = int(timestamp)
                if not row.get("parent_asin") or not row.get("user_id"):
                    raise ValueError("missing product/user identifier")
                yield line_number, timestamp, row, None
            except (ValueError, TypeError, AttributeError) as exc:
                yield line_number, None, None, str(exc)


def cutoff_timestamp(first_seen, target):
    if len(first_seen) < target:
        raise ValueError(f"Only {len(first_seen)} distinct products after start; requested {target}")
    return sorted(first_seen.values())[target - 1]


def usable_product_text(item):
    # Generic category/brand alone is not sufficient to describe this product.
    def meaningful(value):
        if isinstance(value, dict): return any(meaningful(v) for v in value.values())
        if isinstance(value, list): return any(meaningful(v) for v in value)
        if value is None: return False
        return str(value).strip().casefold() not in ("", "unknown", "unknown product", "n/a", "none", "null")
    return any(meaningful(item.get(key)) for key in ("title", "description", "features", "details"))


def split_cutoffs(rows, train_ratio, valid_ratio):
    n = len(rows)
    train = rows[max(0, math.ceil(n * train_ratio) - 1)][1]
    valid = rows[max(0, math.ceil(n * (train_ratio + valid_ratio)) - 1)][1]
    return train, valid


def build(reviews_file, source_catalog, output_dir, start_ms, target=5000,
          train_ratio=.8, valid_ratio=.1, max_history=0, timezone="Asia/Shanghai"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    name = "Office_Products"
    first_seen = {}
    invalid = Counter()
    scanned = eligible = 0
    with (output_dir / "invalid_source_records.jsonl").open("w") as errors:
        for line, timestamp, row, error in iter_reviews(reviews_file):
            scanned += 1
            if error:
                invalid[error] += 1
                errors.write(json.dumps({"source_line": line, "error": error}) + "\n")
                continue
            if timestamp < start_ms:
                continue
            eligible += 1
            asin = row["parent_asin"]
            first_seen[asin] = min(timestamp, first_seen.get(asin, timestamp))
            if scanned % 1000000 == 0:
                print(f"Scan {scanned}: eligible={eligible}, distinct={len(first_seen)}", flush=True)
    cutoff = cutoff_timestamp(first_seen, target)
    asins = sorted((asin for asin, t in first_seen.items() if t <= cutoff),
                   key=lambda asin: (first_seen[asin], asin))
    item2id = {asin: i for i, asin in enumerate(asins)}
    print(f"Cutoff={cutoff}, products={len(asins)}", flush=True)
    retained = []
    for line, timestamp, row, error in iter_reviews(reviews_file):
        if not error and start_ms <= timestamp <= cutoff:
            retained.append((line, timestamp, row))
    retained.sort(key=lambda entry: (entry[1], entry[0]))
    assert {r[2]["parent_asin"] for r in retained} == set(asins)
    users = sorted({r[2]["user_id"] for r in retained})
    user2id = {user: i for i, user in enumerate(users)}
    metadata = {}
    with Path(source_catalog).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["parent_asin"] in item2id:
                metadata[item["parent_asin"]] = item
    catalog = []
    anomalies = []
    for asin in asins:
        item = dict(metadata.get(asin, {}), item_id=item2id[asin], parent_asin=asin,
                    metadata_status="present" if asin in metadata else "missing",
                    first_seen_timestamp_ms=first_seen[asin])
        item.setdefault("images", [])
        item["text_usable"] = usable_product_text(item)
        catalog.append(item)
        if not item["text_usable"]:
            anomalies.append({"item_id": item["item_id"], "parent_asin": asin,
                              "error": "no usable product text", "metadata_status": item["metadata_status"]})
    with (output_dir / f"{name}.item.jsonl").open("w", encoding="utf-8") as f:
        for item in catalog:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(output_dir / f"{name}.item.json", {str(i["item_id"]): i for i in catalog})
    write_json(output_dir / "embedding_input_anomalies.json", anomalies)
    for suffix, mapping in (("item2id", item2id), ("user2id", user2id)):
        with (output_dir / f"{name}.{suffix}").open("w") as f:
            for source_id, numeric in mapping.items(): f.write(f"{source_id}\t{numeric}\n")
    train_end, valid_end = split_cutoffs(retained, train_ratio, valid_ratio)
    histories = defaultdict(list)
    full_histories = defaultdict(list)
    split_counts = Counter()
    empty_counts = Counter()
    split_streams = {s: (output_dir / f"{name}.{s}.inter").open("w") for s in ("train", "valid", "test")}
    for stream in split_streams.values(): stream.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
    try:
        with (output_dir / f"{name}.interactions.jsonl").open("w", encoding="utf-8") as interactions, \
                (output_dir / f"{name}.samples.jsonl").open("w", encoding="utf-8") as samples:
            for line, timestamp, row in retained:
                uid, iid = user2id[row["user_id"]], item2id[row["parent_asin"]]
                split = "train" if timestamp <= train_end else "valid" if timestamp <= valid_end else "test"
                history = histories[uid][-max_history:] if max_history else histories[uid]
                record = {"source_line": line, "timestamp_ms": timestamp, "user_id": uid,
                          "source_user_id": row["user_id"], "item_id": iid,
                          "parent_asin": row["parent_asin"], "split": split}
                interactions.write(json.dumps(dict(record, review=row), ensure_ascii=False) + "\n")
                samples.write(json.dumps(dict(record, history_item_ids=history), ensure_ascii=False) + "\n")
                split_streams[split].write(f"{uid}\t{' '.join(map(str, history))}\t{iid}\n")
                empty_counts[split] += not history
                split_counts[split] += 1
                histories[uid].append(iid)
                full_histories[uid].append({"item_id": iid, "timestamp_ms": timestamp, "source_line": line})
    finally:
        for stream in split_streams.values(): stream.close()
    write_json(output_dir / f"{name}.inter.json", {str(uid): ids for uid, ids in histories.items()})
    with (output_dir / f"{name}.user_histories.jsonl").open("w") as f:
        for uid in sorted(full_histories):
            f.write(json.dumps({"user_id": uid, "interactions": full_histories[uid]}) + "\n")
    tz = ZoneInfo(timezone)
    iso = lambda t: datetime.fromtimestamp(t / 1000, tz).isoformat()
    stats = {"selection": "chronological prefix reaching distinct-product threshold; no popularity sampling or k-core",
             "start_timestamp_ms": start_ms, "start_time": iso(start_ms), "cutoff_timestamp_ms": cutoff,
             "cutoff_time": iso(cutoff), "timezone": timezone, "requested_items": target,
             "items": len(asins), "users": len(users), "interactions": len(retained),
             "sequence_samples": sum(split_counts.values()), "split_counts": dict(split_counts),
             "empty_history_samples": dict(empty_counts), "singleton_users": sum(len(v) == 1 for v in histories.values()),
             "metadata_missing": len(asins) - len(metadata), "with_image_urls": sum(bool(i["images"]) for i in catalog),
             "without_image_urls": sum(not i["images"] for i in catalog), "without_usable_text": len(anomalies),
             "train_end_timestamp_ms": train_end, "train_end_time": iso(train_end),
             "valid_end_timestamp_ms": valid_end, "valid_end_time": iso(valid_end),
             "requested_split_ratios": [train_ratio, valid_ratio, 1-train_ratio-valid_ratio],
             "split_policy": "all records at a boundary timestamp stay in the earlier split",
             "tie_order": "timestamp_ms then original source line; same-time histories use this stable order",
             "max_history": max_history, "source_rows_scanned": scanned, "eligible_rows_after_start": eligible,
             "invalid_source_counts": dict(invalid), "duplicates_preserved": True,
             "embedding_input_policy": "mixed: joint with a usable image, same model text-only without; no replacement products"}
    write_json(output_dir / "subset.stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews_file", type=Path, required=True)
    parser.add_argument("--source_catalog", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_time", default="2018-10-01T00:00:00")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--items", type=int, default=5000)
    parser.add_argument("--train_ratio", type=float, default=.8)
    parser.add_argument("--valid_ratio", type=float, default=.1)
    parser.add_argument("--max_history", type=int, default=0, help="0 preserves full prior history")
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start_time)
    if start.tzinfo is None: start = start.replace(tzinfo=ZoneInfo(args.timezone))
    if args.items < 1 or args.max_history < 0 or not 0 < args.train_ratio < args.train_ratio+args.valid_ratio < 1:
        parser.error("Invalid item count, history length or split ratios")
    build(args.reviews_file, args.source_catalog, args.output_dir, int(start.timestamp()*1000),
          args.items, args.train_ratio, args.valid_ratio, args.max_history, args.timezone)


if __name__ == "__main__": main()
