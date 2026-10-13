"""Build a compact Amazon Reviews 2023 Office_Products recommendation dataset.

Select popular products with metadata images from the existing full 5-core
database, reapply 5-core after restricting products, and rebuild sequences and
chronological train/valid/test splits from the original review records.
"""

import argparse
import collections
import datetime
import gzip
import json
import sqlite3
from pathlib import Path


def dump_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)


def select_items(connection, image_asins, limit, min_count):
    connection.execute("CREATE TEMP TABLE chosen (asin TEXT PRIMARY KEY)")
    ranked = connection.execute(
        "SELECT asin, COUNT(*) AS n FROM reviews GROUP BY asin ORDER BY n DESC, asin"
    )
    selected = []
    for asin, count in ranked:
        if asin in image_asins:
            selected.append((asin, count))
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise ValueError(f"Only {len(selected)} products with images; requested {limit}")
    connection.executemany("INSERT INTO chosen VALUES (?)", ((asin,) for asin, _ in selected))
    connection.execute(
        "CREATE TEMP TABLE subset AS SELECT r.user_id, r.asin FROM reviews r "
        "JOIN chosen c ON r.asin = c.asin"
    )
    connection.execute("CREATE INDEX subset_user ON subset(user_id)")
    connection.execute("CREATE INDEX subset_asin ON subset(asin)")
    while True:
        before = connection.execute("SELECT COUNT(*) FROM subset").fetchone()[0]
        connection.execute(
            "DELETE FROM subset WHERE user_id IN "
            "(SELECT user_id FROM subset GROUP BY user_id HAVING COUNT(*) < ?)",
            (min_count,),
        )
        connection.execute(
            "DELETE FROM subset WHERE asin IN "
            "(SELECT asin FROM subset GROUP BY asin HAVING COUNT(*) < ?)",
            (min_count,),
        )
        after = connection.execute("SELECT COUNT(*) FROM subset").fetchone()[0]
        print(f"Subset 5-core: {before} -> {after} interactions", flush=True)
        if after == before:
            break
    survivors = {row[0] for row in connection.execute("SELECT DISTINCT asin FROM subset")}
    users = {row[0] for row in connection.execute("SELECT DISTINCT user_id FROM subset")}
    return [(asin, count) for asin, count in selected if asin in survivors], users, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source_catalog", type=Path, required=True)
    parser.add_argument("--reviews_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--items", type=int, default=3000)
    parser.add_argument("--min_count", type=int, default=5)
    args = parser.parse_args()
    if args.items < 1 or args.min_count < 1:
        parser.error("--items and --min_count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "Office_Products"

    metadata = {}
    with args.source_catalog.open(encoding="utf-8") as stream:
        for line in stream:
            product = json.loads(line)
            metadata[product["parent_asin"]] = product
    image_asins = {asin for asin, item in metadata.items() if item.get("images")}
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    selected, users, expected = select_items(connection, image_asins, args.items, args.min_count)
    connection.close()
    asins = [asin for asin, _ in selected]
    item2id = {asin: i for i, asin in enumerate(asins)}
    with (args.output_dir / f"{name}.item.jsonl").open("w", encoding="utf-8") as catalog, \
            (args.output_dir / f"{name}.item2id").open("w", encoding="utf-8") as index:
        for asin, iid in item2id.items():
            item = dict(metadata[asin], item_id=iid)
            catalog.write(json.dumps(item, ensure_ascii=False) + "\n")
            index.write(f"{asin}\t{iid}\n")

    start = int(datetime.datetime(2018, 10, 1).timestamp())
    end = int(datetime.datetime(2023, 9, 1).timestamp())
    by_user = collections.defaultdict(list)
    reviewed = 0
    with gzip.open(args.reviews_file, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            user, asin = row.get("user_id"), row.get("parent_asin")
            if user not in users or asin not in item2id:
                continue
            try:
                timestamp = int(row.get("timestamp", 0)) // 1000
            except (TypeError, ValueError):
                continue
            if not start <= timestamp <= end:
                continue
            by_user[user].append((timestamp, asin, row))
            reviewed += 1
    if reviewed != expected or len(by_user) != len(users):
        raise ValueError(
            f"Review scan differs from 5-core database: rows {reviewed}/{expected}, "
            f"users {len(by_user)}/{len(users)}"
        )

    user2id = {user: i for i, user in enumerate(sorted(users))}
    sequences = []
    user2items = {}
    review_data = {}
    for user in sorted(users):
        rows = sorted(by_user[user], key=lambda entry: entry[0])
        uid = user2id[user]
        ids = [item2id[asin] for _, asin, _ in rows]
        user2items[str(uid)] = ids
        for timestamp, asin, row in rows:
            key = str((uid, item2id[asin], timestamp))
            review_data[key] = {
                "review": row.get("text", ""),
                "summary": row.get("title", ""),
                "helpful_votes": row.get("helpful_votes", 0),
                "verified": bool(row.get("verified_purchase", False)),
            }
        for pos in range(1, len(rows)):
            history = ids[max(0, pos - 10):pos]
            sequences.append((rows[pos][0], uid, history, ids[pos]))
    sequences.sort(key=lambda entry: entry[0])
    total = len(sequences)
    boundaries = (0, int(total * 0.8), int(total * 0.9), total)
    for split, lo, hi in zip(("train", "valid", "test"), boundaries, boundaries[1:]):
        with (args.output_dir / f"{name}.{split}.inter").open("w", encoding="utf-8") as stream:
            stream.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
            for _, uid, history, target in sequences[lo:hi]:
                stream.write(f"{uid}\t{' '.join(map(str, history))}\t{target}\n")

    item_features = {
        str(i): dict(metadata[asin], item_id=i)
        for asin, i in item2id.items()
    }
    dump_json(args.output_dir / f"{name}.inter.json", user2items)
    dump_json(args.output_dir / f"{name}.item.json", item_features)
    dump_json(args.output_dir / f"{name}.review.json", review_data)
    with (args.output_dir / f"{name}.user2id").open("w", encoding="utf-8") as stream:
        for user, uid in user2id.items():
            stream.write(f"{user}\t{uid}\n")
    stats = {
        "selection": "most interactions in original 5-core, image available; ties by parent_asin",
        "requested_items": args.items,
        "items": len(asins),
        "users": len(users),
        "interactions": reviewed,
        "sequence_samples": total,
        "train": boundaries[1],
        "valid": boundaries[2] - boundaries[1],
        "test": total - boundaries[2],
        "min_count": args.min_count,
    }
    dump_json(args.output_dir / "subset.stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
