"""Count Amazon23 Office_Products items after the project's time and 5-core filters.

Uses an on-disk SQLite table so the 12.8M reviews do not fill RAM.
"""

import argparse
import datetime
import gzip
import json
import sqlite3
from pathlib import Path


def titles_from_metadata(path):
    titles = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("parent_asin") and str(item.get("title") or "").strip():
                titles.add(item["parent_asin"])
    return titles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_file", type=Path, required=True)
    parser.add_argument("--reviews_file", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--start_month", type=int, default=10)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--end_month", type=int, default=9)
    args = parser.parse_args()
    start = int(datetime.datetime(args.start_year, args.start_month, 1).timestamp())
    end = int(datetime.datetime(args.end_year, args.end_month, 1).timestamp())
    titles = titles_from_metadata(args.metadata_file)
    print(f"Metadata items with title: {len(titles)}", flush=True)

    args.database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE IF NOT EXISTS reviews (user_id TEXT, asin TEXT)")
    if connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0:
        batch = []
        seen = accepted = 0
        with gzip.open(args.reviews_file, "rt", encoding="utf-8") as stream:
            for line in stream:
                seen += 1
                row = json.loads(line)
                asin, user = row.get("parent_asin"), row.get("user_id")
                try:
                    timestamp = int(row.get("timestamp", 0)) // 1000
                except (TypeError, ValueError):
                    continue
                if asin in titles and user and start <= timestamp <= end:
                    batch.append((user, asin))
                    accepted += 1
                if len(batch) >= 10000:
                    connection.executemany("INSERT INTO reviews VALUES (?, ?)", batch)
                    connection.commit()
                    batch.clear()
                if seen % 1000000 == 0:
                    print(f"Read reviews={seen}, selected={accepted}", flush=True)
        if batch:
            connection.executemany("INSERT INTO reviews VALUES (?, ?)", batch)
            connection.commit()
        print(f"Selected reviews before k-core: {accepted}", flush=True)
    connection.execute("CREATE INDEX IF NOT EXISTS reviews_user ON reviews(user_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS reviews_asin ON reviews(asin)")
    connection.commit()
    iteration = 0
    while True:
        iteration += 1
        before = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        connection.execute(
            "DELETE FROM reviews WHERE user_id IN "
            "(SELECT user_id FROM reviews GROUP BY user_id HAVING COUNT(*) < ?)",
            (args.min_count,),
        )
        connection.commit()
        connection.execute(
            "DELETE FROM reviews WHERE asin IN "
            "(SELECT asin FROM reviews GROUP BY asin HAVING COUNT(*) < ?)",
            (args.min_count,),
        )
        connection.commit()
        after = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        print(f"K-core iteration {iteration}: {before} -> {after} reviews", flush=True)
        if after == before:
            break
    users = connection.execute("SELECT COUNT(DISTINCT user_id) FROM reviews").fetchone()[0]
    items = connection.execute("SELECT COUNT(DISTINCT asin) FROM reviews").fetchone()[0]
    print(f"RESULT users={users} items={items} reviews={after}", flush=True)
    connection.close()


if __name__ == "__main__":
    main()
