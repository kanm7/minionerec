"""Audit temporal coverage, full histories, splits and image/text provenance."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream: return list(map(json.loads, stream))


def validate(root, report_dir):
    stats = json.loads((root / "subset.stats.json").read_text())
    catalog = {i["item_id"]: i for i in load_jsonl(root / "Office_Products.local.item.jsonl")}
    raw_catalog = {i["item_id"]: i for i in load_jsonl(root / "Office_Products.item.jsonl")}
    interactions = load_jsonl(root / "Office_Products.interactions.jsonl")
    samples = load_jsonl(root / "Office_Products.samples.jsonl")
    assert len(catalog) == stats["items"] and sorted(catalog) == list(range(stats["items"]))
    assert all(catalog[i]["parent_asin"] == raw_catalog[i]["parent_asin"] for i in catalog)
    assert len(interactions) == len(samples) == stats["interactions"]
    assert len({r["source_line"] for r in interactions}) == len(interactions)
    ordering = [(r["timestamp_ms"], r["source_line"]) for r in interactions]
    assert ordering == sorted(ordering)
    assert all(stats["start_timestamp_ms"] <= t <= stats["cutoff_timestamp_ms"] for t,_ in ordering)
    assert len({r["item_id"] for r in interactions}) == len(catalog)
    assert sum(i["first_seen_timestamp_ms"] < stats["cutoff_timestamp_ms"] for i in catalog.values()) < stats["requested_items"]
    counts = Counter()
    empty = Counter()
    histories = defaultdict(list)
    groups = defaultdict(set)
    first_seen = {}
    split_rows = defaultdict(list)
    for record, sample in zip(interactions, samples):
        uid, iid, t = record["user_id"], record["item_id"], record["timestamp_ms"]
        expected_split = "train" if t <= stats["train_end_timestamp_ms"] else "valid" if t <= stats["valid_end_timestamp_ms"] else "test"
        assert record["split"] == sample["split"] == expected_split
        assert record["source_line"] == sample["source_line"] and record["parent_asin"] == catalog[iid]["parent_asin"]
        assert sample["history_item_ids"] == (histories[uid][-stats["max_history"]:] if stats["max_history"] else histories[uid])
        assert record["review"]["timestamp"] == t and record["review"]["parent_asin"] == catalog[iid]["parent_asin"]
        first_seen[iid] = min(t, first_seen.get(iid,t))
        counts[expected_split] += 1
        empty[expected_split] += not sample["history_item_ids"]
        groups[t].add(expected_split)
        split_rows[expected_split].append(f"{uid}\t{' '.join(map(str,sample['history_item_ids']))}\t{iid}")
        histories[uid].append(iid)
    assert all(len(g) == 1 for g in groups.values())
    assert dict(counts) == stats["split_counts"] and dict(empty) == stats["empty_history_samples"]
    assert len(histories) == stats["users"]
    assert {str(uid): ids for uid,ids in histories.items()} == json.loads((root / "Office_Products.inter.json").read_text())
    assert all(first_seen[i] == catalog[i]["first_seen_timestamp_ms"] for i in catalog)
    for split in ("train", "valid", "test"):
        with (root / f"Office_Products.{split}.inter").open() as f:
            next(f); actual=[line.rstrip("\n") for line in f]
        assert actual == split_rows[split]
    image_states = Counter(i.get("image_status") for i in catalog.values())
    assert all(not i["images"] or (len(i["images"]) == 1 and Path(i["images"][0]).is_absolute()
                                  and Path(i["images"][0]).is_file()) for i in catalog.values())
    result = {"validated": True, "items": len(catalog), "users": len(histories), "interactions":len(interactions),
              "splits":dict(counts), "empty_history_samples":dict(empty),
              "image_status_counts":dict(image_states), "expected_joint_items":sum(bool(i["images"]) for i in catalog.values()),
              "expected_text_items":sum(not i["images"] for i in catalog.values()),
              "unusable_text_items":sum(i.get("text_usable") is False for i in catalog.values()),
              "singleton_users_retained":sum(len(v)==1 for v in histories.values()),
              "no_future_interactions":True,"full_histories_preserved":True,"timestamp_ties_not_split":True,
              "cutoff_time":stats["cutoff_time"],"start_time":stats["start_time"]}
    report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir / "dataset.validation.json").write_text(json.dumps(result,indent=2))
    lines=["# 时间截断子集与用户序列报告", "", "数据与用户序列已生成并核验；全量属性和 embedding 尚在运行。", "",
           f"时间：{stats['start_time']} 至 {stats['cutoff_time']}（上海时间，含边界）。",
           f"商品 **{len(catalog)}**，用户 **{len(histories)}**，交互 **{len(interactions)}**。",
           f"训练/验证/测试：**{counts['train']} / {counts['valid']} / {counts['test']}**。", "",
           "按首次出现时间累计约 5000 商品，截止时刻全部记录保留；不按热度抽样，不做双侧 5-core，不追加截止点之后的记录，不用后续商品补齐。", "",
           f"有可用图片 **{result['expected_joint_items']}**；无可用图片 **{result['expected_text_items']}**：无图片 URL {image_states['no_image_urls']}，原始地址全部不可用 {image_states['no_usable_image']}。",
           "有图使用图片＋结构化文字联合 embedding，无图使用同一个 embedding 模型的纯文本向量；最终 NPZ 逐商品记录 embedding_mode 与 image_used。子集包含两种输入方式。",
           f"无可用文字资料商品 **{result['unusable_text_items']}**，缺元数据 **{stats['metadata_missing']}**。", "",
           f"单次交互用户 **{result['singleton_users_retained']}**，全部保留；首条交互空历史样本计数：{dict(empty)}。",
           "完整历史未截短。全局按交互时间划分 80/10/10，边界相同时刻整组留在较早划分；稳定并列顺序是原始文件行号，样本仅使用当前记录之前的历史。", "",
           f"数据目录：{root.resolve()}。",
           "文件：Office_Products.inter.json、Office_Products.user_histories.jsonl、Office_Products.interactions.jsonl、Office_Products.samples.jsonl、Office_Products.train/valid/test.inter。", "",
           "属性模型：Ollama Qwen3-VL-8B-Instruct Q4_K_M；embedding：Qwen3-VL-Embedding-8B Q4_K_M 与匹配的视觉投影。",
           "运行状态见 pipeline.status.json，详细进度见 attributes-attempt-1.log。最终 REPORT.md 仅在全部 embedding 核验通过后自动写入。未运行 SID/CPT/SFT/RL 训练。"]
    (report_dir / "DATASET_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--report_dir",type=Path,required=True)
    a=p.parse_args();validate(a.root,a.report_dir)
