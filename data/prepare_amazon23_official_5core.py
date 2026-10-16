"""Preserve official deduplicated 5-core rows and absolute-time split histories."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from .build_amazon23_chronological_subset import usable_product_text
except ImportError:
    from build_amazon23_chronological_subset import usable_product_text


T1, T2 = 1628643414042, 1658002729837


def key(row):
    return row['user_id'], row['parent_asin'], int(row['timestamp']), float(row['rating'])


def read_csv(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def build(raw, output, name):
    rows = read_csv(raw/'rating_only.csv.gz')
    keys = [key(r) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError('Duplicate official interaction rows')
    if len({k[:2] for k in keys}) != len(keys): raise ValueError('Repeated user-product pairs in official deduplicated file')
    users = Counter(r['user_id'] for r in rows)
    products = Counter(r['parent_asin'] for r in rows)
    if min(users.values()) < 5 or min(products.values()) < 5: raise ValueError('Official file fails 5-core check')
    item2id = {asin:i for i,asin in enumerate(sorted(products))}
    user2id = {user:i for i,user in enumerate(sorted(users))}
    split_rows = {}
    split_counts = {}
    for split in ['train','valid','test']:
        records = read_csv(raw/f'{split}.csv.gz')
        split_counts[split] = len(records)
        for row in records:
            k = key(row)
            expected = 'train' if k[2]<T1 else 'valid' if k[2]<T2 else 'test'
            if expected != split or k in split_rows: raise ValueError('Official split boundary or duplicate mismatch')
            split_rows[k] = (split,row['history'].split())
    if set(split_rows) != set(keys): raise ValueError('Official splits do not cover exactly the official full file')
    metadata = {}
    with gzip.open(raw/'meta.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            asin = r.get('parent_asin')
            if asin in item2id and asin not in metadata: metadata[asin] = r
    output.mkdir(parents=True,exist_ok=False)
    def save(file,value): (output/file).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    catalog, anomalies = [], []
    for asin,iid in item2id.items():
        m = metadata.get(asin,{})
        urls = []
        for image in m.get('images',[]) or []:
            for field in ['hi_res','large','thumb']:
                url = image.get(field)
                if url and url not in urls: urls.append(url)
        item = dict(item_id=iid,parent_asin=asin,title=m.get('title',''),description=m.get('description',[]),
                    features=m.get('features',[]),details=m.get('details',{}),categories=m.get('categories',[]),
                    brand=m.get('store',''),images=urls,metadata_status='present' if m else 'missing')
        item['text_usable'] = usable_product_text(item)
        if not item['text_usable']: anomalies.append(dict(item_id=iid,parent_asin=asin,error='no usable product text'))
        catalog.append(item)
    with (output/f'{name}.item.jsonl').open('w') as f:
        for item in catalog: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    save(f'{name}.item.json',{str(r['item_id']):r for r in catalog})
    save('embedding_input_anomalies.json',anomalies)
    for suffix,mapping in [('item2id',item2id),('user2id',user2id)]:
        (output/f'{name}.{suffix}').write_text(''.join(f'{s}\t{i}\n' for s,i in mapping.items()))
    histories = defaultdict(list)
    full = defaultdict(list)
    empty = Counter()
    files = {s:(output/f'{name}.{s}.inter').open('w') for s in split_counts}
    for f in files.values():f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
    try:
        with (output/f'{name}.interactions.jsonl').open('w') as interactions, (output/f'{name}.samples.jsonl').open('w') as samples:
            for line,row in sorted(enumerate(rows,1),key=lambda x:(int(x[1]['timestamp']),x[0])):
                k = key(row); split,official_history = split_rows[k]
                source_user,asin,timestamp,rating = k
                history = histories[source_user]
                # Keep exactly official histories, and verify they contain all
                # preceding user events, never future or missing interactions.
                if history != official_history: raise ValueError(f'History mismatch at source row {line}')
                uid,iid = user2id[source_user],item2id[asin]
                numeric_history = [item2id[a] for a in history]
                event = dict(source_line=line,timestamp_ms=timestamp,user_id=uid,source_user_id=source_user,
                             item_id=iid,parent_asin=asin,rating=rating,split=split)
                interactions.write(json.dumps(event)+'\n')
                samples.write(json.dumps(dict(event,history_item_ids=numeric_history))+'\n')
                files[split].write(f"{uid}\t{' '.join(map(str,numeric_history))}\t{iid}\n")
                empty[split] += not history
                full[uid].append(dict(item_id=iid,timestamp_ms=timestamp,source_line=line))
                history.append(asin)
    finally:
        for f in files.values():f.close()
    with (output/f'{name}.user_histories.jsonl').open('w') as f:
        for uid in sorted(full):f.write(json.dumps(dict(user_id=uid,interactions=full[uid]))+'\n')
    save(f'{name}.inter.json',{str(uid):[r['item_id'] for r in events] for uid,events in full.items()})
    fmt = lambda t:datetime.fromtimestamp(t/1000,ZoneInfo('Asia/Shanghai')).isoformat()
    stats = dict(dataset_name=name,selection='official deduplicated full 5-core; official absolute timestamp splits',
        items=len(products),users=len(users),interactions=len(rows),sequence_samples=len(rows),split_counts=split_counts,
        empty_history_samples=dict(empty),singleton_users=sum(v==1 for v in users.values()),
        min_user_interactions=min(users.values()),min_item_interactions=min(products.values()),
        mean_user_interactions=len(rows)/len(users),mean_item_interactions=len(rows)/len(products),
        metadata_missing=sum(r['metadata_status']=='missing' for r in catalog),without_usable_text=len(anomalies),
        without_image_urls=sum(not r['images'] for r in catalog),with_image_urls=sum(bool(r['images']) for r in catalog),
        start_time=fmt(min(k[2] for k in keys)),cutoff_time=fmt(max(k[2] for k in keys)),
        train_end_timestamp_ms=T1,valid_end_timestamp_ms=T2,official_history_match=True,
        split_policy='train timestamp<T1; valid T1<=timestamp<T2; test timestamp>=T2',
        source='https://amazon-reviews-2023.github.io/data_processing/5core.html')
    save('subset.stats.json',stats)
    (output/'DATASET_REPORT.md').write_text('# Industrial & Scientific 官方去重5-core\n\n'+json.dumps(stats,ensure_ascii=False,indent=2)+
        '\n\n不截取前5000商品、不重新采样或做core过滤。官方全部交互及history逐条核验。测试history允许包含该用户更早发生的测试交互，这是官方逐事件历史协议；不含未来记录。图片或资料缺失不改变交互集合。\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw_dir',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--name',default='Industrial_and_Scientific')
    a=p.parse_args();build(a.raw_dir,a.output_dir,a.name)
