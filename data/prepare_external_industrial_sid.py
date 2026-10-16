"""Bind external ASIN-ordered embeddings and supplied leave-last-out samples."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pyarrow.parquet as pq


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'data/Amazon23/Industrial_and_Scientific_external'
    out.mkdir(exist_ok=False)
    name='Industrial_and_Scientific'
    source=root/'output/industrial_external_inspection'
    with zipfile.ZipFile('/Users/meekeral/Desktop/embedding.zip') as z:
        matrix_bytes=z.read('embedding/fused_embeddings.npy')
        ids=json.loads(z.read('embedding/item_ids.json'))
    matrix=out/'fused_embeddings.npy';matrix.write_bytes(matrix_bytes)
    values=np.load(matrix,allow_pickle=False)
    assert values.shape==(len(ids),256) and len(set(ids))==len(ids) and np.isfinite(values).all()
    assert ids==json.loads((source/'item_ids.json').read_text())
    assert hashlib.sha256(matrix_bytes).digest()==hashlib.sha256((source/'fused_embeddings.npy').read_bytes()).digest()
    np.save(str(matrix)+'.item_ids.npy',np.arange(len(ids)))
    (out/'item_ids.json').write_text(json.dumps(ids))
    item2id={asin:i for i,asin in enumerate(ids)}
    original=json.loads((root/'data/Amazon23/Industrial_and_Scientific_5core'/f'{name}.item.json').read_text())
    by_asin={r['parent_asin']:r for r in original.values()}
    meta={str(i):dict(by_asin[asin],item_id=i) for asin,i in item2id.items()}
    (out/f'{name}.item.json').write_text(json.dumps(meta,ensure_ascii=False))
    (out/f'{name}.item2id').write_text(''.join(f'{a}\t{i}\n' for a,i in item2id.items()))
    events={}
    with gzip.open(root/'data/Amazon23/raw/Industrial_and_Scientific_5core/rating_only.csv.gz','rt') as f:
        for r in csv.DictReader(f):events[(r['user_id'],r['parent_asin'])]=(int(r['timestamp']),float(r['rating']))
    tests=pq.read_table(source/'test_sequences.parquet').to_pylist()
    user2id={u:i for i,u in enumerate(sorted(r['user_id'] for r in tests))}
    (out/f'{name}.user2id').write_text(''.join(f'{u}\t{i}\n' for u,i in user2id.items()))
    full={r['user_id']:[*r['history_item_ids'],r['target_item_id']] for r in tests}
    positions={u:{a:i for i,a in enumerate(seq)} for u,seq in full.items()}
    sample_counts={};raw_counts=dict(train=0,valid=0,test=0)
    with (out/f'{name}.samples.jsonl').open('w') as dst:
        for split in raw_counts:
            count=0
            with (out/f'{name}.{split}.inter').open('w') as inter:
                inter.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
                for batch in pq.ParquetFile(source/f'{split}_sequences.parquet').iter_batches(batch_size=8192):
                    for r in batch.to_pylist():
                        u,asin=r['user_id'],r['target_item_id'];pos=positions[u][asin];seq=full[u]
                        expected='test' if pos==len(seq)-1 else 'valid' if pos==len(seq)-2 else 'train'
                        assert split==expected and r['history_item_ids']==seq[:pos]
                        assert events[(u,asin)]==(r['target_timestamp'],r['target_rating'])
                        history=[item2id[a] for a in r['history_item_ids']];iid=item2id[asin];uid=user2id[u]
                        dst.write(json.dumps(dict(user_id=uid,source_user_id=u,item_id=iid,parent_asin=asin,
                            history_item_ids=history,timestamp_ms=r['target_timestamp'],rating=r['target_rating'],split=split))+'\n')
                        inter.write(f"{uid}\t{' '.join(map(str,history))}\t{iid}\n");count+=1
            sample_counts[split]=count
    with (out/f'{name}.interactions.jsonl').open('w') as dst:
        for u,seq in full.items():
            for i,asin in enumerate(seq):
                split='test' if i==len(seq)-1 else 'valid' if i==len(seq)-2 else 'train'
                ts,rating=events[(u,asin)];raw_counts[split]+=1
                dst.write(json.dumps(dict(user_id=user2id[u],source_user_id=u,item_id=item2id[asin],parent_asin=asin,
                                          timestamp_ms=ts,rating=rating,split=split))+'\n')
    stats=dict(items=len(ids),users=len(full),interactions=sum(map(len,full.values())),sequence_samples=sum(sample_counts.values()),
               split_counts=sample_counts,interaction_split_counts=raw_counts,embedding_shape=list(values.shape),
               embedding_source='external fused embeddings; model provenance unknown',
               selection='provided leave-last-out; exclude first-event empty-history targets',
               matrix_sha256=hashlib.sha256(matrix_bytes).hexdigest(),
               distinct_exact_vectors=len({hashlib.sha256(r.tobytes()).digest() for r in values}))
    (out/'subset.stats.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
