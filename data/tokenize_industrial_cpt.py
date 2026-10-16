"""Extend the specified Base tokenizer and verify/tokenize pure SID corpora."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    root=Path(__file__).resolve().parents[1]
    data=root/'data/Amazon23/Industrial_and_Scientific_downstream'
    tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen2.5-1.5B-tokenizer',local_files_only=True)
    old_size=len(tokenizer)
    sid_tokens=json.loads((data/'index/sid_tokens_to_add.json').read_text())
    added=tokenizer.add_tokens(sid_tokens)
    sid_ids={t:tokenizer.convert_tokens_to_ids(t) for t in sid_tokens}
    if len(set(sid_ids.values()))!=len(sid_ids):raise ValueError('SID token ID aliases')
    for t,i in sid_ids.items():
        if tokenizer.encode(t,add_special_tokens=False)!=[i]:raise ValueError('Non-atomic SID token')
    tokenizer.save_pretrained(data/'tokenizer')
    (data/'index/sid_token_ids.json').write_text(json.dumps(sid_ids,indent=2))
    schema=pa.schema([('input_ids',pa.list_(pa.int32())),('attention_mask',pa.list_(pa.int8())),
        ('labels',pa.list_(pa.int32())),('source_user_id',pa.string()),('target_item_id',pa.string())])
    verified={}
    for split in ['train','valid','test']:
        count=0;batch=[]
        with pq.ParquetWriter(data/f'cpt/{split}.parquet',schema,compression='zstd') as writer:
            def write(rows):
                texts=[r['text'] for r in rows]
                encoded=tokenizer(texts,add_special_tokens=False,truncation=False)['input_ids']
                output=[]
                for r,ids in zip(rows,encoded):
                    expected=[sid_ids[t] for t in r['sid_tokens']]
                    if ids!=expected or len(ids)!=3*r['num_items']:raise ValueError('Concatenated SID tokenization mismatch')
                    output.append(dict(input_ids=ids,labels=list(ids),attention_mask=[1]*len(ids),
                                       source_user_id=r['source_user_id'],target_item_id=r['target_item_id']))
                writer.write_table(pa.Table.from_pylist(output,schema=schema))
            with (data/f'cpt/{split}.jsonl').open() as stream:
                for line in stream:
                    batch.append(json.loads(line));count+=1
                    if len(batch)==4096:write(batch);batch=[]
                if batch:write(batch)
        verified[split]=count
    manifest=json.loads((data/'manifest.json').read_text())
    if verified!={s:v['samples'] for s,v in manifest['splits'].items()}:raise ValueError('Tokenized coverage mismatch')
    manifest.update(tokenizer_base_vocabulary_size=old_size,tokenizer_extended_size=len(tokenizer),
        newly_added_sid_tokens=added,all_sid_tokens_atomic=True,all_cpt_sequences_tokenization_verified=True,
        cpt_tokenization='Complete: Parquet input_ids/attention_mask/labels; no BOS/EOS, padding, truncation or cross-sample packing',
        causal_loss='labels equal input_ids; causal LM performs one-position shift; padding labels must be -100 at collation',
        model_weight_resize='Pending before CPT: load Base weights and resize_token_embeddings(len(extended_tokenizer)); weights not downloaded')
    (data/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    with (data/'REPORT.md').open('a') as f:
        f.write(f'\n已扩展并保存Base tokenizer：{old_size}→{len(tokenizer)}，新增SID token {added}个。逐条验证全部CPT样本的实际模型token数及顺序，Parquet含input_ids/attention_mask/labels。\n')
        f.write('本次语料不添加BOS/EOS，不补齐、不截断；训练时因果模型内部shift标签，批次padding对应label必须为-100。模型权重未下载，输入/输出embedding扩容在加载权重后、CPT前执行。\n')
    print(json.dumps(dict(base_size=old_size,new_size=len(tokenizer),added=added,verified_samples=verified),indent=2))


if __name__=='__main__':main()
