"""Construct pure-SID causal sequences and original MiniOneRec CSV inputs."""
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pyarrow.parquet as pq


def sid_sequence(history, target, index):
    products=[*history,target]
    tokens=[token for asin in products for token in index[asin]]
    if len(tokens)!=3*len(products):raise ValueError('SID depth mismatch')
    return products,tokens


def normalize_description(item):
    source=item.get('description', '')
    item['source_description']=source
    if isinstance(source,list):
        choices=[v.strip() for v in source if isinstance(v,str) and v.strip()]
        item['description']=max(choices,key=len) if choices else item['title']
    elif isinstance(source,str) and source.strip(): item['description']=source
    else: item['description']=item['title']


def build(archive, dataset, sid_dir, out):
    name='Industrial_and_Scientific'
    out.mkdir(parents=True,exist_ok=False)
    index=json.loads((sid_dir/'unique.asin.index.json').read_text())
    asins=json.loads((dataset/'item_ids.json').read_text())
    if set(asins)!=set(index):raise ValueError('Embedding/SID product coverage mismatch')
    if len({tuple(s) for s in index.values()})!=len(index):raise ValueError('Nonunique final SIDs')
    for sid in index.values():
        if len(sid)!=3:raise ValueError('Expected three SID levels')
        for prefix,token in zip('abc',sid):
            match=re.fullmatch('<'+prefix+r'_(\d+)>',token)
            if not match or int(match[1])>=2048:raise ValueError('Invalid SID token')
    ids={a:i for i,a in enumerate(asins)}
    numeric_index={str(ids[a]):index[a] for a in asins}
    metadata=json.loads((dataset/f'{name}.item.json').read_text())
    if set(metadata)!=set(numeric_index):raise ValueError('Incomplete metadata')
    empty_titles=0
    for iid,item in metadata.items():
        if item['parent_asin']!=asins[int(iid)]:raise ValueError('Metadata ID mismatch')
        if not str(item.get('title') or '').strip():
            item['source_title']=item.get('title','');item['title']='Item '+item['parent_asin'];empty_titles+=1
        normalize_description(item)
    index_dir=out/'index';index_dir.mkdir()
    (index_dir/f'{name}.index.json').write_text(json.dumps(numeric_index))
    (index_dir/f'{name}.asin.index.json').write_text(json.dumps(index))
    (index_dir/f'{name}.item.json').write_text(json.dumps(metadata,ensure_ascii=False))
    (index_dir/'item_ids.json').write_text(json.dumps(asins))
    vocabulary=sorted({t for s in index.values() for t in s},key=lambda t:(t[1],int(t[3:-1])))
    (index_dir/'sid_tokens_to_add.json').write_text(json.dumps(vocabulary,indent=2))
    info=out/'info';info.mkdir()
    with (info/f'{name}.txt').open('w') as f:
        for a in asins:
            title=metadata[str(ids[a])]['title'].replace('\t',' ').replace('\n',' ').replace('\r',' ')
            f.write(f"{''.join(index[a])}\t{title}\t{ids[a]}\n")
    stats={};archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        for split in ['train','valid','test']:
            entry=next(n for n in z.namelist() if n.replace('\\','/').endswith(f'/{split}_sequences.parquet') or n==f'{split}_sequences.parquet')
            with z.open(entry) as stream:table=pq.read_table(stream)
            split_dir=out/split;split_dir.mkdir()
            cpt=out/'cpt';cpt.mkdir(exist_ok=True)
            count=total_tokens=0;minimum=10**9;maximum=0;users=set();targets=set();three_item_example=None
            fields=['user_id','history_item_title','item_title','history_item_id','item_id','history_item_sid','item_sid',
                    'source_user_id','parent_asin','rating','timestamp']
            with (split_dir/f'{name}.csv').open('w',newline='') as csv_file,(cpt/f'{split}.jsonl').open('w') as corpus:
                writer=csv.DictWriter(csv_file,fieldnames=fields);writer.writeheader()
                for batch in table.to_batches(max_chunksize=8192):
                    for row in batch.to_pylist():
                        history=row['history_item_ids'];target=row['target_item_id'];user=row['user_id']
                        if len(history)!=row['history_length'] or not history:raise ValueError('Invalid supplied history')
                        products,tokens=sid_sequence(history,target,index)
                        key=(user,target)
                        if key in targets:raise ValueError('Duplicate target sample')
                        targets.add(key);users.add(user)
                        title=lambda a:metadata[str(ids[a])]['title']
                        writer.writerow(dict(user_id=user,source_user_id=user,parent_asin=target,
                            history_item_title=json.dumps([title(a) for a in history],ensure_ascii=False),item_title=title(target),
                            history_item_id=json.dumps([ids[a] for a in history]),item_id=ids[target],
                            history_item_sid=json.dumps([''.join(index[a]) for a in history]),item_sid=''.join(index[target]),
                            rating=row['target_rating'],timestamp=row['target_timestamp']))
                        record=dict(source_user_id=user,history_item_ids=history,target_item_id=target,
                            target_timestamp=row['target_timestamp'],split=split,sid_tokens=tokens,
                            text=''.join(tokens),num_items=len(products),num_sid_tokens=len(tokens))
                        corpus.write(json.dumps(record,ensure_ascii=False)+'\n')
                        if len(products)==3 and three_item_example is None:three_item_example=record
                        count+=1;total_tokens+=len(tokens);minimum=min(minimum,len(tokens));maximum=max(maximum,len(tokens))
            stats[split]=dict(samples=count,users=len(users),sid_token_min=minimum,sid_token_max=maximum,
                              sid_token_mean=total_tokens/count,total_sid_tokens=total_tokens,three_item_example=three_item_example)
    expected=json.loads((dataset/'subset.stats.json').read_text())['split_counts']
    if {s:v['samples'] for s,v in stats.items()}!=expected:raise ValueError('Sample counts changed')
    manifest=dict(base_model='Qwen/Qwen2.5-1.5B',model_variant='Base',items=len(asins),sid_vocabulary_size=len(vocabulary),
        tokens_per_level={prefix:sum(t[1]==prefix for t in vocabulary) for prefix in 'abc'},
        sid_source=str(sid_dir),history_archive_sha256=archive_hash,split_protocol='supplied per-user leave-last-out',
        splits=stats,cpt_objective='pure SID causal next-token prediction; no prompt or prefix ranking loss',
        cpt_sequences='provided history plus current target; all prefix samples preserved; no cross-sample packing',
        cpt_tokenization='SID strings prepared; model token IDs pending vocabulary extension and atomicity verification',
        special_tokens='counts exclude any BOS/EOS added during model preprocessing',
        sft_rl='original MiniOneRec CSV schema and metadata/index/info; same split files for both stages',
        auxiliary_catalog_scope='all 25847 static catalog items; no validation/test behavior targets added to training CSV',
        empty_title_fallbacks=empty_titles,no_samples_dropped=True,no_sequences_truncated=True)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    lines=['# Industrial 下游数据输入','',
        '基础模型：Qwen/Qwen2.5-1.5B Base。数据构造完成，尚未扩展模型词表或开始训练。',
        'CPT：每条样本为历史商品＋当前目标商品，按顺序展开三层SID；纯自回归预测，无自然语言提示、无前缀排名损失。所有提供的前缀样本保留，因此历史片段会重复出现；不跨用户或样本拼接、不截断。',
        'SFT/RL：共用train/valid/test的CSV及index、item metadata、info，供原项目数据类构造各自任务。CSV额外保留源用户、ASIN、评分和时间供核验。',
        'SID已使用明确消重后的唯一映射；原始量化结果及其0.61%碰撞记录仍保留在试验目录。',
        f'商品{len(asins)}，待加入词表的SID token共{len(vocabulary)}个。实际模型token ID仍待CPT前扩词表后验证，不将原tokenizer的拆分结果当作一个SID token。',
        '划分遵循附件逐用户留后协议，验证历史可含训练事件，测试历史可含验证事件；相同历史重叠是协议的一部分，训练样本未追加留后目标。',
        f'静态商品资料覆盖全catalog；{empty_titles}个空标题明确以Item ASIN占位，没有伪造商品属性。',
        'CPT长度为SID token数，不含BOS/EOS；三个商品就是九个SID token。原cpt.py包含排名损失和不同标签处理，正式纯序列训练需使用匹配此语料的训练入口。','',
        '| 划分 | 样本数 | SID token长度范围 |','|---|---:|---:|']
    lines += [f"| {s} | {r['samples']} | {r['sid_token_min']}–{r['sid_token_max']} |" for s,r in stats.items()]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({s:{k:v for k,v in r.items() if k!='three_item_example'} for s,r in stats.items()},indent=2))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    build(Path('/Users/meekeral/Desktop/历史行为.zip'),root/'data/Amazon23/Industrial_and_Scientific_external',
          root/'output/Industrial_and_Scientific_external-rfsq/input-clamp-collision-20261014-154105/finetune',
          root/'data/Amazon23/Industrial_and_Scientific_downstream')
