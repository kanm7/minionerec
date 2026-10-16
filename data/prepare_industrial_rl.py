"""Prepare original RL task prompts and explicit row targets, without history-key collisions."""
import ast
import json
from pathlib import Path
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import data as loaders
loaders.eval=ast.literal_eval

def main():
 base=ROOT/'data/Amazon23/Industrial_and_Scientific_downstream';out=base/'rl';out.mkdir(exist_ok=True)
 tokenizer=AutoTokenizer.from_pretrained(base/'tokenizer',local_files_only=True)
 indices=json.loads((base/'index/Industrial_and_Scientific.index.json').read_text());legal={''.join(v) for v in indices.values()}
 meta=dict(item_file=str(base/'index/Industrial_and_Scientific.item.json'),index_file=str(base/'index/Industrial_and_Scientific.index.json'))
 tasks=[('sid_recommendation',loaders.SidDataset,dict(train_file=str(base/'train/Industrial_and_Scientific.csv'))),
 ('item_identification',loaders.RLTitle2SidDataset,meta),
 ('title_history_recommendation',loaders.RLSeqTitle2SidDataset,dict(train_file=str(base/'train/Industrial_and_Scientific.csv'),sample=10000)),
 ('validation',loaders.SidDataset,dict(train_file=str(base/'valid/Industrial_and_Scientific.csv')))]
 manifest=dict(tasks={},reward_type='ranking',prompt_cutoff=512,seed=42,test_used=False,targets='explicit per-row completion; never history2target overwritten mappings')
 for name,cls,kwargs in tasks:
  path=out/f'{name}.parquet'
  if not path.exists():
   dataset=cls(**kwargs,seed=42,category='industrial and scientific items');rows=[];truncated=0
   for row in dataset.inputs:
    target=row['completion'].strip('\n\" ')
    if target not in legal:raise ValueError('Unknown target SID')
    ids=tokenizer.encode(row['prompt'],add_special_tokens=False)
    truncated+=len(ids)>512
    rows.append(dict(input_ids=ids[-512:],target=target,completion=row['completion'],task=name))
   temp=path.with_suffix('.saving');pq.write_table(pa.Table.from_pylist(rows),temp);temp.replace(path)
   (out/f'{name}.preparation.json').write_text(json.dumps(dict(truncated_prompts=truncated,rows=len(rows))))
   del dataset,rows
  manifest['tasks'][name]=dict(samples=pq.read_metadata(path).num_rows)
  print(name,manifest['tasks'][name],flush=True)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('RL data ready',flush=True)
if __name__=='__main__':main()
