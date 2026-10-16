"""Cache the original MiniOneRec SFT dataset outputs without changing prompts."""
import ast
import contextlib
import json
from pathlib import Path
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import data as loaders
# Input fields are Python list literals; never execute dataset content.
loaders.eval=ast.literal_eval

def main():
 base=ROOT/'data/Amazon23/Industrial_and_Scientific_downstream'
 out=base/'sft';out.mkdir(exist_ok=True)
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'output/Industrial_CPT_pure_sid_resume2750/final_checkpoint',local_files_only=True)
 tokenizer.pad_token=tokenizer.eos_token
 common=dict(tokenizer=tokenizer,max_len=512,seed=42,category='industrial and scientific items')
 meta=dict(item_file=str(base/'index/Industrial_and_Scientific.item.json'),index_file=str(base/'index/Industrial_and_Scientific.index.json'))
 tasks=[('sid_recommendation',loaders.SidSFTDataset,dict(train_file=str(base/'train/Industrial_and_Scientific.csv'))),
 ('item_identification',loaders.SidItemFeatDataset,meta),
 ('history_to_title',loaders.FusionSeqRecDataset,dict(train_file=str(base/'train/Industrial_and_Scientific.csv'),**meta)),
 ('validation',loaders.SidSFTDataset,dict(train_file=str(base/'valid/Industrial_and_Scientific.csv')))]
 manifest=dict(source_model='output/Industrial_CPT_pure_sid_resume2750/final_checkpoint',cutoff_len=512,
 objective='response-only causal language modeling',original_loaders=True,test_used=False,tasks={})
 for name,cls,kwargs in tasks:
  path=out/f'{name}.parquet'
  if not path.exists():
   print('Preparing',name,flush=True)
   with (out/f'{name}.preparation.log').open('w') as log,contextlib.redirect_stdout(log):
    dataset=cls(**common,**kwargs)
   rows=dataset.inputs
   for row in rows:
    assert row and len(row['input_ids'])==len(row['labels'])<=512
    assert any(t!=-100 for t in row['labels'][1:])
    assert all(label==-100 or label==token for token,label in zip(row['input_ids'],row['labels']))
   table=pa.Table.from_pylist(rows);temp=path.with_suffix('.saving');pq.write_table(table,temp);temp.replace(path)
   del dataset,rows,table
  table=pq.read_table(path,columns=['input_ids','labels'])
  lengths=[len(x) for x in table['input_ids'].to_pylist()]
  manifest['tasks'][name]=dict(samples=len(table),min_length=min(lengths),max_length=max(lengths))
  print(name,manifest['tasks'][name],flush=True)
  del table,lengths
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('SFT data ready',flush=True)
if __name__=='__main__':main()
