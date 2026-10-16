"""Fixed held-out HR/NDCG comparison: untrained Base+SID, completed CPT, completed SFT."""
import argparse
import ast
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rl_mps import SIDTrie,pad_prompts
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/Amazon23/Industrial_and_Scientific_downstream'
STAGES={
 'base':ROOT/'models/Qwen2.5-1.5B',
 'cpt':ROOT/'output/Industrial_CPT_pure_sid_resume2750/final_checkpoint',
 'sft':ROOT/'output/Industrial_SFT_response_batch8/final_checkpoint'}

def ranking_metrics(predictions,target,ks=(3,5,10)):
 if len(predictions)!=len(set(predictions)):raise ValueError('Duplicate item predictions')
 rank=next((i for i,p in enumerate(predictions) if p==target),None)
 return {f'{metric}@{k}':(float(rank is not None and rank<k) if metric=='HitRate' else 1/math.log2(rank+2) if rank is not None and rank<k else 0.) for metric in ['HitRate','NDCG'] for k in ks}

def make_rows(stage,tokenizer):
 with (DATA/'test/Industrial_and_Scientific.csv').open(newline='') as f:raw=list(csv.DictReader(f))
 order=list(range(len(raw)));random.Random(42).shuffle(order)
 rows=[]
 for ordinal in order:
  row=raw[ordinal];history=ast.literal_eval(row['history_item_sid'])
  if not history:raise ValueError('Empty test history')
  target=row['item_sid'];history_ids=ast.literal_eval(row['history_item_id'])
  if len(history)!=len(history_ids):raise ValueError('History alignment error')
  if stage in ['base','cpt']:
   prompt=''.join(history)
  else:
   prompt='### User Input: \n'+f"The user has interacted with items {', '.join(history)} in chronological order. Can you predict the next possible item that the user may expect?"+'\n\n### Response:\n'
  ids=tokenizer.encode(prompt,add_special_tokens=False)
  # Match original SFT left truncation: 512 total tokens minus 5 response tokens.
  if stage=='sft':
   from data import Tokenizer
   wrapper=Tokenizer(tokenizer)
   instruction='Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\nCan you predict the next possible item that the user may expect?\n\n'
   ids=wrapper.encode(instruction,bos=True,eos=False)+wrapper.encode(prompt,bos=False,eos=False)
   ids=ids[-507:]
  rows.append(dict(input_ids=ids,target=target,user_id=row['source_user_id'],item_id=row['item_id'],test_row=ordinal))
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=list(STAGES),required=True)
 p.add_argument('--output_dir',type=Path,required=True)
 p.add_argument('--batch_size',type=int,default=4)
 p.add_argument('--limit',type=int,default=0)
 p.add_argument('--stop_after',type=int,default=0,help='Pilot boundary; retain full-set protocol and resume later')
 a=p.parse_args();os.chdir(ROOT);a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=True)
 folder=a.output_dir/a.stage;folder.mkdir(exist_ok=True)
 prior_seconds=json.loads((folder/'status.json').read_text()).get('cumulative_prediction_seconds',0.) if (folder/'status.json').exists() else 0.
 status=dict(status='running',stage='loading_model',model_stage=a.stage,model_path=str(STAGES[a.stage]),
  pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),split='test',ks=[3,5,10],num_beams=10,
  batch_size=a.batch_size,base_initialization_seed=42 if a.stage=='base' else None,training_modified=False)
 def save():
  temp=folder/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(folder/'status.json')
 def event(message):print(json.dumps(message),flush=True)
 save();lock=(ROOT/'output/.local-model-gpu.lock').open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if json.loads((ROOT/'output/Industrial_RL_ranking_mps/status.json').read_text())['status']!='paused':raise ValueError('RL must remain paused for this comparison')
  torch.mps.set_per_process_memory_fraction(min(1.,24*2**30/torch.mps.recommended_max_memory()))
  torch.manual_seed(42);random.seed(42);start=time.perf_counter()
  tokenizer=AutoTokenizer.from_pretrained(DATA/'tokenizer',local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(STAGES[a.stage],local_files_only=True,dtype=torch.bfloat16 if a.stage=='base' else torch.float32,attn_implementation='sdpa')
  if a.stage=='base':
   original=AutoTokenizer.from_pretrained(STAGES['base'],local_files_only=True)
   model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
   with torch.no_grad():model.get_input_embeddings().weight[len(original):].normal_(mean=0,std=model.config.initializer_range)
   model.tie_weights()
  if model.get_input_embeddings().weight.shape[0]!=len(tokenizer):raise ValueError('Vocabulary mismatch')
  model.to(device='mps',dtype=torch.float32);model.eval();model.config.use_cache=True
  status['model_load_seconds']=time.perf_counter()-start
  indices=json.loads((DATA/'index/Industrial_and_Scientific.index.json').read_text())
  legal={tuple(tokenizer.encode(''.join(sids),add_special_tokens=False)):''.join(sids) for sids in indices.values()}
  if len(legal)!=25847 or any(len(tokens)!=3 for tokens in legal):raise ValueError('SID catalog mismatch')
  suffix=tokenizer.encode('\n',add_special_tokens=False)+[tokenizer.eos_token_id] if a.stage=='sft' else []
  trie=SIDTrie([list(tokens)+suffix for tokens in legal]);completion_length=3+len(suffix)
  rows=make_rows(a.stage,tokenizer)
  if a.limit:rows=rows[:a.limit]
  legal_strings=set(legal.values())
  if any(row['target'] not in legal_strings for row in rows):raise ValueError('Unknown ground truth')
  # Detect checkpoint/data changes before reusing saved predictions.
  model_stat=(STAGES[a.stage]/'model.safetensors').stat()
  signature=dict(stage=a.stage,model_path=str(STAGES[a.stage]),model_bytes=model_stat.st_size,model_mtime=model_stat.st_mtime_ns,
   test_sha256=hashlib.sha256((DATA/'test/Industrial_and_Scientific.csv').read_bytes()).hexdigest(),
   batch_size=a.batch_size,num_beams=10,history_policy='full pure SID for base/cpt; original SFT instruction+prompt with 507 context cutoff' if a.stage=='sft' else 'full pure SID for base/cpt; original RL prompt with 512 cutoff',limit=a.limit,seed=42)
  manifest=folder/'protocol.json'
  if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Existing predictions use a different model/protocol')
  manifest.write_text(json.dumps(signature,indent=2))
  (folder/'test_rows.json').write_text(json.dumps([r['test_row'] for r in rows]))
  @torch.no_grad()
  def predict(batch):
   ids,mask=pad_prompts(batch,tokenizer.eos_token_id);ids=ids.to('mps');mask=mask.to('mps');prefix=ids.shape[1]
   def allowed(_,tokens):
    result=trie.allowed(tokens[prefix:].tolist())
    if not result:raise ValueError('Invalid generated prefix')
    return result
   options=dict(num_beams=10,num_return_sequences=10,do_sample=False,length_penalty=0.,max_new_tokens=completion_length,
    prefix_allowed_tokens_fn=allowed,pad_token_id=tokenizer.eos_token_id,use_cache=True,logits_to_keep=1)
   if a.stage=='sft':options.update(eos_token_id=tokenizer.eos_token_id,min_new_tokens=completion_length-1)
   else:options.update(eos_token_id=None,forced_eos_token_id=None)
   with torch.autocast('mps',dtype=torch.bfloat16):result=model.generate(input_ids=ids,attention_mask=mask,**options)
   tokens=result[:,prefix:].tolist();torch.mps.synchronize()
   predictions=[]
   for i in range(len(batch)):
    candidates=[legal[tuple(t[:3])] for t in tokens[i*10:(i+1)*10]]
    if len(candidates)!=10 or len(set(candidates))!=10:raise ValueError('Missing/duplicate recommendations')
    predictions.append(candidates)
   del result,ids,mask;torch.mps.empty_cache();return predictions
  warm_start=time.perf_counter();predict(rows[:a.batch_size]);status['warmup_seconds']=time.perf_counter()-warm_start
  path=folder/'predictions.jsonl';totals={f'{metric}@{k}':0. for metric in ['HitRate','NDCG'] for k in [3,5,10]}
  completed=0
  if path.exists():
   for line in path.read_text().splitlines():
    record=json.loads(line);expected=rows[completed]
    if record['test_row']!=expected['test_row'] or record['target']!=expected['target']:raise ValueError('Saved prediction order mismatch')
    for key,value in ranking_metrics(record['predictions'],record['target']).items():totals[key]+=value
    completed+=1
  start=time.perf_counter();start_count=completed
  status.update(stage='evaluating',completed=completed,total=len(rows));save()
  with path.open('a') as f:
   metrics={key:value/completed for key,value in totals.items()} if completed else {}
   while completed<len(rows):
    if a.stop_after and completed>=a.stop_after:
     status.update(status='paused',stage='pilot_complete');save();return
    if (a.output_dir/'PAUSE').exists():status.update(status='paused',stage='paused');save();return
    batch=rows[completed:completed+a.batch_size];predictions=predict(batch)
    for row,candidates in zip(batch,predictions):
     record=dict(test_row=row['test_row'],user_id=row['user_id'],item_id=row['item_id'],target=row['target'],predictions=candidates)
     f.write(json.dumps(record)+'\n')
     for key,value in ranking_metrics(candidates,row['target']).items():totals[key]+=value
    f.flush();completed+=len(batch)
    elapsed=time.perf_counter()-start;rate=elapsed/max(1,completed-start_count)
    metrics={key:value/completed for key,value in totals.items()}
    status.update(completed=completed,total=len(rows),elapsed_seconds=elapsed,cumulative_prediction_seconds=prior_seconds+elapsed,seconds_per_sample=rate,
     remaining_seconds=(len(rows)-completed)*rate,current_metrics=metrics,
     mps_driver_gib=torch.mps.driver_allocated_memory()/2**30);save()
    if completed%100==0 or completed==len(rows):event(dict(model=a.stage,completed=completed,total=len(rows),**metrics,seconds_per_sample=rate))
    if completed==200:
     (folder/'pilot_200.json').write_text(json.dumps(dict(samples=200,metrics=metrics,seconds_per_sample=rate,estimated_full_seconds=len(rows)*rate,preliminary=True),indent=2))
  results=dict(model=a.stage,samples=completed,split='test',candidate_catalog_size=len(legal),ks=[3,5,10],metrics=metrics,
   elapsed_seconds=status.get('cumulative_prediction_seconds',prior_seconds),model_load_seconds=status['model_load_seconds'],warmup_seconds=status['warmup_seconds'])
  (folder/'results.json').write_text(json.dumps(results,indent=2))
  status.update(status='complete',stage='complete',finished_at=datetime.now().astimezone().isoformat());save()
 except BaseException as exc:
  status.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise
 finally:lock.close()
if __name__=='__main__':main()
