"""Audit SFT task weights and evaluate fixed, matched seen/unseen/validation subsets."""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from cpt_sid_sequence import BucketBatches
from sft_mps import collate_sft,response_loss
DATA=ROOT/'data/Amazon23/Industrial_and_Scientific_downstream/sft'
OUT=ROOT/'output/Industrial_SFT_loss_diagnosis'
MODEL=ROOT/'output/Industrial_SFT_response_batch8/final_checkpoint'

class Length:
 def __init__(self,n):self.n=n
 def __len__(self):return self.n
class Sizes:
 def __init__(self,lengths):self.rows=[Length(n) for n in lengths]
 def __len__(self):return len(self.rows)

def fingerprint(row):
 prompt=[token for token,label in zip(row['input_ids'],row['labels']) if label==-100]
 return hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).digest()

def read_selected(name,indices):
 wanted=set(indices);result=[];offset=0
 for batch in pq.ParquetFile(DATA/f'{name}.parquet').iter_batches(batch_size=4096):
  for local,row in enumerate(batch.to_pylist()):
   if offset+local in wanted:result.append(row)
  offset+=len(batch)
 return result

def main():
 OUT.mkdir(exist_ok=True)
 train_status=json.loads((ROOT/'output/Industrial_SFT_response_batch8/status.json').read_text())
 events=[json.loads(x) for x in (ROOT/'output/Industrial_SFT_response_batch8/metrics.jsonl').read_text().splitlines()]
 current=0;best_step=None;best=float('inf')
 for event in events:
  if 'step' in event:current=event['step']
  if event.get('scope') in ['validation_subset','final_validation_subset'] and event['response_loss']<best:
   best=event['response_loss'];best_step=current
 assert best_step is not None
 tasks=['sid_recommendation','item_identification','history_to_title']
 task_stats={};lengths=[];sid_prompt_targets={};sid_targets=Counter();sid_train_count=0
 for name in tasks:
  count=tokens=max_response=full_length=0;response_lengths=Counter();prompt_lengths=Counter();at_cutoff=0
  for batch in pq.ParquetFile(DATA/f'{name}.parquet').iter_batches(batch_size=4096):
   for row in batch.to_pylist():
    target=[label for label in row['labels'] if label!=-100]
    n=len(target);count+=1;tokens+=n;max_response=max(max_response,n);response_lengths[n]+=1
    prompt_lengths[len(row['input_ids'])-n]+=1
    full_length+=len(row['input_ids']);at_cutoff+=len(row['input_ids'])==512;lengths.append(len(row['input_ids']))
    if name=='sid_recommendation':
     key=fingerprint(row);sid_prompt_targets.setdefault(key,set()).add(tuple(target[:3]));sid_targets[tuple(target[:3])]+=1
  if name=='sid_recommendation':sid_train_count=count
  task_stats[name]=dict(samples=count,response_tokens=tokens,mean_response_tokens=tokens/count,max_response_tokens=max_response,
   mean_sequence_length=full_length/count,at_cutoff=at_cutoff,response_length_histogram=dict(response_lengths))
  print(name,task_stats[name]['samples'],tokens,flush=True)
 total_tokens=sum(v['response_tokens'] for v in task_stats.values())
 for stats in task_stats.values():stats['response_token_share']=stats['response_tokens']/total_tokens
 # Reconstruct the exact training exposure of the selected best checkpoint.
 seen=set();consumed=0
 for i,batch in enumerate(BucketBatches(Sizes(lengths),8,42)):
  if i>=best_step*16:break
  for index in batch:
   consumed+=1
   if index<sid_train_count:seen.add(index)
 unseen=set(range(sid_train_count))-seen
 rng=random.Random(42)
 selections=dict(sid_train_seen=('sid_recommendation',rng.sample(sorted(seen),512)),
  sid_train_not_yet_seen_at_best=('sid_recommendation',rng.sample(sorted(unseen),512)),
  sid_validation=('validation',random.Random(42).sample(range(50985),512)),
  history_to_title_train=('history_to_title',random.Random(42).sample(range(task_stats['history_to_title']['samples']),512)),
  item_identification_train=('item_identification',random.Random(42).sample(range(task_stats['item_identification']['samples']),512)))
 overlap=Counter();valid_response_counts=Counter();valid_frequencies=[];valid_lengths=[]
 for batch in pq.ParquetFile(DATA/'validation.parquet').iter_batches(batch_size=4096):
  for row in batch.to_pylist():
   target=[label for label in row['labels'] if label!=-100];key=fingerprint(row)
   valid_response_counts[len(target)]+=1;valid_frequencies.append(sid_targets[tuple(target[:3])]);valid_lengths.append(len(row['input_ids']))
   if key in sid_prompt_targets:
    overlap['same_prompt']+=1
    if tuple(target[:3]) in sid_prompt_targets[key]:overlap['same_prompt_and_target']+=1
    else:overlap['same_prompt_different_target_only']+=1
   else:overlap['new_prompt']+=1
 result=dict(selected_best_step=best_step,selected_checkpoint_train_examples_consumed=consumed,
  selected_checkpoint_sid_examples_seen=len(seen),selected_checkpoint_sid_examples_not_yet_seen=len(unseen),
  task_stats=task_stats,total_train_response_tokens=total_tokens,validation_prompt_overlap=dict(overlap),
  training_unique_sid_prompts=len(sid_prompt_targets),training_ambiguous_sid_prompts=sum(len(t)>1 for t in sid_prompt_targets.values()),
  validation_response_length_histogram=dict(valid_response_counts),validation_mean_sequence_length=sum(valid_lengths)/len(valid_lengths),
  validation_targets_not_observed_as_training_next_item=sum(f==0 for f in valid_frequencies),
  tests_used=False,rl_training_modified=False,evaluation_subsets={k:dict(task=v[0],indices=v[1]) for k,v in selections.items()})
 (OUT/'data_audit.json').write_text(json.dumps(result,indent=2))
 print('Data audit complete; best checkpoint step',best_step,flush=True)
 lock=(ROOT/'output/.local-model-gpu.lock').open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert json.loads((ROOT/'output/Industrial_RL_ranking_mps/status.json').read_text())['status']=='paused'
  torch.mps.set_per_process_memory_fraction(min(1.,16*2**30/torch.mps.recommended_max_memory()))
  model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('mps')
  model.config.use_cache=False;model.eval()
  tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
  results={}
  with torch.no_grad():
   for label,(task,indices) in selections.items():
    selected=read_selected(task,indices);total_loss=tokens=0;start=time.perf_counter();sid_sum=0.;sid_n=0;format_sum=0.;format_n=0
    # Same response-only CE and same 512-validation selection as original SFT.
    for offset in range(0,len(selected),4):
     batch=collate_sft(selected[offset:offset+4],tokenizer.eos_token_id)
     n=int((batch['labels'][:,1:]!=-100).sum());batch={k:v.to('mps') for k,v in batch.items()}
     with torch.autocast('mps',dtype=torch.bfloat16):loss=response_loss(model,batch)
     total_loss+=loss.item()*n;tokens+=n
     del batch,loss
     if offset%128==0:torch.mps.synchronize();torch.mps.empty_cache()
    results[label]=dict(samples=len(selected),response_tokens=tokens,response_loss=total_loss/tokens,elapsed_seconds=time.perf_counter()-start)
    print(label,results[label],flush=True)
    (OUT/'matched_loss_results.json').write_text(json.dumps(results,indent=2))
  print('Analysis complete; RL remains paused',flush=True)
 finally:lock.close()
if __name__=='__main__':main()
