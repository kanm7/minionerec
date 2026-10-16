"""Apple MPS GRPO adapter: original ranking rewards, legal SID beam search, fixed reference."""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from ranking_grpo import ranking_rewards,group_advantages,grpo_loss,completion_logps
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/Amazon23/Industrial_and_Scientific_downstream'

class SIDTrie:
 def __init__(self,sequences):
  self.children={}
  for tokens in sequences:
   for i,token in enumerate(tokens):self.children.setdefault(tuple(tokens[:i]),set()).add(token)
 def allowed(self,prefix):return sorted(self.children.get(tuple(prefix),()))

def pad_prompts(rows,pad):
 size=max(len(row['input_ids']) for row in rows)
 ids=torch.full((len(rows),size),pad,dtype=torch.long);mask=torch.zeros_like(ids)
 for i,row in enumerate(rows):
  n=len(row['input_ids']);ids[i,-n:]=torch.tensor(row['input_ids']);mask[i,-n:]=1
 return ids,mask

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model_path',type=Path,required=True)
 p.add_argument('--output_dir',type=Path,required=True)
 p.add_argument('--reward_type',choices=['ranking'],default='ranking')
 p.add_argument('--micro_batch_size',type=int,default=8,help='Completion sequences per backward microbatch')
 p.add_argument('--rollout_batch_size',type=int,default=1,help='Independent prompts per generation call')
 p.add_argument('--prompts_per_update',type=int,default=8)
 p.add_argument('--num_generations',type=int,default=16)
 p.add_argument('--epochs',type=int,default=1)
 p.add_argument('--learning_rate',type=float,default=1e-6)
 p.add_argument('--beta',type=float,default=.04)
 p.add_argument('--eval_steps',type=int,default=250)
 p.add_argument('--eval_samples',type=int,default=128)
 p.add_argument('--probe',action='store_true')
 p.add_argument('--smoke_steps',type=int,default=0)
 p.add_argument('--resume_from_checkpoint',type=Path)
 a=p.parse_args();os.chdir(ROOT);a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=False)
 status=dict(status='running',stage='loading',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
  configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},test_used=False,
  reward_type='ranking',reference_model='fixed SFT model',optimizer='AdamW FP32 states',memory_limit_gib=44)
 def save():
  path=a.output_dir/'status.tmp';path.write_text(json.dumps(status,indent=2));path.replace(a.output_dir/'status.json')
 def event(record):
  print(json.dumps(record),flush=True)
  with (a.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 def cache():torch.mps.synchronize();torch.mps.empty_cache()
 def memory():return dict(mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
 pause=[False]
 def request_pause(*_):pause[0]=True;print('Pausing after saving the next completed update.',flush=True)
 signal.signal(signal.SIGINT,request_pause);signal.signal(signal.SIGTERM,request_pause)
 save();lock=(ROOT/'output/.local-model-gpu.lock').open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
  if a.prompts_per_update%a.rollout_batch_size:raise ValueError('Rollout batch must divide prompts per update')
  if a.micro_batch_size>a.num_generations*a.rollout_batch_size:raise ValueError('Microbatch exceeds rollout completion count')
  if (a.prompts_per_update*a.num_generations)%a.micro_batch_size:raise ValueError('Microbatch must divide effective completions')
  torch.mps.set_per_process_memory_fraction(min(1.,44*2**30/torch.mps.recommended_max_memory()))
  random.seed(42);torch.manual_seed(42)
  tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True);pad=tokenizer.eos_token_id
  policy=AutoModelForCausalLM.from_pretrained(a.resume_from_checkpoint or a.model_path,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('mps')
  reference=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('mps')
  reference.requires_grad_(False);reference.eval();policy.config.use_cache=False;reference.config.use_cache=False
  policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  indices=json.loads((DATA/'index/Industrial_and_Scientific.index.json').read_text())
  sid_tokens=[tokenizer.encode(''.join(v),add_special_tokens=False) for v in indices.values()]
  if len(tokenizer)!=157619 or any(len(tokens)!=3 for tokens in sid_tokens):raise ValueError('SID tokenizer mismatch')
  legal={tuple(tokens) for tokens in sid_tokens}
  suffix=tokenizer.encode('\n',add_special_tokens=False)+[pad]
  sequences=[tokens+suffix for tokens in sid_tokens]
  completion_length=len(sequences[0]);assert all(len(x)==completion_length for x in sequences)
  trie=SIDTrie(sequences)
  rows=[]
  for task in ['sid_recommendation','item_identification','title_history_recommendation']:
   rows.extend(pq.read_table(DATA/'rl'/f'{task}.parquet').to_pylist())
  valid=pq.read_table(DATA/'rl/validation.parquet').to_pylist()
  valid=[valid[i] for i in random.Random(42).sample(range(len(valid)),min(a.eval_samples,len(valid)))]
  optimizer=torch.optim.AdamW(policy.parameters(),lr=a.learning_rate,weight_decay=.01,foreach=False)
  total=math.ceil(len(rows)/a.prompts_per_update)*a.epochs
  warmup=max(1,int(total*.03))
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:n/warmup if n<warmup else .5*(1+math.cos(math.pi*min(1,(n-warmup)/max(1,total-warmup)))))
  start_epoch=next_position=step=0;best=-1.
  if a.resume_from_checkpoint:
   state=torch.load(a.resume_from_checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
   if state['source_model']!=str(a.model_path.resolve()):raise ValueError('Reference model must remain fixed across resume')
   optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
   start_epoch=state['epoch'];next_position=state['next_position'];step=state['step'];best=state['best_ndcg10']
   torch.set_rng_state(state['torch_rng']);torch.mps.set_rng_state(state['mps_rng']);random.setstate(state['python_rng']);del state
  status.update(train_prompts=len(rows),total_optimizer_steps=total,optimizer_steps=step,
   effective_completion_batch=a.prompts_per_update*a.num_generations,
   gradient_accumulation_steps=a.prompts_per_update*a.num_generations//a.micro_batch_size)
  peak=[0.]
  def measure():
   torch.mps.synchronize();peak[0]=max(peak[0],memory()['mps_driver_gib'])
  @torch.no_grad()
  def rollout(batch,beams):
   policy.eval();ids,mask=pad_prompts(batch,pad);ids=ids.to('mps');mask=mask.to('mps');prefix_length=ids.shape[1]
   def allowed(batch_id,tokens):
    result=trie.allowed(tokens[prefix_length:].tolist())
    if not result:raise ValueError('Invalid constrained generation prefix')
    return result
   with torch.autocast('mps',dtype=torch.bfloat16):
    generated=policy.generate(input_ids=ids,attention_mask=mask,num_beams=beams,num_return_sequences=beams,
     max_new_tokens=completion_length,min_new_tokens=completion_length-1,do_sample=False,
     length_penalty=0.,use_cache=True,pad_token_id=pad,eos_token_id=pad,
     prefix_allowed_tokens_fn=allowed,logits_to_keep=1)
   measure();completions=generated[:,prefix_length:]
   assert completions.shape[1]==completion_length
   if any(tuple(tokens[:3]) not in legal for tokens in completions.tolist()):raise ValueError('Illegal generated SID')
   mask=mask.repeat_interleave(beams,dim=0)
   attention=torch.cat([mask,torch.ones_like(completions)],dim=1)
   texts=tokenizer.batch_decode(completions,skip_special_tokens=True)
   del ids,mask,completions;cache();policy.train()
   return generated,attention,texts
  def train_rollout(batch,normalizer,synthetic_advantages=False):
   ids,mask,texts=rollout(batch,a.num_generations)
   targets=[row['completion'] for row in batch for _ in range(a.num_generations)]
   rewards=torch.tensor(ranking_rewards(texts,targets,a.num_generations),dtype=torch.float32,device='mps')
   advantages=group_advantages(rewards,a.num_generations)
   if synthetic_advantages:advantages=torch.linspace(-1,1,len(texts),device='mps')
   loss_sum=kl_sum=0.
   for offset in range(0,len(texts),a.micro_batch_size):
    chunk_ids=ids[offset:offset+a.micro_batch_size];chunk_mask=mask[offset:offset+a.micro_batch_size]
    with torch.no_grad(),torch.autocast('mps',dtype=torch.bfloat16):ref_logps=completion_logps(reference,chunk_ids,chunk_mask,completion_length)
    with torch.autocast('mps',dtype=torch.bfloat16):logps=completion_logps(policy,chunk_ids,chunk_mask,completion_length)
    loss,kl=grpo_loss(logps,ref_logps,advantages[offset:offset+a.micro_batch_size],a.beta)
    if not torch.isfinite(loss):raise ValueError('Nonfinite GRPO loss')
    measure();(loss*len(chunk_ids)/normalizer).backward();measure()
    loss_sum+=loss.item()*len(chunk_ids);kl_sum+=kl.item()*len(chunk_ids)
    del loss,kl,logps,ref_logps,chunk_ids,chunk_mask;cache()
   hits=sum(any(text.strip('\n\" ')==row['completion'].strip('\n\" ') for text in texts[i*a.num_generations:(i+1)*a.num_generations]) for i,row in enumerate(batch))
   summary=dict(loss_sum=loss_sum,kl_sum=kl_sum,reward_sum=rewards.sum().item(),hit_prompts=hits,completions=len(texts))
   del ids,mask,rewards,advantages;cache();return summary
  @torch.no_grad()
  def evaluate(scope):
   totals={f'{metric}@{k}':0. for metric in ['Recall','NDCG'] for k in [1,3,5,10,20]}
   status.update(stage=scope);save()
   for i,row in enumerate(valid):
    ids,mask,texts=rollout([row],20)
    target=row['completion'].strip('\n\" ')
    found=next((j for j,text in enumerate(texts) if text.strip('\n\" ')==target),None)
    if found is not None:
     for k in [1,3,5,10,20]:
      if found<k:totals[f'Recall@{k}']+=1;totals[f'NDCG@{k}']+=1/math.log2(found+2)
    del ids,mask
    if (i+1)%32==0:event(dict(scope=scope,validated=i+1,total=len(valid)))
   metrics={key:value/len(valid) for key,value in totals.items()}
   event(dict(scope=scope,samples=len(valid),**metrics));status.update(stage='training');save();return metrics
  def checkpoint(epoch,position):
   destination=a.output_dir/'last_checkpoint';status.update(stage='saving_checkpoint');save()
   policy.save_pretrained(destination);tokenizer.save_pretrained(destination)
   torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,
    epoch=epoch,next_position=position,best_ndcg10=best,source_model=str(a.model_path.resolve()),
    torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),python_rng=random.getstate()),destination/'training_state.pt')
   status.update(stage='training',resume_checkpoint=str(destination));save()
  start=time.perf_counter();policy.train()
  if a.probe:
   status.update(stage='memory_probe');save()
   stress=sorted(rows,key=lambda row:len(row['input_ids']),reverse=True)[:a.prompts_per_update]
   for iteration in range(2):
    optimizer.zero_grad(set_to_none=True)
    for position in range(0,len(stress),a.rollout_batch_size):
     train_rollout(stress[position:position+a.rollout_batch_size],len(stress)*a.num_generations,True)
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.3)
    if not torch.isfinite(norm):raise ValueError('Nonfinite probe gradient')
    optimizer.step();measure();optimizer.zero_grad(set_to_none=True);cache()
    event(dict(scope='memory_probe',iteration=iteration+1,peak_driver_gib=peak[0],**memory()))
   status.update(status='complete',stage='probe_complete',peak_driver_gib=peak[0],
    seconds_per_prompt=(time.perf_counter()-start)/(2*len(stress)),elapsed_seconds=time.perf_counter()-start);save();return
  if not a.smoke_steps:
   metrics=evaluate('initial_validation_subset');best=max(best,metrics['NDCG@10'])
   policy.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
  status.update(stage='training');save()
  for epoch in range(start_epoch,a.epochs):
   order=list(range(len(rows)));random.Random(42+epoch).shuffle(order)
   for position in range(next_position if epoch==start_epoch else 0,len(order),a.prompts_per_update):
    batch=[rows[i] for i in order[position:position+a.prompts_per_update]]
    optimizer.zero_grad(set_to_none=True);stats=dict(loss_sum=0.,kl_sum=0.,reward_sum=0.,hit_prompts=0,completions=0)
    for offset in range(0,len(batch),a.rollout_batch_size):
     result=train_rollout(batch[offset:offset+a.rollout_batch_size],len(batch)*a.num_generations)
     for key,value in result.items():stats[key]+=value
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.3)
    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
    optimizer.step();scheduler.step();measure();optimizer.zero_grad(set_to_none=True);cache();step+=1
    record=dict(step=step,total_steps=total,loss=stats['loss_sum']/stats['completions'],kl=stats['kl_sum']/stats['completions'],
     reward=stats['reward_sum']/stats['completions'],hit_prompt_fraction=stats['hit_prompts']/len(batch),
     gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-start,peak_driver_gib=peak[0],**memory())
    event(record);status.update(optimizer_steps=step,last_loss=record['loss'],elapsed_seconds=record['elapsed_seconds']);save()
    if a.smoke_steps and step>=a.smoke_steps:
     metrics=evaluate('smoke_validation');status.update(status='complete',stage='smoke_complete',validation=metrics);save();return
    if pause[0] or (a.output_dir/'PAUSE').exists():
     checkpoint(epoch,position+len(batch));status.update(status='paused',stage='paused');save();return
    if step%a.eval_steps==0:
     metrics=evaluate('validation_subset')
     if metrics['NDCG@10']>best:
      best=metrics['NDCG@10'];policy.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
     checkpoint(epoch,position+len(batch))
  metrics=evaluate('final_validation_subset')
  if metrics['NDCG@10']>best:
   best=metrics['NDCG@10'];policy.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
  checkpoint(a.epochs,0)
  # Hard-link selected weights/tokenizer after training ends; no extra model copy.
  final=a.output_dir/'final_checkpoint';final.mkdir()
  for source in (a.output_dir/'best_checkpoint').iterdir():
   if source.is_file():os.link(source,final/source.name)
  status.update(status='complete',stage='complete',best_validation_ndcg10=best,finished_at=datetime.now().astimezone().isoformat());save()
 except BaseException as exc:
  status.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise
 finally:lock.close()
if __name__=='__main__':main()
