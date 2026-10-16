"""Original MiniOneRec response-only SFT tasks on Apple MPS, from completed CPT."""
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
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from cpt_sid_sequence import BucketBatches
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/Amazon23/Industrial_and_Scientific_downstream/sft'
SOURCE=ROOT/'output/Industrial_CPT_pure_sid_resume2750/final_checkpoint'

class Examples(Dataset):
 def __init__(self,validation=False,limit=None):
  tasks=['validation'] if validation else ['sid_recommendation','item_identification','history_to_title']
  self.examples=[]
  for task in tasks:self.examples.extend(pq.read_table(DATA/f'{task}.parquet').to_pylist())
  if limit and limit<len(self.examples):self.examples=[self.examples[i] for i in random.Random(42).sample(range(len(self.examples)),limit)]
  self.rows=[x['input_ids'] for x in self.examples]
 def __len__(self):return len(self.examples)
 def __getitem__(self,i):return self.examples[i]

def collate_sft(rows,pad):
 size=max(len(x['input_ids']) for x in rows)
 result={k:torch.full((len(rows),size),-100 if k=='labels' else pad if k=='input_ids' else 0,dtype=torch.long) for k in ['input_ids','attention_mask','labels']}
 for i,row in enumerate(rows):
  for key in result:result[key][i,:len(row[key])]=torch.tensor(row[key])
 return result

def response_loss(model,batch):
 # Hidden position p predicts label p+1. Keep the union of supervised
 # positions, retaining -100 masking for rows with no target at that position.
 targets=batch['labels'][:,1:]
 positions=torch.where((targets!=-100).any(dim=0))[0]
 if positions.numel()==0:raise ValueError('No response tokens in batch')
 targets=targets.index_select(1,positions)
 result=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],logits_to_keep=positions)
 return torch.nn.functional.cross_entropy(result.logits.float().reshape(-1,result.logits.shape[-1]),targets.reshape(-1),ignore_index=-100)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output_dir',type=Path,required=True)
 p.add_argument('--micro_batch_size',type=int,default=8)
 p.add_argument('--batch_size',type=int,default=128)
 p.add_argument('--epochs',type=int,default=1)
 p.add_argument('--learning_rate',type=float,default=5e-5)
 p.add_argument('--eval_steps',type=int,default=500)
 p.add_argument('--eval_samples',type=int,default=512)
 p.add_argument('--probe',action='store_true')
 p.add_argument('--resume_from_checkpoint',type=Path)
 a=p.parse_args();os.chdir(ROOT);a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(exist_ok=False,parents=True)
 status=dict(status='running',stage='loading',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
  configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},source_model=str(SOURCE),
  objective='original MiniOneRec three-task response-only SFT',test_used=False)
 def save_status():
  path=a.output_dir/'status.tmp';path.write_text(json.dumps(status,indent=2));path.replace(a.output_dir/'status.json')
 def event(record):
  print(json.dumps(record),flush=True)
  with (a.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 def cache():torch.mps.synchronize();torch.mps.empty_cache()
 pause=[False]
 def request_pause(*_):pause[0]=True;print('Pause requested; saving after the next complete update.',flush=True)
 signal.signal(signal.SIGINT,request_pause);signal.signal(signal.SIGTERM,request_pause)
 save_status();lock=(ROOT/'output/.local-model-gpu.lock').open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
  # Keep at least ~20 GiB of the 64 GiB machine outside Metal allocation.
  torch.mps.set_per_process_memory_fraction(min(1.,44*2**30/torch.mps.recommended_max_memory()))
  random.seed(42);torch.manual_seed(42)
  tokenizer=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
  model=AutoModelForCausalLM.from_pretrained(a.resume_from_checkpoint or SOURCE,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
  assert model.get_input_embeddings().weight.shape[0]==len(tokenizer)==157619
  model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  model.to('mps');model.train()
  optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01,foreach=False)
  train=Examples();make_batch=lambda rows:collate_sft(rows,tokenizer.pad_token_id)
  status['train_samples']=len(train);status['vocabulary_size']=len(tokenizer)
  def memory():return dict(mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
  if a.probe:
   status['stage']='longest_sequence_memory_probe';save_status()
   longest=sorted(train.examples,key=lambda row:len(row['input_ids']),reverse=True)[:a.micro_batch_size]
   responses=sorted(train.examples,key=lambda row:sum(t!=-100 for t in row['labels']),reverse=True)[:a.micro_batch_size]
   mixed=[longest[i] if i%2==0 else responses[i] for i in range(a.micro_batch_size)]
   peak=0.;start=time.perf_counter()
   for iteration in range(3):
    optimizer.zero_grad(set_to_none=True)
    probe_rows=longest if iteration==0 else mixed
    batch={k:v.to('mps') for k,v in make_batch(probe_rows).items()}
    with torch.autocast('mps',dtype=torch.bfloat16):loss=response_loss(model,batch)
    if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
    torch.mps.synchronize();peak=max(peak,memory()['mps_driver_gib'])
    loss.backward();torch.mps.synchronize();peak=max(peak,memory()['mps_driver_gib'])
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise ValueError('Nonfinite probe gradient')
    del loss
    with torch.autocast('mps',dtype=torch.bfloat16):loss=response_loss(model,batch)
    loss.backward();torch.mps.synchronize();peak=max(peak,memory()['mps_driver_gib'])
    optimizer.step();torch.mps.synchronize();peak=max(peak,memory()['mps_driver_gib'])
    event(dict(scope='memory_probe',iteration=iteration+1,micro_batch_size=a.micro_batch_size,
      lengths=list(map(lambda r:len(r['input_ids']),probe_rows)),response_tokens=[sum(t!=-100 for t in r['labels']) for r in probe_rows],loss=loss.item(),peak_driver_gib=peak,**memory()))
    del batch,loss;optimizer.zero_grad(set_to_none=True);cache()
   status.update(status='complete',stage='probe_complete',peak_driver_gib=peak,elapsed_seconds=time.perf_counter()-start);save_status();return
  if a.batch_size%a.micro_batch_size:raise ValueError('Batch size must be divisible by microbatch size')
  accumulation=a.batch_size//a.micro_batch_size
  updates_per_epoch=math.ceil(math.ceil(len(train)/a.micro_batch_size)/accumulation)
  total_updates=updates_per_epoch*a.epochs
  warmup=20
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:n/max(1,warmup) if n<warmup else max(0.,(total_updates-n)/max(1,total_updates-warmup)))
  start_epoch=step=skip=0;best=float('inf')
  if a.resume_from_checkpoint:
   state=torch.load(a.resume_from_checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
   if state['micro_batch_size']!=a.micro_batch_size:raise ValueError('Resume requires the same microbatch ordering')
   optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
   start_epoch=state['epoch'];step=state['step'];skip=state['next_batch_index'];best=state['best_validation_loss']
   torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
   if 'mps_rng' in state:torch.mps.set_rng_state(state['mps_rng'])
   del state
  valid=Examples(validation=True,limit=a.eval_samples)
  @torch.no_grad()
  def evaluate(dataset,scope):
   model.eval();cache();loss_sum=count=0
   loader=DataLoader(dataset,batch_sampler=BucketBatches(dataset,a.micro_batch_size,42),collate_fn=make_batch)
   for i,batch in enumerate(loader):
    n=int((batch['labels'][:,1:]!=-100).sum())
    batch={k:v.to('mps') for k,v in batch.items()}
    with torch.autocast('mps',dtype=torch.bfloat16):loss=response_loss(model,batch)
    if not torch.isfinite(loss):raise ValueError('Nonfinite validation loss')
    loss_sum+=loss.item()*n;count+=n
    del loss,batch
    if (i+1)%100==0:
     cache();event(dict(scope=scope,validation_batches=i+1,total_batches=len(loader)))
   value=loss_sum/count;event(dict(scope=scope,response_loss=value,samples=len(dataset),response_tokens=count))
   cache();model.train();return value
  def checkpoint(name,epoch,next_index):
   destination=a.output_dir/name;status.update(stage='saving_checkpoint');save_status()
   model.save_pretrained(destination);tokenizer.save_pretrained(destination)
   torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,epoch=epoch,
    next_batch_index=next_index,micro_batch_size=a.micro_batch_size,best_validation_loss=best,
    torch_rng=torch.get_rng_state(),python_rng=random.getstate(),mps_rng=torch.mps.get_rng_state()),destination/'training_state.pt')
   status.update(stage='training',resume_checkpoint=str(destination));save_status()
  initial=evaluate(valid,'initial_validation_subset');best=min(best,initial)
  model.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
  status.update(stage='training',optimizer_steps=step,total_optimizer_steps=total_updates,initial_validation_loss=initial);save_status()
  start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
  for epoch in range(start_epoch,a.epochs):
   loader=DataLoader(train,batch_sampler=BucketBatches(train,a.micro_batch_size,42+epoch),collate_fn=make_batch)
   token_count=loss_sum=micro=0
   for index,batch in enumerate(loader):
    if epoch==start_epoch and index<skip:continue
    n=int((batch['labels'][:,1:]!=-100).sum());batch={k:v.to('mps') for k,v in batch.items()}
    with torch.autocast('mps',dtype=torch.bfloat16):loss=response_loss(model,batch)
    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
    (loss*n).backward();loss_sum+=loss.item()*n;token_count+=n;micro+=1
    del loss,batch
    if micro%accumulation==0 or index==len(loader)-1:
     for parameter in model.parameters():
      if parameter.grad is not None:parameter.grad.div_(token_count)
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
     if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
     optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
     record=dict(epoch=epoch+1,step=step,total_steps=total_updates,train_response_loss=loss_sum/token_count,
      gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-start,**memory())
     cache();event(record);status.update(optimizer_steps=step,last_train_loss=record['train_response_loss'],elapsed_seconds=record['elapsed_seconds']);save_status()
     token_count=loss_sum=0
     if pause[0] or (a.output_dir/'PAUSE').exists():
      checkpoint('last_checkpoint',epoch,index+1)
      status.update(status='paused',stage='paused',saved_optimizer_steps=step);save_status();return
     if step%a.eval_steps==0:
      value=evaluate(valid,'validation_subset')
      if value<best:
       best=value;model.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
      checkpoint('last_checkpoint',epoch,index+1)
  value=evaluate(valid,'final_validation_subset')
  if value<best:
   best=value;model.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
  checkpoint('last_checkpoint',a.epochs,0)
  del optimizer,scheduler,model;cache()
  model=AutoModelForCausalLM.from_pretrained(a.output_dir/'best_checkpoint',local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('mps')
  model.config.use_cache=False
  status.update(stage='full_validation');save_status()
  full=evaluate(Examples(validation=True),'selected_full_validation')
  model.save_pretrained(a.output_dir/'final_checkpoint');tokenizer.save_pretrained(a.output_dir/'final_checkpoint')
  status.update(status='complete',stage='complete',best_validation_subset_loss=best,full_validation_loss=full,finished_at=datetime.now().astimezone().isoformat());save_status()
 except BaseException as exc:
  status.update(status='failed',error=f'{type(exc).__name__}: {exc}');save_status();raise
 finally:lock.close()
if __name__=='__main__':main()
