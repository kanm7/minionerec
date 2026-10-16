"""Pure SID causal CPT for Qwen Base; single-device Apple GPU support."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent


class Sequences(Dataset):
    def __init__(self,path,limit=None):
        self.rows=pq.read_table(path,columns=['input_ids']).column(0).to_pylist()
        if limit:self.rows=self.rows[:limit]
        if not self.rows or any(len(r)<2 for r in self.rows):raise ValueError('Invalid causal sequences')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]


class BucketBatches(Sampler):
    def __init__(self,data,batch,seed):self.data,self.batch,self.seed=data,batch,seed
    def __len__(self):return math.ceil(len(self.data)/self.batch)
    def __iter__(self):
        rng=random.Random(self.seed);order=list(range(len(self.data)));rng.shuffle(order);batches=[]
        for start in range(0,len(order),512):
            block=sorted(order[start:start+512],key=lambda i:len(self.data.rows[i]))
            batches.extend(block[i:i+self.batch] for i in range(0,len(block),self.batch))
        rng.shuffle(batches)
        return iter(batches)


def collate(rows,pad):
    length=max(map(len,rows));inputs=torch.full((len(rows),length),pad,dtype=torch.long)
    mask=torch.zeros_like(inputs);labels=torch.full_like(inputs,-100)
    for i,row in enumerate(rows):
        values=torch.tensor(row);inputs[i,:len(row)]=values;mask[i,:len(row)]=1;labels[i,:len(row)]=values
    return dict(input_ids=inputs,attention_mask=mask,labels=labels)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--micro_batch_size',type=int,default=4)
    p.add_argument('--gradient_accumulation_steps',type=int,default=16)
    p.add_argument('--learning_rate',type=float,default=5e-5)
    p.add_argument('--eval_steps',type=int,default=500)
    p.add_argument('--eval_samples',type=int,default=2048)
    p.add_argument('--smoke_steps',type=int,default=0)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--resume_from_checkpoint',type=Path)
    p.add_argument('--sampler_batch_size',type=int,default=0)
    p.add_argument('--preflight_longest_batch',action='store_true')
    a=p.parse_args();os.chdir(ROOT);a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=False)
    if a.detach:
        # Directory is created by the child; parent keeps its log next to it.
        a.output_dir.rmdir()
        args=[x for x in sys.argv[1:] if x!='--detach']
        with a.output_dir.with_suffix('.background.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*args],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        a.output_dir.with_suffix('.pid').write_text(str(proc.pid))
        subprocess.Popen(['caffeinate','-i','-w',str(proc.pid)],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        print('CPT background PID:',proc.pid);return
    data=ROOT/'data/Amazon23/Industrial_and_Scientific_downstream'
    status=dict(status='running',stage='waiting_for_gpu',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
                configuration=vars(a).copy(),objective='pure SID causal next-token prediction',test_used=False)
    status['configuration']['output_dir']=str(a.output_dir)
    if a.resume_from_checkpoint:status['configuration']['resume_from_checkpoint']=str(a.resume_from_checkpoint)
    def save():
        temp=a.output_dir/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(a.output_dir/'status.json')
    def event(record):
        with (a.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    save()
    lock=(ROOT/'output/.local-model-gpu.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU unavailable')
        random.seed(42);torch.manual_seed(42);status['stage']='loading_and_extending_vocabulary';save()
        tokenizer=AutoTokenizer.from_pretrained(data/'tokenizer',local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(a.resume_from_checkpoint or ROOT/'models/Qwen2.5-1.5B',local_files_only=True,
                   dtype=torch.float32 if a.resume_from_checkpoint else torch.bfloat16,attn_implementation='sdpa')
        old_tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-1.5B',local_files_only=True)
        sid_ids=json.loads((data/'index/sid_token_ids.json').read_text())
        model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
        # Qwen has spare unregistered rows: initialize all new SID rows,
        # including those within the old model's padded vocabulary capacity.
        with torch.no_grad():
            if not a.resume_from_checkpoint:model.get_input_embeddings().weight[len(old_tokenizer):].normal_(mean=0,std=model.config.initializer_range)
        if not model.config.tie_word_embeddings:raise ValueError('Expected tied Qwen Base embeddings')
        model.tie_weights()
        assert model.get_input_embeddings().weight.shape[0]==model.get_output_embeddings().weight.shape[0]==len(tokenizer)
        assert all(tokenizer.encode(t,add_special_tokens=False)==[i] for t,i in sid_ids.items())
        if not a.resume_from_checkpoint:
            tokenizer.save_pretrained(a.output_dir/'expanded_initial')
            model.save_pretrained(a.output_dir/'expanded_initial')
        status['vocabulary']=dict(size=len(tokenizer),sid_tokens=len(sid_ids),input_output_embeddings_tied=True)
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # FP32 master parameters/Adam states; BF16 autocast for forward compute.
        model.to(device='mps',dtype=torch.float32)
        train=Sequences(data/'cpt/train.parquet',32 if a.smoke_steps else None)
        valid=Sequences(data/'cpt/valid.parquet',a.eval_samples)
        make_batch=lambda rows:collate(rows,tokenizer.pad_token_id or tokenizer.eos_token_id)
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01,foreach=False)
        sampler_batch=a.sampler_batch_size or a.micro_batch_size
        effective_batch=a.micro_batch_size*a.gradient_accumulation_steps
        if sampler_batch%a.micro_batch_size or effective_batch%sampler_batch:raise ValueError('Incompatible sampler/micro/effective batch sizes')
        groups_per_update=effective_batch//sampler_batch
        batches_per_epoch=math.ceil(len(train)/sampler_batch)
        updates=math.ceil(batches_per_epoch/groups_per_update)*a.epochs
        warmup=max(1,int(updates*.03))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:step/warmup if step<warmup else max(0.,(updates-step)/max(1,updates-warmup)))
        resume_step=0;resume_epoch=0;resume_index=0
        if a.resume_from_checkpoint:
            state=torch.load(a.resume_from_checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            resume_step=state['step'];resume_epoch=state['epoch'];resume_index=state.get('next_batch_index',resume_step*groups_per_update)
            torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);del state
            status.update(resumed_from_step=resume_step,total_optimizer_steps=updates);save()
        def clear_cache():
            torch.mps.synchronize();torch.mps.empty_cache()
        def evaluate(dataset,scope):
            clear_cache()
            model.eval();total=weight=0;target_total=target_count=0
            loader=DataLoader(dataset,batch_sampler=BucketBatches(dataset,a.micro_batch_size,42),collate_fn=make_batch)
            with torch.no_grad():
                for batch in loader:
                    n=int((batch['labels'][:,1:]!=-100).sum())
                    batch={k:v.to('mps') for k,v in batch.items()}
                    with torch.autocast('mps',dtype=torch.bfloat16):result=model(**batch)
                    loss=result.loss
                    if not torch.isfinite(loss):raise ValueError('Nonfinite validation loss')
                    total+=loss.item()*n;weight+=n
                    lengths=batch['attention_mask'].sum(1).tolist()
                    logits=torch.stack([result.logits[i,length-4:length-1] for i,length in enumerate(lengths)])
                    targets=torch.stack([batch['input_ids'][i,length-3:length] for i,length in enumerate(lengths)])
                    target_total+=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='sum').item()
                    target_count+=targets.numel()
            value=target_total/target_count;event(dict(scope=scope,next_item_sid_loss=value,
                full_sequence_loss=total/weight,samples=len(dataset),predicted_tokens=weight))
            del result,logits,targets,loss,batch
            clear_cache();model.train();return value
        if a.preflight_longest_batch:
            status['stage']='testing_longest_batch';save()
            longest=sorted(train.rows,key=len,reverse=True)[:a.micro_batch_size]
            model.train();optimizer.zero_grad(set_to_none=True)
            probe=make_batch(longest);probe={k:v.to('mps') for k,v in probe.items()}
            with torch.autocast('mps',dtype=torch.bfloat16):probe_loss=model(**probe).loss
            if not torch.isfinite(probe_loss):raise ValueError('Nonfinite preflight loss')
            probe_loss.backward();torch.mps.synchronize()
            event(dict(scope='longest_batch_preflight',micro_batch_size=a.micro_batch_size,
                sequence_lengths=list(map(len,longest)),loss=probe_loss.item(),
                mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,
                mps_driver_gib=torch.mps.driver_allocated_memory()/2**30))
            del probe_loss,probe;optimizer.zero_grad(set_to_none=True);clear_cache()
        baseline_loss=evaluate(valid,'initial_validation_subset')
        status.update(stage='training',train_samples=len(train),initial_validation_loss=baseline_loss,optimizer_steps=resume_step,total_optimizer_steps=updates);save()
        model.train();optimizer.zero_grad(set_to_none=True);step=resume_step;micro=0;best=baseline_loss;start=time.perf_counter()
        model.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
        for epoch in range(resume_epoch,a.epochs):
            loader=DataLoader(train,batch_sampler=BucketBatches(train,sampler_batch,42+epoch),collate_fn=make_batch)
            accumulated_loss=0;accumulated_tokens=0
            for index,batch in enumerate(loader):
                if epoch==resume_epoch and index<resume_index:continue
                for offset in range(0,len(batch['input_ids']),a.micro_batch_size):
                    chunk={k:v[offset:offset+a.micro_batch_size].to('mps') for k,v in batch.items()}
                    n=int((chunk['labels'][:,1:]!=-100).sum())
                    with torch.autocast('mps',dtype=torch.bfloat16):loss=model(**chunk).loss
                    if not torch.isfinite(loss):raise ValueError('Nonfinite train loss')
                    (loss*n).backward();accumulated_loss+=loss.item()*n;accumulated_tokens+=n
                    del loss,chunk
                micro+=1
                if micro%groups_per_update==0 or index==len(loader)-1:
                    for parameter in model.parameters():
                        if parameter.grad is not None:parameter.grad.div_(accumulated_tokens)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                    record=dict(epoch=epoch+1,step=step,train_loss=accumulated_loss/accumulated_tokens,
                        gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-start)
                    record.update(mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
                    clear_cache()
                    event(record);status.update(optimizer_steps=step,last_train_loss=record['train_loss'],elapsed_seconds=record['elapsed_seconds']);save()
                    accumulated_loss=accumulated_tokens=0
                    if a.smoke_steps and step>=a.smoke_steps:break
                    if step%a.eval_steps==0:
                        value=evaluate(valid,'validation_subset')
                        if value<best:
                            best=value;model.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
                        # Last checkpoint includes optimizer/scheduler and RNG.
                        model.save_pretrained(a.output_dir/'last_checkpoint');tokenizer.save_pretrained(a.output_dir/'last_checkpoint')
                        torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,
                                        epoch=epoch,next_batch_index=index+1,torch_rng=torch.get_rng_state(),python_rng=random.getstate()),a.output_dir/'last_checkpoint/training_state.pt')
            if a.smoke_steps and step>=a.smoke_steps:break
        value=evaluate(valid,'final_validation_subset')
        if value<best:
            best=value;model.save_pretrained(a.output_dir/'best_checkpoint');tokenizer.save_pretrained(a.output_dir/'best_checkpoint')
        if not a.smoke_steps:
            del optimizer,scheduler
            selected=AutoModelForCausalLM.from_pretrained(a.output_dir/'best_checkpoint',local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
            del model;torch.mps.empty_cache();model=selected.to('mps');model.config.use_cache=False
            final_loss=evaluate(Sequences(data/'cpt/valid.parquet'),'selected_full_validation')
            status['full_validation_loss']=final_loss
        model.save_pretrained(a.output_dir/'final_checkpoint');tokenizer.save_pretrained(a.output_dir/'final_checkpoint')
        status.update(status='complete',stage='complete',best_validation_subset_loss=best,finished_at=datetime.now().astimezone().isoformat());save()
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise
    finally:lock.close()


if __name__=='__main__':main()
