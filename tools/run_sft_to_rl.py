"""Wait for completed SFT, tune Apple GPU RL memory, smoke-test, then train ranking GRPO."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from cleanup_completed_sft import cleanup_completed_sft
ROOT=Path(__file__).resolve().parents[1]
SFT=ROOT/'output/Industrial_SFT_response_batch8'
WORK=ROOT/'output/Industrial_SFT_to_RL_ranking'

def main():
 WORK.mkdir(exist_ok=True)
 lock=(WORK/'controller.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('The SFT-to-RL controller is already running')
 status=dict(status='running',stage='waiting_for_sft',pid=os.getpid(),reward_type='ranking',
  sft_dir=str(SFT),started_at=datetime.now().astimezone().isoformat(),memory_limit_gib=44,
  maximum_probe_peak_gib=40,effective_prompt_batch=8,num_generations=16,effective_completion_batch=128)
 def save(**kwargs):
  status.update(kwargs);temp=WORK/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(WORK/'status.json')
  print(json.dumps(kwargs),flush=True)
 def cancel_requested():return (WORK/'CANCEL').exists()
 try:
  save()
  last=None
  while True:
   if cancel_requested():save(status='cancelled',stage='cancelled_before_rl');return
   s=json.loads((SFT/'status.json').read_text())
   progress=(s['status'],s.get('optimizer_steps'),s.get('stage'))
   if progress!=last:save(sft_status=s['status'],sft_step=s.get('optimizer_steps'),sft_stage=s.get('stage'));last=progress
   if s['status']=='complete':break
   if s['status'] in ['failed','paused','stopped']:
    save(status='blocked',stage='sft_not_complete',reason='SFT paused or failed; RL will not start');return
   time.sleep(30)
  model=SFT/'final_checkpoint'
  for filename in ['model.safetensors','tokenizer.json','config.json']:
   if not (model/filename).is_file():raise RuntimeError(f'Missing SFT artifact: {filename}')
  if 'full_validation_loss' not in s:raise RuntimeError('SFT full validation did not finish')
  # Wait for the SFT process to release GPU allocations, not just its complete status.
  while True:
   try:os.kill(s['pid'],0)
   except ProcessLookupError:break
   if cancel_requested():save(status='cancelled',stage='cancelled_before_rl');return
   time.sleep(2)
  save(stage='cleaning_completed_sft_checkpoints')
  cleanup=cleanup_completed_sft(SFT)
  save(sft_cleanup_status=cleanup['status'],sft_cleanup_removed_gib=cleanup['removed_gib'])
  free=shutil.disk_usage(ROOT).free/2**30
  save(stage='checking_resources',free_disk_gib=free)
  if free<32:raise RuntimeError(f'Only {free:.1f} GiB disk free; need at least 32 GiB for RL checkpoints and reserve')
  base=[sys.executable,'-u',str(ROOT/'rl_mps.py'),'--model_path',str(model),'--reward_type','ranking',
   '--num_generations','16','--prompts_per_update','8','--epochs','1','--learning_rate','0.000001','--beta','0.04']
  trials=[]
  def probe(micro,rollout):
   if cancel_requested():raise RuntimeError('Automatic RL start cancelled')
   destination=WORK/f'probe-micro{micro}-rollout{rollout}'
   args=base+['--output_dir',str(destination),'--micro_batch_size',str(micro),'--rollout_batch_size',str(rollout),'--probe']
   save(stage='memory_tuning',probe_micro_batch=micro,probe_rollout_batch=rollout)
   with (WORK/f'probe-micro{micro}-rollout{rollout}.log').open('w') as log:
    result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   record=json.loads((destination/'status.json').read_text())
   passed=result.returncode==0 and record['status']=='complete' and record['peak_driver_gib']<=40
   trial=dict(micro_batch_size=micro,rollout_batch_size=rollout,passed=passed,
    peak_driver_gib=record.get('peak_driver_gib'),seconds_per_prompt=record.get('seconds_per_prompt'),error=record.get('error'))
   trials.append(trial);(WORK/'memory_trials.json').write_text(json.dumps(trials,indent=2));save(last_probe=trial)
   if result.returncode and 'out of memory' not in record.get('error','').lower():raise RuntimeError(f'RL compatibility probe failed: {record.get("error")}')
   return trial
  for micro in [2,4,8,16]:
   trial=probe(micro,1)
   if not trial['passed']:break
  passed=[t for t in trials if t['passed']]
  if not passed:raise RuntimeError('No RL microbatch passed memory/compatibility tests')
  fastest=min(passed,key=lambda t:t['seconds_per_prompt']);micro=fastest['micro_batch_size']
  for rollout in [2,4]:
   trial=probe(micro,rollout)
   if not trial['passed']:break
  # Explore larger completion microbatches when enough generation grouping permits it.
  if any(t['passed'] and t['micro_batch_size']==16 for t in trials):
   trial=probe(32,2)
   if trial['passed']:probe(64,4)
  chosen=min((t for t in trials if t['passed']),key=lambda t:t['seconds_per_prompt'])
  selected=dict(**chosen,gradient_accumulation_steps=128//chosen['micro_batch_size'],
   memory_limit_gib=44,probe_headroom_gib=44-chosen['peak_driver_gib'],reward_type='ranking')
  (WORK/'selected_configuration.json').write_text(json.dumps(selected,indent=2))
  args=base+['--micro_batch_size',str(chosen['micro_batch_size']),'--rollout_batch_size',str(chosen['rollout_batch_size'])]
  smoke=WORK/'smoke';save(stage='real_reward_smoke_test',selected=selected)
  with (WORK/'smoke.log').open('w') as log:
   result=subprocess.run(args+['--output_dir',str(smoke),'--smoke_steps','2','--eval_samples','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  check=json.loads((smoke/'status.json').read_text())
  if result.returncode or check.get('stage')!='smoke_complete':raise RuntimeError(f'Actual ranking RL smoke failed: {check.get("error")}')
  if cancel_requested():save(status='cancelled',stage='cancelled_before_rl');return
  if shutil.disk_usage(ROOT).free<32*2**30:raise RuntimeError('Insufficient disk space after probes; RL not started')
  destination=ROOT/'output/Industrial_RL_ranking_mps'
  command=args+['--output_dir',str(destination)]
  (WORK/'formal_command.json').write_text(json.dumps(command,indent=2))
  save(stage='rl_training',rl_output_dir=str(destination))
  # Stream formal RL output to the visible terminal; no detached trainer.
  result=subprocess.run(command,cwd=ROOT)
  final=json.loads((destination/'status.json').read_text())
  if result.returncode or final['status']=='failed':raise RuntimeError(final.get('error','RL process failed'))
  save(status=final['status'],stage='rl_'+final['stage'],finished_at=datetime.now().astimezone().isoformat())
 except KeyboardInterrupt:save(status='cancelled',stage='controller_interrupted')
 except BaseException as exc:
  save(status='blocked',stage='blocked',error=f'{type(exc).__name__}: {exc}')
  raise
 finally:lock.close()
if __name__=='__main__':main()
