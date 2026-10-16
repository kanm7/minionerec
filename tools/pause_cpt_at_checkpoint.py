"""Pause the current trainer only after its next checkpoint is fully saved."""
import json
import os
from pathlib import Path
import signal
import time

folder=Path(__file__).resolve().parents[1]/'output/Industrial_CPT_pure_sid_batch4'
status_path=folder/'status.json'
s=json.loads(status_path.read_text());pid=s['pid'];target=(s['optimizer_steps']//250+1)*250
(folder/'pause_request.json').write_text(json.dumps(dict(target_step=target,pid=pid,reason='User requested pause with saved progress')))
print(f'等待第 {target} 步保存完毕后暂停。',flush=True)
last=None
while True:
 s=json.loads(status_path.read_text());step=s.get('optimizer_steps',0)
 if step!=last:print(f'进度 {step}/4063；等待保存到 {target} 步',flush=True);last=step
 if s['status']!='running':raise SystemExit('训练已退出，请检查现有检查点。')
 if step>target:break
 time.sleep(2)
os.kill(pid,signal.SIGINT)
while True:
 try:os.kill(pid,0)
 except ProcessLookupError:break
 time.sleep(1)
s=json.loads(status_path.read_text())
s.update(status='paused',stage='paused',saved_optimizer_steps=target,
 resume_checkpoint=str(folder/'last_checkpoint'),pause_reason='User requested pausing and saving training progress')
s.pop('error',None)
temp=folder/'status.pause.tmp';temp.write_text(json.dumps(s,indent=2));temp.replace(status_path)
print(f'已暂停，可从第 {target} 步恢复：{folder/"last_checkpoint"}',flush=True)
