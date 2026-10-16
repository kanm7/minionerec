"""Read-only SFT progress display."""
import json
from pathlib import Path
import time
root=Path(__file__).resolve().parents[1]
folder=root/'output/Industrial_SFT_response_batch8'
print('SFT 实时进度；Ctrl+C 只退出显示，不停止训练。',flush=True)
last=None
try:
 while True:
  try:
   s=json.loads((folder/'status.json').read_text());step=s.get('optimizer_steps',0);total=s.get('total_optimizer_steps',0)
   key=(s['status'],s['stage'],step)
   if key!=last:
    progress=f'{step}/{total} ({step/total:.1%})' if total else '准备中'
    print(f"{progress} | {s['stage']} | response loss: {s.get('last_train_loss','—')}",flush=True);last=key
    if s['status'] in ['complete','failed','paused']:
     print(s.get('error',s['status']),flush=True);break
  except FileNotFoundError:pass
  time.sleep(2)
except KeyboardInterrupt:pass
