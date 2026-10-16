"""Read-only comparison progress display; Ctrl+C does not stop evaluation."""
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]/'output/Industrial_Base_CPT_SFT_test'
last=None
try:
 while True:
  summary=json.loads((ROOT/'comparison_status.json').read_text())
  key=[summary['status'],summary['phase']];lines=[]
  for name in ['base','cpt','sft']:
   p=ROOT/name/'status.json'
   if not p.exists():lines.append(f'{name}: 等待');continue
   s=json.loads(p.read_text());done=s.get('completed',0);total=s.get('total',50985)
   key.extend([name,done,s['status']]);lines.append(f"{name}: {done}/{total} ({done/total:.1%}) | {s['stage']}")
  if key!=last:print('\n'+summary['phase']+'\n'+'\n'.join(lines),flush=True);last=key
  if summary['status'] in ['complete','failed','paused']:
   print(summary.get('error',summary['status']),flush=True);break
  time.sleep(2)
except KeyboardInterrupt:pass
