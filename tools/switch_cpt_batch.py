"""Switch at the next saved checkpoint; display the resumed trainer in this terminal."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
source=root/'output/Industrial_CPT_pure_sid_resume2'
marker=root/'output/cpt_batch_switch.json'
status=json.loads((source/'status.json').read_text())
pid=status['pid'];target=(status['optimizer_steps']//250+1)*250
marker.write_text(json.dumps(dict(status='waiting_for_checkpoint',source=str(source),pid=pid,target_step=target)))
print(f'等待第 {target} 步检查点保存完成，再切换单批次 4。',flush=True)
last=None
while True:
    status=json.loads((source/'status.json').read_text())
    step=status.get('optimizer_steps',0)
    if step!=last:
        print(f'当前训练 {step}/4063 ({step/4063:.1%})，loss={status.get("last_train_loss")}',flush=True);last=step
    if status['status']!='running':raise SystemExit('原训练已停止，取消自动切换。')
    # The update after the target proves the preceding checkpoint save returned.
    if step>target:break
    time.sleep(2)
os.kill(pid,signal.SIGINT)
for _ in range(120):
    try:os.kill(pid,0)
    except ProcessLookupError:break
    time.sleep(1)
else:raise SystemExit('原进程尚未退出，取消切换以免同时加载模型。')
checkpoint=source/'last_checkpoint'
for batch,accum,name in [(4,16,'Industrial_CPT_pure_sid_batch4'),(2,32,'Industrial_CPT_pure_sid_batch2_fallback')]:
    destination=root/'output'/name
    marker.write_text(json.dumps(dict(status='starting',batch=batch,output_dir=str(destination),checkpoint=str(checkpoint))))
    print(f'从保存的检查点恢复：单批次 {batch}，梯度累积 {accum}；先检查最长序列。',flush=True)
    args=[sys.executable,str(root/'cpt_sid_sequence.py'),'--output_dir',str(destination),
        '--resume_from_checkpoint',str(checkpoint),'--epochs','1','--micro_batch_size',str(batch),
        '--gradient_accumulation_steps',str(accum),'--sampler_batch_size','4',
        '--eval_steps','250','--eval_samples','512','--preflight_longest_batch']
    code=subprocess.call(args,cwd=root)
    result=json.loads((destination/'status.json').read_text())
    if code==0:
        marker.write_text(json.dumps(dict(status='complete',output_dir=str(destination))));break
    if batch==4 and 'out of memory' in result.get('error','').lower():
        # Continue from any newer fully written checkpoint rather than losing updates.
        newer=destination/'last_checkpoint'
        if (newer/'training_state.pt').exists():checkpoint=newer
        print('单批次 4 内存不足，自动恢复单批次 2。',flush=True)
        continue
    marker.write_text(json.dumps(dict(status='failed',output_dir=str(destination),error=result.get('error'))))
    raise SystemExit(code)
