"""Display live CPT status without modifying or restarting the trainer."""
import json
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parents[1]
folder=Path(sys.argv[1]) if len(sys.argv)>1 else root/'output/Industrial_CPT_pure_sid_resume'
last=None
print('Industrial & Scientific — CPT 实时进度',flush=True)
print('此窗口只显示进度；关闭窗口或 Ctrl+C 不会停止训练。',flush=True)
try:
    while True:
        try:
            status=json.loads((folder/'status.json').read_text())
            step=status.get('optimizer_steps',status.get('resumed_from_step',0))
            key=(status['status'],status['stage'],step)
            if key!=last:
                total=status.get('total_optimizer_steps',4063)
                loss=status.get('last_train_loss')
                memory=''
                if (folder/'metrics.jsonl').exists():
                    for line in reversed((folder/'metrics.jsonl').read_text().splitlines()):
                        metric=json.loads(line)
                        if metric.get('step')==step:
                            if 'mps_allocated_gib' in metric:
                                memory=f" | MPS 张量 {metric['mps_allocated_gib']:.1f} GiB / 驱动 {metric['mps_driver_gib']:.1f} GiB"
                            break
                filled=min(30,int(30*step/total))
                bar='█'*filled+'░'*(30-filled)
                text=f"[{bar}] {step}/{total} ({step/total:.1%}) | {status['stage']}"
                if loss is not None:text+=f' | loss {loss:.4f}'
                print(text+memory,flush=True);last=key
                if status['status'] in ('failed','complete'):
                    print(status.get('error','训练完成。'),flush=True)
                    break
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(2)
except KeyboardInterrupt:
    print('\n已关闭进度显示，训练继续运行。')
