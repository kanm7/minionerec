"""Pilot then resume full held-out inference for CPT, completed SFT, and Base+SID."""
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/Industrial_Base_CPT_SFT_test'

def main():
 OUT.mkdir(exist_ok=True)
 lock=(OUT/'controller.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Comparison already running')
 state=dict(status='running',phase='pilot',started_at=datetime.now().astimezone().isoformat(),models=['cpt','sft','base'],test_samples=50985)
 def save():
  p=OUT/'comparison_status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'comparison_status.json')
 def report():
  text='''# Base / CPT / SFT test-set HitRate and NDCG

Dataset: Industrial & Scientific external 25,847-item catalog; same held-out 50,985 interaction samples, seed-42 shuffled order. One target per sample, HitRate@K equals Recall@K. Full-catalog constrained 10-beam search (approximate search, not exhaustive item scoring), no negative sampling or seen-item filtering. K=3/5/10. Predictions and sample row IDs are saved for audit. No model selection or training uses this test run.

Models: Qwen2.5-1.5B Base with the same 5,954 added atomic SID tokens, untrained SID rows reinitialized with seed42 using the original CPT initialization procedure; CPT final selected checkpoint; SFT final selected checkpoint (step 3,500, training completed at 4,466). Base/CPT use the pure concatenated full-history SID input; SFT uses its exact training instruction + SID prompt, preserving 507 prompt tokens to match the original 512-token total cutoff with 5 response tokens and includes newline/EOS completion scores. These are stage-appropriate end-to-end interfaces, not a weights-only controlled ablation. Base/CPT score 3 SID tokens, SFT scores 3 SID tokens + newline/EOS, length penalty 0. The added Base SID embeddings have no learned product semantics. All model files remain unchanged and RL remains paused.

The first 200 common samples are a throughput pilot, not a final metric table. Full evaluation resumes those saved predictions. Separate model-load/warmup timing is recorded. Final results appear below only after all requested test samples for a model are complete.

| Model | Samples | HitRate@3 | HitRate@5 | HitRate@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
  all_results={}
  for name in ['base','cpt','sft']:
   path=OUT/name/'results.json'
   if path.exists():
    result=json.loads(path.read_text());all_results[name]=result;metrics=result['metrics']
    values=[f"{metrics[f'HitRate@{k}']:.4%}" for k in [3,5,10]]+[f"{metrics[f'NDCG@{k}']:.6f}" for k in [3,5,10]]
    text+=f"| {name} | {result['samples']:,} | "+' | '.join(values)+' |\n'
   else:text+=f'| {name} | Pending | — | — | — | — | — | — |\n'
  (OUT/'REPORT.md').write_text(text)
  (OUT/'results.json').write_text(json.dumps(all_results,indent=2))
 try:
  save();report()
  for pilot in [True,False]:
   state['phase']='pilot' if pilot else 'full_test'
   for name in ['cpt','sft','base']:
    state.update(model=name);save()
    path=OUT/name/'status.json'
    if path.exists() and json.loads(path.read_text())['status']=='complete':continue
    command=[sys.executable,'-u',str(ROOT/'evaluate_stage_comparison.py'),'--stage',name,'--output_dir',str(OUT),'--batch_size','4']
    if pilot:command+=['--stop_after','200']
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode:raise RuntimeError(f'{name} evaluation failed; inspect {path}')
    current=json.loads(path.read_text())
    if current['status']=='paused' and current['stage']!='pilot_complete':
     state.update(status='paused',phase='paused');save();return
    if pilot:
     pilots={}
     for stage in ['cpt','sft','base']:
      p=OUT/stage/'pilot_200.json'
      if p.exists():pilots[stage]=json.loads(p.read_text())
     (OUT/'pilot_summary.json').write_text(json.dumps(pilots,indent=2))
    report()
  state.update(status='complete',phase='complete',finished_at=datetime.now().astimezone().isoformat());save();report()
 except BaseException as exc:
  state.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise
 finally:lock.close()
if __name__=='__main__':main()
