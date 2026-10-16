"""Remove only superseded SFT checkpoints after verifying a completed final export."""
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

def sha256(path):
 digest=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()

def verify_export(final):
 for name in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json']:
  path=final/name
  if not path.is_file() or path.is_symlink() or path.stat().st_size==0:raise ValueError(f'Missing/unsafe final artifact: {path}')
 config=json.loads((final/'config.json').read_text())
 path=final/'model.safetensors'
 with path.open('rb') as f:
  raw=f.read(8)
  if len(raw)!=8:raise ValueError('Truncated safetensors header')
  size=struct.unpack('<Q',raw)[0]
  if not 0<size<100*1024*1024:raise ValueError('Invalid safetensors header size')
  header=json.loads(f.read(size))
 tensors={k:v for k,v in header.items() if k!='__metadata__'}
 embedding=tensors.get('model.embed_tokens.weight',{})
 if embedding.get('shape')!=[config['vocab_size'],config['hidden_size']]:raise ValueError('Final model embedding/config mismatch')
 if not tensors or max(v['data_offsets'][1] for v in tensors.values())+8+size!=path.stat().st_size:raise ValueError('Truncated final model payload')
 tokenizer=json.loads((final/'tokenizer.json').read_text())
 ids=set(tokenizer['model']['vocab'].values())|{token['id'] for token in tokenizer.get('added_tokens',[])}
 if len(ids)!=config['vocab_size'] or min(ids)!=0 or max(ids)!=config['vocab_size']-1:raise ValueError('Final tokenizer/model vocabulary mismatch')
 return sha256(path)

def cleanup_completed_sft(run_dir):
 run=Path(run_dir)
 if run.is_symlink():raise ValueError('Refusing a symlinked training directory')
 with (run/'cleanup.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  state=json.loads((run/'status.json').read_text())
  if state.get('status')!='complete' or state.get('stage')!='complete':raise ValueError('SFT is not successfully complete; preserve recovery checkpoints')
  loss=state.get('full_validation_loss')
  if not isinstance(loss,(int,float)) or not math.isfinite(loss):raise ValueError('Missing finite full-validation result')
  final=run/'final_checkpoint'
  if final.is_symlink():raise ValueError('Refusing symlinked final export')
  final_hash=verify_export(final)
  targets=[]
  for name in ['best_checkpoint','last_checkpoint']:
   path=run/name
   if path.exists():
    if path.is_symlink() or path.resolve().parent!=run.resolve():raise ValueError('Unsafe deletion target')
    if name=='best_checkpoint' and sha256(path/'model.safetensors')!=final_hash:
     raise ValueError('Best model differs from final export; preserve both for investigation')
    targets.append(dict(path=str(path),bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())))
  report_path=run/'checkpoint_cleanup.json'
  # Repeated calls preserve the original successful cleanup audit.
  if not targets and report_path.exists():
   previous=json.loads(report_path.read_text())
   if previous.get('status')=='complete' and previous.get('retained_model_sha256')==final_hash:return previous
  report=dict(status='deleting',started_at=datetime.now().astimezone().isoformat(),targets=targets,
   retained_model=str(final),retained_model_sha256=final_hash,full_validation_loss=loss,
   policy='Keep final model/tokenizer and logs/config/plots; remove only superseded best/last SFT checkpoints after successful completion')
  report_path.write_text(json.dumps(report,indent=2))
  for target in targets:shutil.rmtree(target['path'])
  if sha256(final/'model.safetensors')!=final_hash:raise ValueError('Final model changed during cleanup')
  report.update(status='complete',finished_at=datetime.now().astimezone().isoformat(),removed_gib=sum(t['bytes'] for t in targets)/2**30)
  report_path.write_text(json.dumps(report,indent=2));return report
