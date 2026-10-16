import json
from pathlib import Path
import struct
import tempfile
import unittest
from tools.cleanup_completed_sft import cleanup_completed_sft

class SFTCleanupTests(unittest.TestCase):
 def setup_run(self,root,status='complete'):
  (root/'status.json').write_text(json.dumps(dict(status=status,stage=status,full_validation_loss=1.2)))
  (root/'metrics.jsonl').write_text('keep log')
  final=root/'final_checkpoint';final.mkdir()
  header=json.dumps({'model.embed_tokens.weight':dict(dtype='F32',shape=[2,3],data_offsets=[0,24])}).encode()
  model=struct.pack('<Q',len(header))+header+b'\0'*24
  (final/'model.safetensors').write_bytes(model)
  (final/'config.json').write_text(json.dumps(dict(vocab_size=2,hidden_size=3)))
  (final/'tokenizer.json').write_text(json.dumps(dict(model=dict(vocab={'a':0,'b':1}),added_tokens=[])))
  (final/'tokenizer_config.json').write_text('{}')
  for name in ['best_checkpoint','last_checkpoint']:
   p=root/name;p.mkdir();(p/'model.safetensors').write_bytes(model)
   (p/'training_state.pt').write_bytes(b'optimizer state')
  return final
 def test_cleanup_retains_final_and_logs_and_is_idempotent(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);final=self.setup_run(root);before=(final/'model.safetensors').read_bytes()
   report=cleanup_completed_sft(root)
   self.assertEqual(report['status'],'complete');self.assertFalse((root/'last_checkpoint').exists());self.assertFalse((root/'best_checkpoint').exists())
   self.assertEqual((root/'metrics.jsonl').read_text(),'keep log');self.assertEqual((final/'model.safetensors').read_bytes(),before)
   self.assertEqual(cleanup_completed_sft(root),report)
 def test_running_sft_is_never_cleaned(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);self.setup_run(root,'running')
   with self.assertRaises(ValueError):cleanup_completed_sft(root)
   self.assertTrue((root/'last_checkpoint/training_state.pt').exists())
 def test_mismatched_final_preserves_best_and_last(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);self.setup_run(root)
   (root/'best_checkpoint/model.safetensors').write_bytes(b'different best model')
   with self.assertRaises(ValueError):cleanup_completed_sft(root)
   self.assertTrue((root/'last_checkpoint').exists());self.assertTrue((root/'best_checkpoint').exists())
 def test_truncated_export_preserves_recovery_state(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);final=self.setup_run(root)
   p=final/'model.safetensors';p.write_bytes(p.read_bytes()[:-1])
   with self.assertRaises(ValueError):cleanup_completed_sft(root)
   self.assertTrue((root/'last_checkpoint/training_state.pt').exists())
if __name__=='__main__':unittest.main()
