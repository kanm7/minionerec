import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
import sys
from unittest.mock import patch

directory=Path(__file__).parents[1]/'data'
spec=importlib.util.spec_from_file_location('official_5core',directory/'prepare_amazon23_official_5core.py')
module=importlib.util.module_from_spec(spec)
with patch.object(sys,'path',[str(directory),*sys.path]):spec.loader.exec_module(module)
build,T1,T2=module.build,module.T1,module.T2


class OfficialFiveCoreTests(unittest.TestCase):
    def prepare(self, root, corrupt=False):
        raw=root/'raw';raw.mkdir()
        full=[];splits={s:[] for s in ['train','valid','test']}
        for u in range(5):
            history=[]
            for i,t in enumerate([T1-2,T1-1,T1,T2,T2+1]):
                r=dict(user_id=str(u),parent_asin=f'ASIN{i}',rating=5,timestamp=t)
                full.append(r)
                split='train' if t<T1 else 'valid' if t<T2 else 'test'
                splits[split].append(dict(r,history=' '.join(history)))
                history.append(r['parent_asin'])
        if corrupt:splits['valid'][0]['history']='ASIN4'
        for name,rows in [('rating_only',full),*splits.items()]:
            with gzip.open(raw/f'{name}.csv.gz','wt',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with gzip.open(raw/'meta.jsonl.gz','wt') as f:
            # Product four lacks metadata; interactions must still be retained.
            for i in range(4):f.write(json.dumps(dict(parent_asin=f'ASIN{i}',title=f'Product {i}',images=[]))+'\n')
        return raw

    def test_full_coverage_official_boundaries_histories_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=self.prepare(root);out=root/'data'
            build(raw,out,'Custom')
            s=json.loads((out/'subset.stats.json').read_text())
            self.assertEqual(s['split_counts'],dict(train=10,valid=5,test=10))
            self.assertEqual(s['items'],5);self.assertEqual(s['interactions'],25)
            self.assertEqual(s['min_user_interactions'],5)
            self.assertEqual(s['without_usable_text'],1)
            rows=[json.loads(r) for r in (out/'Custom.samples.jsonl').open()]
            target=[r for r in rows if r['source_user_id']=='0' and r['parent_asin']=='ASIN4'][0]
            self.assertEqual(target['history_item_ids'],[0,1,2,3])
            self.assertEqual(target['split'],'test')

    def test_future_or_wrong_official_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=self.prepare(root,True)
            with self.assertRaisesRegex(ValueError,'History mismatch'):
                build(raw,root/'data','Custom')


if __name__=='__main__':unittest.main()
