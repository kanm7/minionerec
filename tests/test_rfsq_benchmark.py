import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from rq.datasets import EmbDataset
from rq.models.rfsq import RobustResidualFSQ, FiniteScalarQuantizer
from rq.sid_metrics import compute_metrics
from rq.trainer import Trainer


class RFSQBenchmarkTests(unittest.TestCase):
    def test_input_clamp_changes_saturation_gradient_not_forward_codes(self):
        x=torch.tensor([[2.,-2.]],requires_grad=True)
        bounded=FiniteScalarQuantizer([4,2],input_clamp=True)
        legacy=FiniteScalarQuantizer([4,2])
        y,ids=bounded(x);old,old_ids=legacy(x)
        torch.testing.assert_close(y,old);torch.testing.assert_close(ids,old_ids)
        torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],torch.zeros_like(x))
        torch.testing.assert_close(torch.autograd.grad(old.sum(),x)[0],torch.ones_like(x))

    def test_finite_scalar_coordinates_and_inverse_with_extreme_inputs(self):
        q = FiniteScalarQuantizer([4, 2])
        values = torch.tensor([[-100., 100.], [100., -100.], [0., 0.]], requires_grad=True)
        codes, indices = q(values)
        self.assertTrue(((indices >= 0) & (indices < 8)).all())
        torch.testing.assert_close(codes, q.indices_to_codes(indices))
        codes.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))
        # Legacy mode remains available only for reproducing old checkpoints.
        _, old_indices = FiniteScalarQuantizer([4, 2], scalar_bound=False)(values)
        self.assertTrue(((old_indices < 0) | (old_indices >= 8)).any())

    def test_residual_scale_cannot_escape_finite_cells(self):
        q = RobustResidualFSQ([[4, 2], [4, 2], [4, 2]])
        with torch.no_grad(): q.raw_scales.fill_(20.)
        _, indices = q(torch.tensor([[100., -100.], [.3, -.7]]))
        self.assertTrue(((indices >= 0) & (indices < 8)).all())

    def test_native_implicit_cells_match_hand_computed_metrics(self):
        q=RobustResidualFSQ([[2,2],[2,2]])
        indices=torch.tensor([[0,0],[0,0],[0,1],[1,2]])
        meta={str(i):{"categories":["Office",c]} for i,c in zip([10,20,30,40],["A","B","A","C"])}
        r=q.benchmark(indices,[10,20,30,40],item_meta=meta,popularity={"10":10,"20":1,"40":2})
        self.assertFalse(r["learned_codebook_vectors"])
        self.assertEqual(r["nominal_stage_capacities"],[4,4])
        self.assertEqual(r["layers"][0]["occupancy_rate"],.5)
        self.assertEqual(r["full_sid"]["exclusive_item_rate"],.5)
        self.assertEqual(r["full_sid_category_purity"],.75)

    def test_out_of_range_ids_are_reported_not_clamped(self):
        q=RobustResidualFSQ([[2,2]])
        r=q.benchmark(torch.tensor([[-1],[0],[4]]))
        self.assertEqual(r["out_of_range_index_counts"],[2])
        self.assertIsNone(r["layers"][0]["occupancy_rate"])
        self.assertEqual(r["layers"][0]["valid_code_occupancy_rate"],.25)
        self.assertEqual(r["items"],3)

    def test_default_stage_capacity_is_2048_without_centroid_table(self):
        q=RobustResidualFSQ([[4,4,4,2,2,2,2,2]]*3)
        self.assertEqual(q.codebook_sizes,[2048]*3)
        self.assertFalse(any(isinstance(m,torch.nn.Embedding) for m in q.modules()))

    def test_training_evaluation_stable_ids_full_coverage_and_saved_reports(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.quantizer=RobustResidualFSQ([[2,2]])
            def get_indices(self, values):return values[:,0].long().unsqueeze(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'emb.npy',np.array([[0.,1.],[0.,2.],[1.,3.]],dtype=np.float32))
            np.save(str(root/'emb.npy')+'.item_ids.npy',np.array([10,20,30]))
            data=EmbDataset(str(root/'emb.npy'))
            trainer=Trainer.__new__(Trainer)
            trainer.model=Model();trainer.device=torch.device('cpu');trainer.args=SimpleNamespace()
            trainer.rfsq_benchmark=trainer.model.quantizer.benchmark
            trainer.benchmark_meta={"10":{"categories":["A"]},"20":{"categories":["B"]},"30":{"categories":["A"]}}
            trainer.benchmark_popularity={"10":4,"20":1,"30":2}
            trainer.last_sid_benchmark=None;trainer.ckpt_dir=str(root);trainer.logger=logging.getLogger(__name__)
            trainer.optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=.001)
            # Input training loader shuffles and drops the tail. Evaluation must
            # reconstruct stable order and include the omitted row.
            collision=trainer._valid_epoch(DataLoader(data,batch_size=2,shuffle=True,drop_last=True))
            self.assertAlmostEqual(collision,1/3)
            self.assertEqual(trainer.last_sid_benchmark['items'],3)
            self.assertAlmostEqual(trainer.last_sid_benchmark['full_sid_category_purity'],2/3)
            trainer._save_sid_benchmark(50)
            saved=json.loads((root/'sid_benchmark/epoch_000050.json').read_text())
            self.assertEqual(saved['epoch'],50)
            self.assertEqual(saved['learning_rate'],.001)
            self.assertEqual(saved['sid_variant'],'raw')
            self.assertEqual(json.loads((root/'sid_benchmark/history.jsonl').read_text()),saved)


if __name__=='__main__':unittest.main()
