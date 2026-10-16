import unittest
import torch
from rq.trainer import Trainer


class SchedulerTests(unittest.TestCase):
    def schedule(self, kind):
        trainer=Trainer.__new__(Trainer)
        trainer.optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=0.0003)
        trainer.lr_scheduler_type=kind;trainer.warmup_steps=2;trainer.max_steps=10
        scheduler=trainer._get_scheduler()
        rates=[trainer.optimizer.param_groups[0]['lr']]
        for _ in range(10):
            trainer.optimizer.step();scheduler.step()
            rates.append(trainer.optimizer.param_groups[0]['lr'])
        return rates

    def test_linear_warms_up_then_decays_to_zero(self):
        rates=self.schedule('linear')
        self.assertEqual(rates[0],0.)
        self.assertAlmostEqual(rates[2],0.0003)
        self.assertEqual(rates[-1],0.)
        self.assertTrue(all(left>right for left,right in zip(rates[2:-1],rates[3:])))

    def test_constant_preserves_baseline_after_warmup(self):
        rates=self.schedule('constant')
        self.assertTrue(all(abs(r-0.0003)<1e-12 for r in rates[2:]))


if __name__=='__main__':unittest.main()
