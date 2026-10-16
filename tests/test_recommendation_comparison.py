import unittest
import math
from evaluate_stage_comparison import ranking_metrics
class RecommendationMetricsTests(unittest.TestCase):
 def test_exact_rank_boundaries(self):
  predictions=[str(i) for i in range(10)]
  for rank in [0,2,3,4,5,9]:
   result=ranking_metrics(predictions,str(rank))
   for k in [3,5,10]:
    self.assertEqual(result[f'HitRate@{k}'],float(rank<k))
    self.assertAlmostEqual(result[f'NDCG@{k}'],1/math.log2(rank+2) if rank<k else 0.)
 def test_miss_and_duplicate_rejection(self):
  self.assertTrue(all(v==0 for v in ranking_metrics(['a','b'],'c').values()))
  with self.assertRaises(ValueError):ranking_metrics(['a','a'],'a')
if __name__=='__main__':unittest.main()
