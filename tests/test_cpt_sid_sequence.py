import unittest
from cpt_sid_sequence import BucketBatches, collate


class CPTSequenceTests(unittest.TestCase):
    def test_padding_is_excluded_from_causal_labels(self):
        rows=[[10,11,12,13,14,15],[20,21,22,23,24,25,26,27,28]]
        b=collate(rows,0)
        self.assertEqual(b['labels'][0].tolist(),[10,11,12,13,14,15,-100,-100,-100])
        self.assertEqual(b['attention_mask'][0].tolist(),[1,1,1,1,1,1,0,0,0])
        self.assertEqual(b['labels'][1].tolist(),rows[1])

    def test_bucketing_preserves_every_sample_exactly_once(self):
        class Data:
            rows=[[1]*(i%100+6) for i in range(1027)]
            def __len__(self):return len(self.rows)
        batches=list(BucketBatches(Data(),4,42))
        self.assertEqual(sorted(i for b in batches for i in b),list(range(1027)))
        self.assertEqual(len(batches),257)
        self.assertEqual(batches,list(BucketBatches(Data(),4,42)))


if __name__=='__main__':unittest.main()
