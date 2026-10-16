import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from rq.sid_metrics import compute_metrics, load_items, load_popularity


class SIDMetricsTests(unittest.TestCase):
    def setUp(self):
        self.indices = {"0": ["<a_0>", "<b_0>"], "1": ["<a_0>", "<b_0>"],
                        "2": ["<a_0>", "<b_1>"], "3": ["<a_1>", "<b_2>"]}
        self.meta = {str(i): {"categories": ["Office", c]} for i, c in enumerate(("A", "B", "A", "C"))}

    def test_hand_calculated_exclusivity_capacity_collisions_and_purity(self):
        r = compute_metrics(self.indices, [4, 4], self.meta)
        self.assertEqual(r["unique_sid_rate"], .75)
        self.assertEqual(r["collision_rate"], .25)
        self.assertEqual(r["full_sid"]["collision_item_rate"], .5)
        self.assertEqual(r["full_sid"]["exclusive_item_rate"], .5)
        self.assertAlmostEqual(r["full_sid"]["singleton_rate"], 2/3)
        self.assertEqual(r["layers"][0]["occupancy_rate"], .5)
        self.assertEqual(r["prefix"][1]["occupancy_rate"], 3/16)
        self.assertEqual(r["full_sid"]["purity"]["item_weighted_purity"], .75)
        self.assertEqual(r["full_sid"]["purity"]["non_singleton_cluster_item_weighted_purity"], .5)
        self.assertEqual(r["full_sid"]["purity"]["labeled_item_share_in_singleton_clusters"], .5)
        self.assertAlmostEqual(r["full_sid"]["capacity_normalized_entropy"], .375)

    def test_hot_clusters_use_interactions_not_item_counts(self):
        r = compute_metrics(self.indices, [4,4], popularity={"0":10,"1":1,"2":0,"3":2}, hot_percent=50)
        hot = r["full_sid"]["hot_clusters"]
        self.assertEqual(hot["cluster_count"],2)
        self.assertEqual(hot["item_share"],.75)
        self.assertEqual(hot["popularity_share"],1.)
        self.assertEqual(r["layers"][0]["hot_clusters"]["popularity_share"],11/13)
        r=compute_metrics(self.indices,popularity={"0":10,"1":1,"2":0,"3":2},hot_multiplier=2)
        self.assertAlmostEqual(r["full_sid"]["hot_clusters_above_mean"]["cluster_rate"],1/3)
        self.assertEqual(r["full_sid"]["hot_clusters_above_mean"]["popularity_share"],11/13)
        r=compute_metrics(self.indices,popularity={})
        self.assertIsNone(r["full_sid"]["hot_clusters_above_mean"]["cluster_rate"])

    def test_missing_labels_and_singletons_are_not_misleading(self):
        r=compute_metrics(self.indices,item_meta={"2":self.meta["2"]})
        self.assertEqual(r["full_sid"]["purity"]["labeled_item_coverage"],.25)
        self.assertIsNone(r["full_sid"]["purity"]["non_singleton_cluster_item_weighted_purity"])
        r=compute_metrics(self.indices,item_meta={})
        self.assertIsNone(r["full_sid_category_purity"])
        r=compute_metrics(self.indices,item_meta=self.meta,category_level="root")
        self.assertEqual(r["full_sid_category_purity"],1.)
        self.assertEqual(r["full_sid"]["purity"]["distinct_labels"],1)

    def test_bad_capacities_and_weights_fail(self):
        for sizes in ([2], [2,0], [1,4], [4,2]):
            with self.assertRaises(ValueError):compute_metrics(self.indices,sizes)
        for pop in ({"0":-1},{"0":float("nan")}):
            with self.assertRaises(ValueError):compute_metrics(self.indices,popularity=pop)
        with self.assertRaises(ValueError):compute_metrics({"0":[]})

    def test_jsonl_unicode_and_training_only_popularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            items=root/"items.jsonl"
            items.write_text(json.dumps({"item_id":0,"categories":["Office","pen\u2028fine"]},ensure_ascii=False)+"\n")
            self.assertEqual(load_items(items)["0"]["categories"][-1],"pen\u2028fine")
            interactions=root/"interactions.jsonl"
            interactions.write_text(''.join(json.dumps(r)+"\n" for r in [
                {"item_id":0,"split":"train"},{"item_id":0,"split":"train"},
                {"item_id":1,"split":"test"},{"item_id":0,"split":"valid"}]))
            self.assertEqual(load_popularity(interactions),{"0":2})
            interactions.write_text('{"item_id":0}\n')
            with self.assertRaises(ValueError):load_popularity(interactions)

    def test_cli_raw_repaired_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=root/"raw.json";raw.write_text(json.dumps(self.indices))
            repaired=dict(self.indices);repaired["1"]=["<a_0>","<b_3>"]
            index=root/"repaired.json";index.write_text(json.dumps(repaired))
            meta=root/"meta.json";meta.write_text(json.dumps(self.meta))
            output=root/"nested/report.json"
            subprocess.run([sys.executable,str(Path(__file__).parents[1]/"rq/sid_metrics.py"),
                "--index_file",str(index),"--raw_index_file",str(raw),"--sid_variant","repaired",
                "--codebook_sizes","4","4","--item_file",str(meta),"--output_file",str(output)],
                check=True,capture_output=True,text=True)
            r=json.loads(output.read_text())
            self.assertEqual(r["full_sid"]["exclusive_item_rate"],1.)
            self.assertEqual(r["raw_comparison"]["full_sid"]["exclusive_item_rate"],.5)
            self.assertEqual(r["layers"][0],r["raw_comparison"]["layers"][0])


if __name__ == "__main__": unittest.main()
