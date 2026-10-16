import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("chronological_subset", Path(__file__).parents[1] / "data/build_amazon23_chronological_subset.py")
subset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subset_module)
build = subset_module.build
usable_product_text = subset_module.usable_product_text
export_spec = importlib.util.spec_from_file_location("export_embeddings", Path(__file__).parents[1] / "data/export_amazon23_embeddings.py")
export_module = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(export_module)
from rq.text2emb import amazon23_summary_image_embed as embed
from rq.datasets import EmbDataset
from rq.summary.generate_structured_summaries import OllamaTextModel


class ChronologicalSubsetTests(unittest.TestCase):
    def test_ollama_residency_number_and_metadata_only_request(self):
        with patch("rq.summary.generate_structured_summaries.requests.post") as post:
            post.return_value.json.return_value = {"message": {"content": "{}"}}
            OllamaTextModel("vision", "http://localhost", 512, num_ctx=8192, keep_alive="-1").generate("paper")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["keep_alive"], -1)
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["messages"][1]["images"], [])
    def test_sid_rows_keep_original_ids_and_reject_invalid_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "emb.npy", np.array([[1., 2.], [3., 4.]], dtype=np.float32))
            np.save(str(root / "emb.npy")+".item_ids.npy", np.array([0, 7], dtype=np.int64))
            dataset = EmbDataset(str(root / "emb.npy"))
            self.assertEqual(dataset.item_ids.tolist(), [0, 7])
            np.save(root / "bad.npy", np.array([[np.nan, 1.]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "finite product vectors"):
                EmbDataset(str(root / "bad.npy"))
    def test_generic_or_empty_metadata_cannot_fabricate_product_text(self):
        self.assertFalse(usable_product_text(dict(brand="Acme", categories=["Office"],
                                                 title="unknown", features=["", None], details={"Color": ""})))
        self.assertTrue(usable_product_text(dict(features=["A4 refill paper"])))
    def test_cutoff_ties_missing_metadata_duplicates_and_singletons_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                dict(timestamp=4000, parent_asin="later", user_id="u"),
                dict(timestamp=2000, parent_asin="no-image", user_id="u"),
                dict(timestamp=1000, parent_asin="a", user_id="u"),
                dict(timestamp=2000, parent_asin="missing-meta", user_id="singleton"),
                dict(timestamp=2000, parent_asin="no-image", user_id="u"),
                dict(timestamp=500, parent_asin="before", user_id="other"),
            ]
            reviews = root / "reviews.gz"
            with gzip.open(reviews, "wt") as f:
                for r in rows: f.write(json.dumps(r) + "\n")
            catalog = root / "catalog.jsonl"
            catalog.write_text(json.dumps(dict(parent_asin="a", title="pen", images=["image"])) + "\n" +
                               json.dumps(dict(parent_asin="no-image", title="paper", images=[])) + "\n")
            stats = build(reviews, catalog, root / "out", 1000, target=2, train_ratio=.5, valid_ratio=.25)
            self.assertEqual(stats["cutoff_timestamp_ms"], 2000)
            self.assertEqual(stats["items"], 3)
            self.assertEqual(stats["interactions"], 4)
            self.assertEqual(stats["metadata_missing"], 1)
            self.assertEqual(stats["without_image_urls"], 2)
            self.assertEqual(stats["without_usable_text"], 1)
            history = json.loads((root / "out/Office_Products.inter.json").read_text())
            self.assertEqual(sorted(map(len, history.values())), [1, 3])
            samples = [json.loads(x) for x in (root / "out/Office_Products.samples.jsonl").read_text().splitlines()]
            self.assertEqual(len({s["split"] for s in samples if s["timestamp_ms"] == 2000}), 1)
            self.assertEqual(samples[-1]["history_item_ids"], [0, 2])
            self.assertEqual(sum(len(h) for h in history.values()), 4)

    def run_mixed(self, root, broken=False, anomalous=False):
        image = root / "image.png"
        Image.new("RGB", (20, 30)).save(image)
        items = [dict(item_id=0, parent_asin="a", images=[str(root / "missing.png") if broken else str(image)],
                      image_status="available", text_usable=True),
                 dict(item_id=1, parent_asin="b", images=[], image_status="no_image_urls", text_usable=True)]
        if anomalous: items.append(dict(item_id=2, parent_asin="c", images=[], text_usable=False))
        catalog = root / "catalog.jsonl"
        catalog.write_text("".join(json.dumps(i) + "\n" for i in items))
        summaries = root / "summary.jsonl"
        summaries.write_text("".join(json.dumps(dict(item_id=i["item_id"], parent_asin=i["parent_asin"],
            summary_text="pen", summary_version="product-attributes-v2", summary_image_used=bool(i["images"]))) + "\n"
            for i in items if i["text_usable"]))
        model = root / "model.gguf"; model.touch()
        projector = root / "projector.gguf"; projector.touch()
        argv = ["embed", "--catalog", str(catalog), "--summary_path", str(summaries), "--output_dir", str(root / "out"),
                "--model", str(model), "--mmproj", str(projector), "--allow_unembeddable"]
        with patch("sys.argv", argv), patch.object(embed.requests, "get") as get, \
                patch.object(embed, "gguf_embed", return_value=np.ones((1, 3), dtype=np.float32)) as call:
            get.return_value.json.side_effect = [{"models": [{"capabilities": ["multimodal"]}]},
                                                {"media_marker": "<media>", "modalities": {"vision": True}}]
            embed.main()
            return call.call_args_list

    def test_mixed_per_item_modes_and_exception_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = self.run_mixed(root, anomalous=True)
            self.assertIn("image", calls[0].args[1][0])
            self.assertNotIn("image", calls[1].args[1][0])
            with np.load(next((root / "out").glob("*.npz"))) as d:
                self.assertEqual(d["item_id"].tolist(), [0, 1])
                self.assertEqual(d["embedding_mode"].tolist(), ["joint", "text"])
                self.assertEqual(d["image_used"].tolist(), [True, False])
            self.assertEqual(json.loads((root / "out/exceptions.json").read_text())[0]["item_id"], 2)
            # U+2028 is a legal character inside a JSONL string, not a new record.
            with (root / "catalog.jsonl").open() as f: items = list(map(json.loads, f))
            items[0]["title"] = "pen\u2028fine"
            (root / "catalog.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False)+"\n" for i in items))
            validation = export_module.export(root / "catalog.jsonl", root / "out", root / "emb.npy")
            self.assertEqual(validation["unembeddable_items"], 1)
            self.assertEqual(validation["embedded_items"], 2)
            self.assertEqual(np.load(str(root / "emb.npy")+".item_ids.npy").tolist(), [0, 1])
            self.assertFalse(validation["row_ids_are_dense_catalog_ids"])

    def test_broken_image_never_silently_uses_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "refusing silent text fallback"):
                self.run_mixed(Path(tmp), broken=True)


if __name__ == "__main__": unittest.main()
