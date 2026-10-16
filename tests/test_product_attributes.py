"""Focused pipeline checks without training or a loaded inference model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from rq.summary.generate_structured_summaries import (
    OllamaTextModel, canonical_summary_text, load_product_images, read_completed,
)
from rq.text2emb.amazon_summary_vl2emb import build_inputs, gguf_embed
from rq.text2emb import amazon23_summary_image_embed as catalog_embed


class ProductAttributeTests(unittest.TestCase):
    def test_canonicalization_preserves_model_and_sorts_attributes(self):
        a = {"product_type": " Phone   CASE ", "color": "clear",
             "model": "iPhone 15", "structural_features": ["Raised Rim", "Corners", "Corners"]}
        b = {"product_type": "phone case", "color": "transparent",
             "model": "iphone 15", "structural_features": ["corners", "raised rim"]}
        self.assertEqual(canonical_summary_text(a), canonical_summary_text(b))
        self.assertNotEqual(canonical_summary_text(a), canonical_summary_text(dict(b, model="iphone 16")))

    def test_image_extraction_loads_fallback_source_and_sends_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "product.png"
            Image.new("RGB", (40, 20)).save(image_path)
            images = load_product_images({"images": [str(Path(tmp) / "missing.png"), str(image_path)]}, 1, 10, 1)
            self.assertEqual(images[0].size, (10, 5))
            with patch("rq.summary.generate_structured_summaries.requests.post") as post:
                post.return_value.json.return_value = {"message": {"content": "{}"}}
                OllamaTextModel("vision", "http://localhost", 512).generate("phone case", images)
            self.assertTrue(post.call_args.kwargs["json"]["messages"][1]["images"][0])

    def test_text_embedding_never_loads_image_or_adds_marker(self):
        with patch("rq.text2emb.amazon_summary_vl2emb.load_image") as load:
            entries = list(build_inputs({"0": {"images": ["bad.png"]}}, {"0": "phone case"}, 1, 1, True))
        load.assert_not_called()
        with patch("rq.text2emb.amazon_summary_vl2emb.requests.post") as post:
            post.return_value.json.return_value = {"embedding": [1, 2]}
            gguf_embed("http://localhost", [entries[0][1]], None)
        self.assertEqual(post.call_args.kwargs["json"]["content"], "phone case")

    def test_old_summaries_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.jsonl"
            path.write_text(json.dumps({"item_id": 0, "summary_text": "phone case"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Old summary"):
                read_completed(path)
            with self.assertRaisesRegex(ValueError, "Old summary"):
                catalog_embed.load_summary_records(path)

    def test_verified_same_product_reuses_vector_across_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, summaries, mapping, model = [root / name for name in ("catalog.jsonl", "summary.jsonl", "map.json", "model.gguf")]
            model.touch()
            items = [{"item_id": i, "parent_asin": f"asin{i}", "images": ["missing.png"]} for i in range(2)]
            catalog.write_text("".join(json.dumps(item) + "\n" for item in items))
            summaries.write_text("".join(json.dumps(dict(item, summary_text=f"phone case {i}",
                summary_version="product-attributes-v2", summary_image_used=True)) + "\n" for i, item in enumerate(items)))
            mapping.write_text('{"1": "0"}')
            argv = ["embed", "--catalog", str(catalog), "--summary_path", str(summaries),
                    "--output_dir", str(root / "out"), "--model", str(model),
                    "--embedding_mode", "text", "--verified_product_map", str(mapping), "--shard_size", "1"]
            with patch("sys.argv", argv), patch.object(catalog_embed.requests, "get") as get, \
                    patch.object(catalog_embed, "gguf_embed", return_value=np.array([[1, 2]], dtype=np.float32)) as embed, \
                    patch.object(catalog_embed, "load_image") as load:
                get.return_value.json.return_value = {}
                catalog_embed.main()
            load.assert_not_called()
            self.assertEqual(embed.call_count, 1)
            shards = sorted((root / "out").glob("*.npz"))
            self.assertEqual(len(shards), 2)
            with np.load(shards[1]) as shard:
                self.assertEqual(shard["canonical_summary_item_id"].tolist(), [0])
                self.assertEqual(shard["image_used"].tolist(), [False])
                self.assertEqual(shard["summary_image_used"].tolist(), [True])
                np.testing.assert_array_equal(shard["embedding"], [[1, 2]])


if __name__ == "__main__":
    unittest.main()
