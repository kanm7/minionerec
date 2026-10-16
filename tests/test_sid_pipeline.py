import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import numpy as np
import torch

from cpt import (
    PrefixNegativeSampler,
    PrefixRankingTrainer,
    SIDCPTCollator,
    SIDCPTDataset,
    split_sid,
)
from rq.collab.train_projection import (
    SharedResidualProjector,
    build_pair_statistics,
    build_positive_neighbors,
    info_nce_loss,
    sample_negatives,
    select_positive_pairs,
)
from rq.generate_rfsq_indices import repair_last_layer_collisions
from rq.finetune_rfsq_collisions import collision_pairs
from rq.models.rfsq import FiniteScalarQuantizer, RobustResidualFSQ
from rq.sid_metrics import normalized_entropy
from rq.summary.generate_structured_summaries import (
    build_product_text,
    canonical_summary_text,
    extract_json_object,
)
from rq.text2emb.amazon_summary_vl2emb import build_inputs, gguf_embed


class SummaryTests(unittest.TestCase):
    def test_joint_comparison_keeps_image(self):
        with patch("rq.text2emb.amazon_summary_vl2emb.load_image", return_value=Image.new("RGB", (2, 2))):
            items = {"1": {"title": "Pen", "images": ["product.png"]}}
            self.assertEqual(build_product_text(items["1"]), "title: Pen")
            prepared = list(build_inputs(items, {"1": "product_type: pen"}, 1, 1.0, True, "joint"))
            self.assertEqual(prepared[0][1]["text"], "product_type: pen")
            self.assertIsInstance(prepared[0][1]["image"], Image.Image)

    def test_gguf_request_contains_summary_and_image_marker(self):
        with patch("rq.text2emb.amazon_summary_vl2emb.requests.post") as post:
            post.return_value.json.return_value = {"embedding": [0.5, 0.5]}
            vector = gguf_embed("http://127.0.0.1:8080", [
                {"text": "product_type: pen", "image": Image.new("RGB", (2, 2))}
            ], "<image-marker>")
        content = post.call_args.kwargs["json"]["content"]
        self.assertEqual(content["prompt_string"], "product_type: pen\n<image-marker>")
        self.assertTrue(content["multimodal_data"][0])
        self.assertEqual(vector.shape, (1, 2))

    def test_extract_and_canonicalize_summary(self):
        summary = extract_json_object(
            '```json\n{"product_type":"pen","usage_scenario":"office, study"}\n```'
        )
        self.assertEqual(summary["usage_scenario"], ["office", "study"])
        text = canonical_summary_text(summary)
        self.assertIn("product_type: pen", text)
        self.assertIn("usage_scenario: office, study", text)


class CollaborativePairTests(unittest.TestCase):
    def test_pair_support_is_counted_once_per_user(self):
        sequences = {"u1": [0, 1, 0, 1], "u2": [0, 1, 2]}
        pair_users, item_users = build_pair_statistics(sequences, window_size=2)
        self.assertEqual(pair_users[(0, 1)], 2)
        self.assertEqual(item_users[0], 2)
        pairs, weights = select_positive_pairs(pair_users, item_users, 2, 0.5)
        self.assertEqual(pairs, [(0, 1)])
        self.assertGreater(weights[0], 0)

    def test_negative_sampling_excludes_known_positives_in_both_directions(self):
        neighbors = build_positive_neighbors([(0, 1), (0, 2)])
        self.assertEqual(neighbors[0], {1, 2})
        self.assertEqual(neighbors[1], {0})
        negatives = sample_negatives(
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            4,
            100,
            torch.device("cpu"),
            neighbors,
        )
        self.assertEqual(set(negatives[0].tolist()), {3})
        self.assertTrue(set(negatives[1].tolist()).isdisjoint({0, 1}))
        with self.assertRaisesRegex(ValueError, "No valid negative"):
            sample_negatives(
                torch.tensor([0]),
                torch.tensor([1]),
                3,
                1,
                torch.device("cpu"),
                neighbors,
            )

    def test_symmetric_loss_is_independent_of_pair_orientation(self):
        embeddings = torch.randn(3, 4)
        model = SharedResidualProjector(4, 8, 4, 0.2)
        neighbors = build_positive_neighbors([(0, 1)])
        forward = info_nce_loss(
            model,
            embeddings,
            torch.tensor([0]),
            torch.tensor([1]),
            2,
            0.07,
            neighbors,
        )
        backward = info_nce_loss(
            model,
            embeddings,
            torch.tensor([1]),
            torch.tensor([0]),
            2,
            0.07,
            neighbors,
        )
        self.assertTrue(torch.allclose(forward, backward))
        forward.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


class RFSQTests(unittest.TestCase):
    def test_collision_pairs_skip_identical_input_vectors(self):
        indices = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 4]])
        embeddings = np.array([[1.0], [1.0], [2.0], [3.0]])
        self.assertEqual(collision_pairs(indices, embeddings), [(0, 2), (1, 2)])

    def test_fsq_indices_and_straight_through_gradient(self):
        quantizer = FiniteScalarQuantizer([4, 3])
        values = torch.tensor([[0.2, -0.7], [1.5, -2.0]], requires_grad=True)
        quantized, indices = quantizer(values)
        self.assertEqual(tuple(quantized.shape), (2, 2))
        self.assertEqual(indices[0].item(), 2)
        self.assertTrue(torch.allclose(quantized[1], torch.tensor([1.0, -1.0])))
        self.assertTrue(bool(((indices >= 0) & (indices < quantizer.codebook_size)).all()))
        self.assertTrue(torch.allclose(quantized, quantizer.indices_to_codes(indices)))
        quantized.sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))

    def test_legacy_unbounded_fsq_remains_explicitly_available(self):
        quantizer = FiniteScalarQuantizer([4, 3], scalar_bound=False)
        values = torch.tensor([[1.5, -2.0]], requires_grad=True)
        quantized, _ = quantizer(values)
        self.assertTrue(torch.allclose(quantized[0], torch.tensor([5.0 / 3.0, -2.0])))
        quantized.sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))

    def test_residual_fsq_returns_one_code_per_stage(self):
        model = RobustResidualFSQ([[4, 4], [4, 4], [4, 4]], conditioning="scale")
        quantized, indices = model(torch.randn(5, 2))
        self.assertEqual(tuple(quantized.shape), (5, 2))
        self.assertEqual(tuple(indices.shape), (5, 3))
        self.assertEqual(model.codebook_sizes, [16, 16, 16])

    def test_collision_repair_preserves_depth_and_prefix(self):
        output = {
            "0": ["<a_0>", "<b_1>", "<c_2>"],
            "1": ["<a_0>", "<b_1>", "<c_2>"],
            "2": ["<a_0>", "<b_1>", "<c_3>"],
        }
        repaired = repair_last_layer_collisions(output, last_layer_capacity=4)
        self.assertEqual(repaired, 1)
        self.assertEqual(len({tuple(value) for value in output.values()}), 3)
        self.assertTrue(
            all(value[:2] == ["<a_0>", "<b_1>"] for value in output.values())
        )


class PrefixSamplerTests(unittest.TestCase):
    def test_difficulty_has_expected_prefix_overlap(self):
        catalog = [
            ("<a_0>", "<b_0>", "<c_0>"),
            ("<a_0>", "<b_0>", "<c_1>"),
            ("<a_0>", "<b_1>", "<c_0>"),
            ("<a_1>", "<b_0>", "<c_0>"),
        ]
        positive = catalog[0]
        negatives, labels = PrefixNegativeSampler(catalog, seed=1).sample(
            positive, 1, 1, 1
        )
        by_label = dict(zip(labels, negatives))
        self.assertEqual(by_label["hard"][:2], positive[:2])
        self.assertEqual(by_label["medium"][:1], positive[:1])
        self.assertNotEqual(by_label["medium"][:2], positive[:2])
        self.assertNotEqual(by_label["easy"][0], positive[0])

    def test_sid_parser_rejects_non_sid_text(self):
        self.assertEqual(split_sid("<a_1><b_2>"), ("<a_1>", "<b_2>"))
        with self.assertRaises(ValueError):
            split_sid("prefix<a_1>")

    def test_cpt_dataset_and_collator_shapes(self):
        catalog = [
            ("<a_0>", "<b_0>", "<c_0>"),
            ("<a_0>", "<b_0>", "<c_1>"),
            ("<a_0>", "<b_1>", "<c_0>"),
            ("<a_1>", "<b_0>", "<c_0>"),
        ]

        class DummyTokenizer:
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 2
            unk_token_id = 0

            def __init__(self):
                tokens = sorted({token for sid in catalog for token in sid})
                self.vocab = {token: index + 3 for index, token in enumerate(tokens)}

            def convert_tokens_to_ids(self, tokens):
                return [self.vocab.get(token, 0) for token in tokens]

        Path("tmp").mkdir(exist_ok=True)
        csv_path = Path("tmp") / "test_cpt_dataset.csv"
        try:
            csv_path.write_text(
                "user_id,history_item_sid,item_sid\n"
                "u1,\"['<a_0><b_1><c_0>']\",<a_0><b_0><c_0>\n",
                encoding="utf-8",
            )
            tokenizer = DummyTokenizer()
            dataset = SIDCPTDataset(
                str(csv_path), tokenizer, catalog, 32, -1, 1, 1, 1, 1
            )
            batch = SIDCPTCollator(tokenizer)([dataset[0]])
            self.assertEqual(tuple(batch["rank_candidate_ids"].shape), (1, 4, 3))
            self.assertEqual(int(batch["rank_candidate_valid"].sum()), 4)
            self.assertEqual(
                batch["rank_candidate_difficulty"].tolist(), [[0, 1, 2, 3]]
            )

            class NextTokenModel(torch.nn.Module):
                def forward(self, input_ids, attention_mask):
                    del attention_mask
                    logits = torch.zeros((*input_ids.shape, 32))
                    for position in range(input_ids.shape[1] - 1):
                        logits[:, position, input_ids[:, position + 1]] = 5.0
                    return type("Output", (), {"logits": logits})()

            trainer = object.__new__(PrefixRankingTrainer)
            trainer.processing_class = tokenizer
            scores = trainer._candidate_scores(
                NextTokenModel(),
                batch["rank_context_ids"],
                batch["rank_context_mask"],
                batch["rank_candidate_ids"],
                batch["rank_candidate_mask"],
                batch["rank_candidate_valid"],
            )
            self.assertEqual(tuple(scores.shape), (1, 4))
            self.assertTrue(torch.isfinite(scores).all())
        finally:
            csv_path.unlink(missing_ok=True)


class MetricTests(unittest.TestCase):
    def test_entropy_bounds(self):
        from collections import Counter

        self.assertAlmostEqual(normalized_entropy(Counter({"a": 2, "b": 2})), 1.0)
        self.assertGreaterEqual(normalized_entropy(Counter({"a": 3, "b": 1})), 0.0)


if __name__ == "__main__":
    unittest.main()
