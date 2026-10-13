#!/usr/bin/env python3
"""Continual pre-training on SID sequences with prefix-aware ranking.

The causal objective learns all transitions in the flattened SID sequence.
The auxiliary objective teacher-forces complete valid SIDs and ranks the true
next item above hard/medium/easy negatives sampled from the catalog SID trie.
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fire
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SID_PATTERN = re.compile(r"<[^<>]+>")


def split_sid(sid: str) -> Tuple[str, ...]:
    tokens = tuple(SID_PATTERN.findall(str(sid)))
    if not tokens or "".join(tokens) != str(sid).strip():
        raise ValueError(f"Invalid SID string: {sid!r}")
    return tokens


def load_sid_catalog(index_file: str) -> List[Tuple[str, ...]]:
    with open(index_file, "r", encoding="utf-8") as stream:
        indices = json.load(stream)
    catalog = [tuple(tokens) for tokens in indices.values()]
    if not catalog:
        raise ValueError("SID catalog is empty")
    depths = {len(tokens) for tokens in catalog}
    if len(depths) != 1:
        raise ValueError(
            "CPT requires a fixed SID depth; found depths " + str(sorted(depths))
        )
    unique_catalog = list(dict.fromkeys(catalog))
    if len(unique_catalog) != len(catalog):
        print(
            "Warning: SID catalog has "
            f"{len(catalog) - len(unique_catalog)} duplicate rows. Prefix ranking "
            "operates on unique SID strings; item-level ambiguity remains."
        )
    return unique_catalog


class PrefixNegativeSampler:
    def __init__(self, catalog: Sequence[Tuple[str, ...]], seed: int = 42):
        self.catalog = list(catalog)
        self.depth = len(self.catalog[0])
        self.rng = random.Random(seed)
        self.prefix_maps: Dict[int, Dict[Tuple[str, ...], List[Tuple[str, ...]]]] = {}
        for length in range(1, self.depth):
            groups: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = {}
            for sid in self.catalog:
                groups.setdefault(sid[:length], []).append(sid)
            self.prefix_maps[length] = groups

    def _choose(self, pool, positive, excluded, count):
        candidates = [sid for sid in pool if sid != positive and sid not in excluded]
        self.rng.shuffle(candidates)
        return candidates[:count]

    def sample(
        self,
        positive: Tuple[str, ...],
        hard_count: int,
        medium_count: int,
        easy_count: int,
    ) -> Tuple[List[Tuple[str, ...]], List[str]]:
        selected: List[Tuple[str, ...]] = []
        labels: List[str] = []

        hard_prefix = max(1, self.depth - 1)
        hard_pool = self.prefix_maps.get(hard_prefix, {}).get(
            positive[:hard_prefix], []
        )
        hard = self._choose(hard_pool, positive, set(selected), hard_count)
        selected.extend(hard)
        labels.extend(["hard"] * len(hard))

        medium_prefix = max(1, self.depth - 2)
        medium_pool = self.prefix_maps.get(medium_prefix, {}).get(
            positive[:medium_prefix], []
        )
        medium_pool = [
            sid for sid in medium_pool if sid[:hard_prefix] != positive[:hard_prefix]
        ]
        medium = self._choose(medium_pool, positive, set(selected), medium_count)
        selected.extend(medium)
        labels.extend(["medium"] * len(medium))

        easy = []
        excluded = set(selected)
        attempts = 0
        max_attempts = max(100, easy_count * 20)
        while len(easy) < easy_count and attempts < max_attempts:
            candidate = self.rng.choice(self.catalog)
            attempts += 1
            if (
                candidate[0] != positive[0]
                and candidate not in excluded
                and candidate not in easy
            ):
                easy.append(candidate)
        if len(easy) < easy_count:
            easy_pool = [sid for sid in self.catalog if sid[0] != positive[0]]
            easy.extend(
                self._choose(
                    easy_pool, positive, excluded | set(easy), easy_count - len(easy)
                )
            )
        selected.extend(easy)
        labels.extend(["easy"] * len(easy))
        return selected, labels


class SIDCPTDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer,
        catalog: Sequence[Tuple[str, ...]],
        max_len: int,
        sample: int,
        seed: int,
        hard_negatives: int,
        medium_negatives: int,
        easy_negatives: int,
    ):
        frame = pd.read_csv(csv_file)
        if sample > 0 and sample < len(frame):
            frame = frame.sample(sample, random_state=seed)
        self.rows = frame.to_dict("records")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sampler = PrefixNegativeSampler(catalog, seed)
        self.hard_negatives = hard_negatives
        self.medium_negatives = medium_negatives
        self.easy_negatives = easy_negatives

    def __len__(self):
        return len(self.rows)

    def token_ids(self, tokens: Sequence[str]) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(list(tokens))
        if isinstance(ids, int):
            ids = [ids]
        if any(value == self.tokenizer.unk_token_id for value in ids):
            raise ValueError("A SID token was not registered in the tokenizer")
        return list(ids)

    def __getitem__(self, index):
        row = self.rows[index]
        history_raw = ast.literal_eval(row["history_item_sid"])
        history = [split_sid(sid) for sid in history_raw]
        positive = split_sid(row["item_sid"])
        negatives, difficulties = self.sampler.sample(
            positive,
            self.hard_negatives,
            self.medium_negatives,
            self.easy_negatives,
        )

        flat_history = [token for sid in history for token in sid]
        context_ids = self.token_ids(flat_history)
        if self.tokenizer.bos_token_id is not None:
            context_ids = [self.tokenizer.bos_token_id] + context_ids
        elif not context_ids:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer needs a BOS or EOS token for an empty history"
                )
            context_ids = [self.tokenizer.eos_token_id]
        positive_ids = self.token_ids(positive)
        suffix = (
            [self.tokenizer.eos_token_id]
            if self.tokenizer.eos_token_id is not None
            else []
        )
        full_ids = (context_ids + positive_ids + suffix)[-self.max_len :]
        # Preserve enough recent context for every complete candidate.
        context_limit = max(1, self.max_len - len(positive_ids))
        rank_context = context_ids[-context_limit:]
        candidates = [positive_ids] + [self.token_ids(sid) for sid in negatives]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": list(full_ids),
            "rank_context_ids": rank_context,
            "rank_candidate_ids": candidates,
            "negative_difficulties": difficulties,
        }


class SIDCPTCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.pad_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _round_length(self, length):
        if not self.pad_to_multiple_of:
            return length
        return int(
            math.ceil(length / self.pad_to_multiple_of) * self.pad_to_multiple_of
        )

    def __call__(self, features):
        batch_size = len(features)
        sequence_length = self._round_length(
            max(len(row["input_ids"]) for row in features)
        )
        input_ids = torch.full(
            (batch_size, sequence_length), self.pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
        labels = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
        for row_index, row in enumerate(features):
            length = len(row["input_ids"])
            input_ids[row_index, :length] = torch.tensor(row["input_ids"])
            attention_mask[row_index, :length] = 1
            labels[row_index, :length] = torch.tensor(row["labels"])

        context_length = max(len(row["rank_context_ids"]) for row in features)
        max_candidates = max(len(row["rank_candidate_ids"]) for row in features)
        candidate_length = max(
            len(candidate)
            for row in features
            for candidate in row["rank_candidate_ids"]
        )
        context_ids = torch.full(
            (batch_size, context_length), self.pad_id, dtype=torch.long
        )
        context_mask = torch.zeros((batch_size, context_length), dtype=torch.bool)
        candidate_ids = torch.full(
            (batch_size, max_candidates, candidate_length),
            self.pad_id,
            dtype=torch.long,
        )
        candidate_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
        candidate_valid = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
        candidate_difficulty = torch.full(
            (batch_size, max_candidates), -1, dtype=torch.long
        )
        for row_index, row in enumerate(features):
            context = row["rank_context_ids"]
            context_ids[row_index, : len(context)] = torch.tensor(context)
            context_mask[row_index, : len(context)] = True
            for candidate_index, candidate in enumerate(row["rank_candidate_ids"]):
                candidate_ids[row_index, candidate_index, : len(candidate)] = (
                    torch.tensor(candidate)
                )
                candidate_mask[row_index, candidate_index, : len(candidate)] = True
                candidate_valid[row_index, candidate_index] = True
            difficulty_ids = {"hard": 1, "medium": 2, "easy": 3}
            candidate_difficulty[row_index, 0] = 0
            for candidate_index, difficulty in enumerate(
                row["negative_difficulties"], start=1
            ):
                candidate_difficulty[row_index, candidate_index] = difficulty_ids[
                    difficulty
                ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rank_context_ids": context_ids,
            "rank_context_mask": context_mask,
            "rank_candidate_ids": candidate_ids,
            "rank_candidate_mask": candidate_mask,
            "rank_candidate_valid": candidate_valid,
            "rank_candidate_difficulty": candidate_difficulty,
        }


class PrefixRankingTrainer(transformers.Trainer):
    def __init__(self, *args, rank_loss_weight=0.2, rank_temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_loss_weight = rank_loss_weight
        self.rank_temperature = rank_temperature
        self._prefix_metric_totals = {
            "train": {
                "batches": 0,
                "cpt_loss": 0.0,
                "rank_loss": 0.0,
                "rank_correct": 0,
                "rows": 0,
                "hard_inversions": 0,
                "hard_pairs": 0,
            },
            "eval": {
                "batches": 0,
                "cpt_loss": 0.0,
                "rank_loss": 0.0,
                "rank_correct": 0,
                "rows": 0,
                "hard_inversions": 0,
                "hard_pairs": 0,
            },
        }

    def _candidate_scores(
        self,
        model,
        context_ids,
        context_mask,
        candidate_ids,
        candidate_mask,
        candidate_valid,
    ):
        sequences, metadata = [], []
        batch_size, candidate_count, _ = candidate_ids.shape
        for row in range(batch_size):
            context = context_ids[row][context_mask[row]].tolist()
            for candidate in range(candidate_count):
                if not candidate_valid[row, candidate]:
                    continue
                target = candidate_ids[row, candidate][
                    candidate_mask[row, candidate]
                ].tolist()
                sequences.append(context + target)
                metadata.append((row, candidate, len(context), len(target)))

        max_length = max(len(sequence) for sequence in sequences)
        processor = getattr(self, "processing_class", None)
        if processor is None:
            processor = self.tokenizer
        padded = context_ids.new_full(
            (len(sequences), max_length), processor.pad_token_id
        )
        mask = context_ids.new_zeros((len(sequences), max_length))
        for index, sequence in enumerate(sequences):
            padded[index, : len(sequence)] = torch.tensor(
                sequence, device=padded.device
            )
            mask[index, : len(sequence)] = 1
        logits = model(input_ids=padded, attention_mask=mask).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        scores = logits.new_full(
            (batch_size, candidate_count), -torch.inf, dtype=torch.float32
        )
        for index, (row, candidate, context_length, target_length) in enumerate(
            metadata
        ):
            positions = log_probs[
                index, context_length - 1 : context_length + target_length - 1
            ]
            targets = padded[index, context_length : context_length + target_length]
            scores[row, candidate] = positions.gather(-1, targets[:, None]).sum()
        return scores

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        context_ids = inputs.pop("rank_context_ids")
        context_mask = inputs.pop("rank_context_mask")
        candidate_ids = inputs.pop("rank_candidate_ids")
        candidate_mask = inputs.pop("rank_candidate_mask")
        candidate_valid = inputs.pop("rank_candidate_valid")
        candidate_difficulty = inputs.pop("rank_candidate_difficulty")
        outputs = model(**inputs)
        cpt_loss = outputs.loss
        scores = self._candidate_scores(
            model,
            context_ids,
            context_mask,
            candidate_ids,
            candidate_mask,
            candidate_valid,
        )
        targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        rank_loss = F.cross_entropy(scores / self.rank_temperature, targets)
        loss = cpt_loss + self.rank_loss_weight * rank_loss

        with torch.no_grad():
            mode = "train" if model.training else "eval"
            totals = self._prefix_metric_totals[mode]
            negative_scores = scores[:, 1:]
            if negative_scores.shape[1]:
                row_correct = scores[:, 0] > negative_scores.max(dim=1).values
            else:
                row_correct = torch.ones(
                    scores.shape[0], dtype=torch.bool, device=scores.device
                )
            hard_mask = candidate_difficulty.eq(1)
            hard_inversions = hard_mask & scores.ge(scores[:, :1])
            totals["batches"] += 1
            totals["cpt_loss"] += float(cpt_loss.detach())
            totals["rank_loss"] += float(rank_loss.detach())
            totals["rank_correct"] += int(row_correct.sum())
            totals["rows"] += scores.shape[0]
            totals["hard_inversions"] += int(hard_inversions.sum())
            totals["hard_pairs"] += int(hard_mask.sum())
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        mode = "eval" if any(str(key).startswith("eval_") for key in logs) else "train"
        totals = self._prefix_metric_totals[mode]
        prefix = "eval_" if mode == "eval" else ""
        if totals["batches"]:
            logs[f"{prefix}cpt_loss"] = totals["cpt_loss"] / totals["batches"]
            logs[f"{prefix}prefix_rank_loss"] = totals["rank_loss"] / totals["batches"]
            logs[f"{prefix}prefix_rank_accuracy"] = totals["rank_correct"] / max(
                totals["rows"], 1
            )
            logs[f"{prefix}hard_inversion_rate"] = totals["hard_inversions"] / max(
                totals["hard_pairs"], 1
            )
            for key in totals:
                totals[key] = 0.0 if "loss" in key else 0
        super().log(logs, start_time=start_time)


def train(
    base_model: str,
    train_file: str,
    eval_file: str,
    sid_index_path: str,
    output_dir: str,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    cutoff_len: int = 512,
    sample: int = -1,
    seed: int = 42,
    hard_negatives: int = 1,
    medium_negatives: int = 1,
    easy_negatives: int = 1,
    rank_loss_weight: float = 0.2,
    rank_temperature: float = 1.0,
    bf16: bool = True,
    resume_from_checkpoint: Optional[str] = None,
):
    random.seed(seed)
    torch.manual_seed(seed)
    catalog = load_sid_catalog(sid_index_path)
    sid_tokens = sorted({token for sid in catalog for token in sid})
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if bf16 and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    added = tokenizer.add_tokens(sid_tokens)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    print(f"Registered {len(sid_tokens)} SID tokens ({added} newly added)")

    dataset_args = dict(
        tokenizer=tokenizer,
        catalog=catalog,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        hard_negatives=hard_negatives,
        medium_negatives=medium_negatives,
        easy_negatives=easy_negatives,
    )
    train_dataset = SIDCPTDataset(train_file, **dataset_args)
    eval_dataset = SIDCPTDataset(eval_file, **dataset_args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation = max(1, batch_size // micro_batch_size // world_size)
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=bf16 and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )
    trainer = PrefixRankingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SIDCPTCollator(tokenizer),
        processing_class=tokenizer,
        rank_loss_weight=rank_loss_weight,
        rank_temperature=rank_temperature,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = Path(output_dir) / "final_checkpoint"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    fire.Fire(train)
