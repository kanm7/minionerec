#!/usr/bin/env python3
"""Inject collaborative co-occurrence signal into frozen semantic embeddings.

Positive pairs are counted once per user, which avoids repeated sliding-window
rows from inflating support. A shared residual MLP is trained with InfoNCE;
the original embedding table is never updated.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


def read_longest_user_sequences(paths: Sequence[str]) -> Dict[str, List[int]]:
    """Reconstruct longest sequences from .inter files or converted CSVs."""
    longest: Dict[str, List[int]] = {}
    for path in paths:
        if str(path).lower().endswith(".csv"):
            import pandas as pd

            frame = pd.read_csv(path)
            for row in frame.to_dict("records"):
                user = str(row["user_id"])
                history = ast.literal_eval(str(row["history_item_id"]))
                sequence = [int(value) for value in history] + [int(row["item_id"])]
                if user not in longest or len(sequence) > len(longest[user]):
                    longest[user] = sequence
            continue
        with open(path, "r", encoding="utf-8") as stream:
            next(stream, None)
            for line in stream:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 3:
                    continue
                user, history, target = fields
                sequence = [int(value) for value in history.split() if value]
                sequence.append(int(target))
                if user not in longest or len(sequence) > len(longest[user]):
                    longest[user] = sequence
    return longest


def build_pair_statistics(
    user_sequences: Dict[str, List[int]], window_size: int
) -> Tuple[Counter, Counter]:
    """Count undirected pair support and item popularity by distinct users."""
    pair_users: Counter = Counter()
    item_users: Counter = Counter()
    for sequence in user_sequences.values():
        unique_items = set(sequence)
        item_users.update(unique_items)
        seen_pairs = set()
        for left in range(len(sequence)):
            for right in range(left + 1, min(len(sequence), left + window_size + 1)):
                a, b = sequence[left], sequence[right]
                if a != b:
                    seen_pairs.add((min(a, b), max(a, b)))
        pair_users.update(seen_pairs)
    return pair_users, item_users


def select_positive_pairs(
    pair_users: Counter,
    item_users: Counter,
    min_user_support: int,
    popularity_alpha: float,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    pairs, weights = [], []
    for (left, right), support in pair_users.items():
        if support < min_user_support:
            continue
        denominator = (item_users[left] * item_users[right]) ** popularity_alpha
        score = support / max(denominator, 1.0)
        pairs.append((left, right))
        weights.append(score)
    if not pairs:
        raise ValueError("No positive pairs survived min_user_support")
    return pairs, weights


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[int, int]]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        return self.pairs[index]


class SharedResidualProjector(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, residual_weight: float
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim, bias=False)
        )
        self.residual_weight = residual_weight

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.skip(values) + self.residual_weight * self.net(values)
        return F.normalize(projected, dim=-1)


def build_positive_neighbors(pairs: Sequence[Tuple[int, int]]) -> Dict[int, Set[int]]:
    neighbors: Dict[int, Set[int]] = {}
    for left, right in pairs:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    return neighbors


def sample_negatives(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    num_items: int,
    num_negatives: int,
    device: torch.device,
    positive_neighbors: Dict[int, Set[int]],
) -> torch.Tensor:
    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")
    negatives = torch.randint(
        0, num_items, (len(anchors), num_negatives), device=device
    )
    for anchor in anchors.unique().tolist():
        rows = anchors.eq(anchor)
        forbidden = positive_neighbors.get(anchor, set()) | {anchor}
        forbidden.update(positives[rows].tolist())
        if len(forbidden) >= num_items:
            raise ValueError(f"No valid negative items remain for anchor {anchor}")
        forbidden_ids = torch.tensor(sorted(forbidden), device=device)
        selected = negatives[rows]
        if len(forbidden) > num_items // 2:
            allowed = torch.arange(num_items, device=device)
            allowed = allowed[~torch.isin(allowed, forbidden_ids)]
            selected = allowed[
                torch.randint(len(allowed), selected.shape, device=device)
            ]
        else:
            invalid = torch.isin(selected, forbidden_ids)
            while invalid.any():
                selected[invalid] = torch.randint(
                    0, num_items, (int(invalid.sum()),), device=device
                )
                invalid = torch.isin(selected, forbidden_ids)
        negatives[rows] = selected
    return negatives


def directional_info_nce_loss(
    model: SharedResidualProjector,
    embedding_table: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    num_negatives: int,
    temperature: float,
    positive_neighbors: Dict[int, Set[int]],
) -> torch.Tensor:
    negatives = sample_negatives(
        anchors,
        positives,
        len(embedding_table),
        num_negatives,
        embedding_table.device,
        positive_neighbors,
    )
    anchor_z = model(embedding_table[anchors])
    positive_z = model(embedding_table[positives])
    negative_z = model(embedding_table[negatives])
    positive_logits = (anchor_z * positive_z).sum(dim=-1, keepdim=True)
    negative_logits = torch.einsum("bd,bnd->bn", anchor_z, negative_z)
    logits = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    labels = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
    return F.cross_entropy(logits, labels)


def info_nce_loss(
    model: SharedResidualProjector,
    embedding_table: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    num_negatives: int,
    temperature: float,
    positive_neighbors: Dict[int, Set[int]],
) -> torch.Tensor:
    forward = directional_info_nce_loss(
        model,
        embedding_table,
        anchors,
        positives,
        num_negatives,
        temperature,
        positive_neighbors,
    )
    backward = directional_info_nce_loss(
        model,
        embedding_table,
        positives,
        anchors,
        num_negatives,
        temperature,
        positive_neighbors,
    )
    return (forward + backward) / 2


@torch.inference_mode()
def map_catalog(
    model: SharedResidualProjector,
    embedding_table: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(embedding_table), batch_size):
        chunks.append(model(embedding_table[start : start + batch_size]).cpu())
    return torch.cat(chunks).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train collaborative residual projection with InfoNCE"
    )
    parser.add_argument("--embedding_path", required=True)
    parser.add_argument("--train_inter", nargs="+", required=True)
    parser.add_argument("--output_embedding_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--min_user_support", type=int, default=3)
    parser.add_argument("--popularity_alpha", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--output_dim", type=int, default=0)
    parser.add_argument("--residual_weight", type=float, default=0.2)
    parser.add_argument("--num_negatives", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    embeddings_np = np.load(args.embedding_path).astype(np.float32)
    embeddings = torch.from_numpy(embeddings_np).to(device)
    sequences = read_longest_user_sequences(args.train_inter)
    pair_users, item_users = build_pair_statistics(sequences, args.window_size)
    pairs, weights = select_positive_pairs(
        pair_users, item_users, args.min_user_support, args.popularity_alpha
    )
    if max(max(pair) for pair in pairs) >= len(embeddings):
        raise ValueError("Interaction item id is outside the embedding table")
    positive_neighbors = build_positive_neighbors(pairs)

    dataset = PairDataset(pairs)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    output_dim = args.output_dim or embeddings.shape[1]
    model = SharedResidualProjector(
        embeddings.shape[1], args.hidden_dim, output_dim, args.residual_weight
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for anchors, positives in tqdm(loader, desc=f"InfoNCE epoch {epoch + 1}"):
            anchors, positives = anchors.to(device), positives.to(device)
            loss = info_nce_loss(
                model,
                embeddings,
                anchors,
                positives,
                args.num_negatives,
                args.temperature,
                positive_neighbors,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(anchors)
        print(f"epoch={epoch + 1} loss={total / len(dataset):.6f}")

    output_path = Path(args.output_embedding_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, map_catalog(model, embeddings, args.batch_size))

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": embeddings.shape[1],
            "hidden_dim": args.hidden_dim,
            "output_dim": output_dim,
            "residual_weight": args.residual_weight,
            "args": vars(args),
        },
        checkpoint_path,
    )

    stats = {
        "users": len(sequences),
        "items": len(embeddings),
        "raw_pairs": len(pair_users),
        "positive_pairs": len(pairs),
        "window_size": args.window_size,
        "min_user_support": args.min_user_support,
        "popularity_alpha": args.popularity_alpha,
    }
    stats_path = (
        Path(args.stats_path)
        if args.stats_path
        else output_path.with_suffix(".stats.json")
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
