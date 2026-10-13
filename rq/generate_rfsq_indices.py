#!/usr/bin/env python3
"""Generate MiniOneRec-compatible SID JSON from an RFSQ checkpoint."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .datasets import EmbDataset
    from .models.rfsq import RFSQVAE
except ImportError:  # direct execution: python rq/generate_rfsq_indices.py
    from datasets import EmbDataset
    from models.rfsq import RFSQVAE


TOKEN_CODE = re.compile(r"^<([a-z])_(\d+)>$")


def repair_last_layer_collisions(output, last_layer_capacity):
    """Keep SID depth fixed by assigning unused last-layer codes per prefix.

    This is a post-quantization uniqueness constraint: repaired codes are valid
    tokens but are no longer the raw nearest scalar-grid assignment.
    """
    prefix_groups = defaultdict(list)
    for item_id, sid in output.items():
        prefix_groups[tuple(sid[:-1])].append((item_id, sid))
    repaired = 0
    for group in prefix_groups.values():
        if len(group) > last_layer_capacity:
            raise ValueError(
                "A prefix contains more items than the last-layer capacity; "
                "increase the final-stage FSQ codebook size"
            )
        used = set()
        duplicates = []
        for item_id, sid in group:
            match = TOKEN_CODE.match(sid[-1])
            if not match:
                raise ValueError(f"Invalid SID token: {sid[-1]}")
            code = int(match.group(2))
            if code in used:
                duplicates.append((item_id, sid, match.group(1)))
            else:
                used.add(code)
        available = iter(
            code for code in range(last_layer_capacity) if code not in used
        )
        for _, sid, prefix in duplicates:
            sid[-1] = f"<{prefix}_{next(available)}>"
            repaired += 1
    return repaired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--repair_collisions",
        action="store_true",
        help=(
            "reassign duplicate last-layer codes within each prefix to keep "
            "three-layer SIDs unique"
        ),
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    data = EmbDataset(
        train_args.data_path,
        standardize=getattr(train_args, "standardize", False),
    )
    model = RFSQVAE(
        in_dim=data.dim,
        levels_per_stage=train_args.stage_levels,
        layers=train_args.layers,
        conditioning=train_args.conditioning,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        latent_activation=getattr(train_args, "latent_activation", "tanh"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()

    prefixes = ["a", "b", "c", "d", "e", "f", "g", "h"]
    if len(train_args.stage_levels) > len(prefixes):
        raise ValueError("More RFSQ stages than available SID prefixes")
    output = {}
    offset = 0
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    with torch.inference_mode():
        for values in tqdm(loader, desc="Generating RFSQ SIDs"):
            indices = model.get_indices(values.to(args.device)).cpu().tolist()
            for row in indices:
                output[str(offset)] = [
                    f"<{prefixes[stage]}_{int(code)}>" for stage, code in enumerate(row)
                ]
                offset += 1

    repaired = 0
    if args.repair_collisions:
        repaired = repair_last_layer_collisions(
            output, model.quantizer.codebook_sizes[-1]
        )
        print(f"Repaired {repaired} colliding SIDs in the last layer")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")
    print(f"Saved {len(output)} SIDs to {output_path}")


if __name__ == "__main__":
    main()
