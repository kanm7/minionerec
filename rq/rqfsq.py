#!/usr/bin/env python3
"""Train an RFSQ tokenizer on item embeddings."""

import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .datasets import EmbDataset
    from .models.rfsq import RFSQVAE
    from .trainer import Trainer
except ImportError:  # direct execution: python rq/rqfsq.py
    from datasets import EmbDataset
    from models.rfsq import RFSQVAE
    from trainer import Trainer


def parse_stage_levels(value: str):
    stages = []
    for stage in value.split(";"):
        levels = [int(level) for level in stage.split(",") if level.strip()]
        if not levels:
            raise argparse.ArgumentTypeError("Empty stage in --stage_levels")
        stages.append(levels)
    if len({len(levels) for levels in stages}) != 1:
        raise argparse.ArgumentTypeError(
            "All stages must have the same number of scalar dimensions"
        )
    return stages


def parse_args():
    parser = argparse.ArgumentParser(description="Train Robust Residual FSQ")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument(
        "--stage_levels", type=parse_stage_levels,
        default="4,4,4,2,2,2,2,2;4,4,4,2,2,2,2,2;4,4,4,2,2,2,2,2"
    )
    parser.add_argument(
        "--conditioning", choices=["none", "scale", "layernorm"], default="scale"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64]
    )
    parser.add_argument("--loss_type", choices=["mse", "l1"], default="mse")
    parser.add_argument("--standardize", dest="standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.add_argument("--latent_activation", choices=["tanh", "none"], default="tanh")
    parser.add_argument("--scalar_bound", choices=["clamp", "legacy"], default="clamp",
                        help="Clamp scalar level IDs before packing; legacy reproduces old unbounded checkpoints")
    parser.add_argument("--quantizer_input_clamp", action="store_true", help="Clamp each stage input before rounding and STE")
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=50)
    parser.add_argument("--learner", default="AdamW")
    parser.add_argument("--lr_scheduler_type", choices=["constant", "linear"], default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--metrics_item_file", help="Item JSON/JSONL for implicit-cell category purity")
    parser.add_argument("--metrics_interaction_file", help="Interaction JSONL; heat uses train records only")
    parser.add_argument("--metrics_category_field", default="categories")
    parser.add_argument("--metrics_category_level", choices=("leaf", "root"), default="leaf")
    parser.add_argument("--metrics_hot_cluster_percent", type=float, default=1.)
    parser.add_argument("--metrics_hot_cluster_multiplier", type=float, default=5.)
    return parser.parse_args()


def main():
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args = parse_args()
    # argparse does not apply type conversion to a string default.
    if isinstance(args.stage_levels, str):
        args.stage_levels = parse_stage_levels(args.stage_levels)
    logging.basicConfig(level=logging.INFO)
    data = EmbDataset(args.data_path, standardize=args.standardize)
    model = RFSQVAE(
        in_dim=data.dim,
        levels_per_stage=args.stage_levels,
        layers=args.layers,
        conditioning=args.conditioning,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        latent_activation=args.latent_activation,
        scalar_bound=args.scalar_bound == "clamp",
        input_clamp=args.quantizer_input_clamp,
    )
    print(model)
    print("Fixed implicit stage capacities (no learned codebook):", model.quantizer.codebook_sizes)
    loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(args, model, len(loader))
    best_loss, best_collision_rate = trainer.fit(loader)
    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)


if __name__ == "__main__":
    main()
