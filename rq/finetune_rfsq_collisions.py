#!/usr/bin/env python3
"""Fine-tune an RFSQ checkpoint by separating items with identical SIDs."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .datasets import EmbDataset
    from .models.rfsq import RFSQVAE
except ImportError:
    from datasets import EmbDataset
    from models.rfsq import RFSQVAE


def collision_pairs(indices, embeddings):
    groups = defaultdict(list)
    for item_id, sid in enumerate(indices.tolist()):
        groups[tuple(sid)].append(item_id)
    pairs = []
    for group in groups.values():
        for left_pos, left in enumerate(group):
            for right in group[left_pos + 1 :]:
                # Deterministic quantization cannot distinguish identical inputs.
                if not np.array_equal(embeddings[left], embeddings[right]):
                    pairs.append((left, right))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--collision_weight", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--pair_batch_size", type=int, default=128)
    parser.add_argument("--mine_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Evaluations without collision or MSE improvement; 0 disables")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.pair_batch_size, args.mine_every, args.eval_every) < 1:
        parser.error("epochs, batch sizes, and intervals must be positive")
    if args.collision_weight < 0 or args.margin <= 0:
        parser.error("collision_weight must be nonnegative and margin must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    dataset = EmbDataset(
        train_args.data_path,
        standardize=getattr(train_args, "standardize", False),
    )
    embeddings = np.asarray(dataset.embeddings, dtype=np.float32)
    values = torch.from_numpy(embeddings).to(args.device)
    model = RFSQVAE(
        in_dim=dataset.dim,
        levels_per_stage=train_args.stage_levels,
        layers=train_args.layers,
        conditioning=train_args.conditioning,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        latent_activation=getattr(train_args, "latent_activation", "tanh"),
        scalar_bound=getattr(train_args, "scalar_bound", "legacy") == "clamp",
        input_clamp=getattr(train_args, "quantizer_input_clamp", False),
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_collision_rate = float("inf")
    history = []
    pair_indices = torch.empty((0, 2), dtype=torch.long, device=args.device)
    # Include the starting model so fine-tuning never selects a worse raw
    # collision rate simply because all evaluated updates were worse.
    model.eval()
    with torch.no_grad():
        initial_indices=model.get_indices(values)
        best_collision_rate=(len(values)-torch.unique(initial_indices,dim=0).shape[0])/len(values)
    torch.save(dict(args=train_args,state_dict=model.state_dict(),epoch=0,
                    best_collision_rate=best_collision_rate,fine_tune_args=vars(args)),
               output_dir/'best_collision_model.pth')
    print(json.dumps(dict(epoch=0,collision_rate=best_collision_rate,phase='starting_checkpoint')),flush=True)
    stale = 0
    with torch.no_grad():
        initial_reconstruction,_,_ = model(values)
        best_reconstruction = F.mse_loss(initial_reconstruction, values).item()

    for epoch in range(args.epochs):
        if epoch % args.mine_every == 0:
            model.eval()
            with torch.no_grad():
                indices = model.get_indices(values).cpu()
            pairs = collision_pairs(indices, embeddings)
            pair_indices = torch.tensor(pairs, dtype=torch.long, device=args.device).reshape(-1, 2)
            model.train()

        order = torch.randperm(len(values), device=args.device)
        for batch_indices in order.split(args.batch_size):
            batch = values[batch_indices]
            optimizer.zero_grad()
            reconstructed, quant_loss, _ = model(batch)
            loss, _ = model.compute_loss(reconstructed, quant_loss, xs=batch)
            if args.collision_weight and len(pair_indices):
                selected = pair_indices[
                    torch.randint(
                        len(pair_indices),
                        (min(args.pair_batch_size, len(pair_indices)),),
                        device=args.device,
                    )
                ]
                left = model.encoder(values[selected[:, 0]])
                right = model.encoder(values[selected[:, 1]])
                if model.latent_activation == "tanh":
                    left, right = torch.tanh(left), torch.tanh(right)
                distance = torch.linalg.vector_norm(left - right, dim=-1)
                loss = loss + args.collision_weight * F.relu(args.margin - distance).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            with torch.no_grad():
                reconstructed, _, indices = model(values)
                unique = torch.unique(indices, dim=0).shape[0]
                collision_rate = (len(values) - unique) / len(values)
                reconstruction_mse = F.mse_loss(reconstructed, values).item()
            record = {
                "epoch": epoch + 1,
                "collision_rate": collision_rate,
                "unique_sids": unique,
                "reconstruction_mse": reconstruction_mse,
                "mined_pairs": len(pair_indices),
            }
            history.append(record)
            improved = collision_rate < best_collision_rate or reconstruction_mse < best_reconstruction - 1e-5
            best_reconstruction = min(best_reconstruction, reconstruction_mse)
            stale = 0 if improved else stale + 1
            print(json.dumps(record), flush=True)
            if collision_rate < best_collision_rate:
                best_collision_rate = collision_rate
                torch.save(
                    {
                        "args": train_args,
                        "state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "best_collision_rate": collision_rate,
                        "fine_tune_args": vars(args),
                    },
                    output_dir / "best_collision_model.pth",
                )
            model.train()
            if args.early_stop_patience and stale >= args.early_stop_patience:
                print(json.dumps(dict(early_stop=True,epoch=epoch+1,stale_evaluations=stale)),flush=True)
                break

    (output_dir / "metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best collision rate: {best_collision_rate:.6f}")


if __name__ == "__main__":
    main()
