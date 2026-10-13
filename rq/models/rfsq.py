"""Robust Residual Finite Scalar Quantization (RFSQ).

Implements arXiv:2508.15860. The implicit codebook is the Cartesian product
of fixed scalar levels; there is no learned embedding table or nearest-neighbor
lookup. Later residual stages can use learned scale or invertible LayerNorm
conditioning to counter residual magnitude decay.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLPLayers


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels: Sequence[int]):
        super().__init__()
        if not levels or any(level < 2 for level in levels):
            raise ValueError("Every FSQ dimension must have at least two levels")
        self.register_buffer("levels", torch.tensor(list(levels), dtype=torch.long))
        strides = [1]
        for level in levels[:-1]:
            strides.append(strides[-1] * int(level))
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.long))

    @property
    def codebook_size(self) -> int:
        return int(torch.prod(self.levels).item())

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[-1] != len(self.levels):
            raise ValueError(
                f"Expected latent dim {len(self.levels)}, got {values.shape[-1]}"
            )
        levels = self.levels.to(device=values.device)
        # No input clipping: callers must ensure the range for finite code IDs.
        scalar_indices = torch.round((values + 1.0) * (levels - 1) / 2.0).long()
        quantized = scalar_indices.to(values.dtype) * 2.0 / (levels - 1) - 1.0
        quantized_ste = values + (quantized - values).detach()
        code_indices = (scalar_indices * self.strides.to(values.device)).sum(dim=-1)
        return quantized_ste, code_indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        scalar_indices = []
        remainder = indices.long()
        for level in self.levels:
            scalar_indices.append(remainder.remainder(int(level)))
            remainder = torch.div(remainder, int(level), rounding_mode="floor")
        scalar = torch.stack(scalar_indices, dim=-1)
        levels = self.levels.to(indices.device)
        return scalar.to(torch.float32) * 2.0 / (levels - 1) - 1.0


class RobustResidualFSQ(nn.Module):
    def __init__(
        self,
        levels_per_stage: Sequence[Sequence[int]],
        conditioning: str = "scale",
        eps: float = 1e-5,
    ):
        super().__init__()
        if not levels_per_stage:
            raise ValueError("levels_per_stage cannot be empty")
        latent_dims = {len(levels) for levels in levels_per_stage}
        if len(latent_dims) != 1:
            raise ValueError("All RFSQ stages must use the same scalar dimension")
        if conditioning not in {"none", "scale", "layernorm"}:
            raise ValueError("conditioning must be one of: none, scale, layernorm")
        self.levels_per_stage = [list(levels) for levels in levels_per_stage]
        self.conditioning = conditioning
        self.eps = eps
        self.quantizers = nn.ModuleList(
            [FiniteScalarQuantizer(levels) for levels in levels_per_stage]
        )
        # softplus(0.5413) is approximately 1.0. Stage zero is intentionally
        # unconditioned to match Algorithm 1 in the paper.
        self.raw_scales = nn.Parameter(torch.full((len(self.quantizers),), 0.54132485))

    @property
    def codebook_sizes(self) -> List[int]:
        return [quantizer.codebook_size for quantizer in self.quantizers]

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = values
        total = torch.zeros_like(values)
        all_indices = []

        for stage, quantizer in enumerate(self.quantizers):
            if self.conditioning == "scale" and stage > 0:
                scale = F.softplus(self.raw_scales[stage]) + self.eps
                conditioned = residual * scale
                quantized_conditioned, indices = quantizer(conditioned)
                contribution = quantized_conditioned / scale
            elif self.conditioning == "layernorm" and stage > 0:
                mean = residual.mean(dim=-1, keepdim=True)
                std = torch.sqrt(
                    residual.var(dim=-1, keepdim=True, unbiased=False) + self.eps
                )
                conditioned = (residual - mean) / std
                quantized_conditioned, indices = quantizer(conditioned)
                contribution = quantized_conditioned * std + mean
            else:
                contribution, indices = quantizer(residual)

            residual = residual - contribution
            total = total + contribution
            all_indices.append(indices)

        return total, torch.stack(all_indices, dim=-1)


class RFSQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        levels_per_stage: Sequence[Sequence[int]],
        layers: Sequence[int],
        conditioning: str = "scale",
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        latent_activation: str = "tanh",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.levels_per_stage = [list(levels) for levels in levels_per_stage]
        self.layers = list(layers)
        self.conditioning = conditioning
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        if latent_activation not in {"tanh", "none"}:
            raise ValueError("latent_activation must be tanh or none")
        self.latent_activation = latent_activation
        latent_dim = len(self.levels_per_stage[0])
        encoder_dims = [in_dim] + self.layers + [latent_dim]
        self.encoder = MLPLayers(layers=encoder_dims, dropout=dropout_prob, bn=bn)
        self.quantizer = RobustResidualFSQ(
            self.levels_per_stage, conditioning=conditioning
        )
        self.decoder = MLPLayers(layers=encoder_dims[::-1], dropout=dropout_prob, bn=bn)

    def forward(self, values: torch.Tensor):
        encoded = self.encoder(values)
        if self.latent_activation == "tanh":
            encoded = torch.tanh(encoded)
        quantized, indices = self.quantizer(encoded)
        reconstructed = self.decoder(quantized)
        auxiliary_loss = reconstructed.new_zeros(())
        return reconstructed, auxiliary_loss, indices

    @torch.no_grad()
    def get_indices(self, values: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        del use_sk
        encoded = self.encoder(values)
        if self.latent_activation == "tanh":
            encoded = torch.tanh(encoded)
        _, indices = self.quantizer(encoded)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        del quant_loss
        if self.loss_type == "mse":
            reconstruction = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            reconstruction = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return reconstruction, reconstruction
