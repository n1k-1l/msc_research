"""
Configurable MLP.

A single class covers every architecture used in the project (for example
784-512-256-10, 784-1024-512-256-10, 3072-1024-512-256-10): the layer sizes are
passed as the `widths` list rather than defining a new class per architecture.

Each hidden layer has an associated dropout module. dropout_modules() exposes
them in forward order so callers can iterate over them and call set_probs().
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

from .dropout import build_dropout, PerNeuronDropout


class MLP(nn.Module):
    """
    Args:
        widths: full layer sizes including input and output, e.g.
                [784, 512, 256, 10]. Hidden layers are everything between.
        dropout_kind: "uniform" | "per_neuron" | "none".
        p: dropout probability (the p_base budget).
        activation: "relu" (default) or "gelu".
    """
    def __init__(
        self,
        widths: List[int],
        dropout_kind: str = "uniform",
        p: float = 0.5,
        activation: str = "relu",
    ):
        super().__init__()
        self.widths = list(widths)
        self.dropout_kind = dropout_kind
        self.p = p
        assert len(widths) >= 3, "need at least input, one hidden, output"

        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU}[activation.lower()]

        layers: List[nn.Module] = []
        # Hidden layers: Linear -> activation -> dropout.
        for in_dim, out_dim in zip(widths[:-2], widths[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_cls())
            layers.append(build_dropout(dropout_kind, out_dim, p))
        # Output layer: Linear only. No activation or dropout, since
        # CrossEntropyLoss applies log-softmax internally.
        layers.append(nn.Linear(widths[-2], widths[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten (batch, C, H, W) -> (batch, features).
        x = x.flatten(start_dim=1)
        return self.net(x)

    def dropout_modules(self) -> List[nn.Module]:
        """Per-hidden-layer dropout modules, in forward order.

        Lets a caller iterate over the dropout layers and update per-neuron
        probabilities, for example:
            for layer_idx, drop in enumerate(model.dropout_modules()):
                if isinstance(drop, PerNeuronDropout):
                    drop.set_probs(probs_for_layer(layer_idx))
        """
        return [m for m in self.net
                if isinstance(m, PerNeuronDropout)
                or m.__class__.__name__ in ("UniformDropout", "NoDropout")]

    def linear_layers(self) -> List[nn.Linear]:
        """The Linear layers in forward order.

        These are the weight matrices of the network's weight graph: the k-th
        Linear connects layer k to layer k+1, and the k-th PerNeuronDropout
        (from dropout_modules()) follows the output of the k-th Linear.
        """
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
