"""
Dropout modules.

Every dropout strategy is an nn.Module with the same interface: it takes a
post-activation tensor of shape (batch, num_neurons) and returns a tensor of
the same shape, so the MLP is agnostic to which one it holds.

Three are provided: UniformDropout (the baseline), NoDropout (an ablation), and
PerNeuronDropout, which holds an independent retention probability per neuron.
PerNeuronDropout is the integration point for curvature-aware dropout: a
curvature-derived probability vector is passed to set_probs() and nothing else
in the pipeline changes. Initialised with a constant p it is equivalent to
UniformDropout.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class NoDropout(nn.Module):
    """Ablation: dropout disabled. Identity at train and eval time."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def extra_repr(self) -> str:
        return "no-op"


class UniformDropout(nn.Module):
    """
    Standard dropout (primary baseline).

    Wraps nn.Dropout so it shares the interface of the per-neuron variants. Uses
    inverted dropout: at train time survivors are scaled by 1/(1-p) so expected
    activations are unchanged and no rescaling is needed at eval time.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "p must be in [0, 1)"
        self.p = p
        self._drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._drop(x)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class PerNeuronDropout(nn.Module):
    """
    Dropout with an independent retention probability per neuron.

    Initialised with a single p, every neuron shares that probability, making it
    behave identically to UniformDropout; comparing the two is a sanity check.
    A per-neuron probability vector can be supplied at any time via set_probs(),
    which is how curvature-aware dropout drives this module.

    Args:
        num_neurons: width of the layer this dropout follows.
        p: initial drop probability, applied uniformly to every neuron.
    """
    def __init__(self, num_neurons: int, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "p must be in [0, 1)"
        self.num_neurons = num_neurons
        # Registered as a buffer so it moves with .to(device) and is saved in
        # the state_dict, but is not a learnable parameter.
        self.register_buffer("drop_p", torch.full((num_neurons,), float(p)))

    @torch.no_grad()
    def set_probs(self, probs: torch.Tensor) -> None:
        """Replace the per-neuron drop probabilities."""
        probs = probs.to(self.drop_p.device, dtype=self.drop_p.dtype)
        assert probs.shape == self.drop_p.shape, (
            f"expected {self.drop_p.shape}, got {probs.shape}")
        assert torch.all((probs >= 0) & (probs < 1)), "probs must be in [0, 1)"
        self.drop_p.copy_(probs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x  # eval mode: identity, no dropout or rescaling
        keep_p = 1.0 - self.drop_p                      # (num_neurons,)
        mask = torch.bernoulli(keep_p.expand_as(x))     # (batch, num_neurons)
        return x * mask / keep_p                        # inverted scaling

    def extra_repr(self) -> str:
        lo, hi = self.drop_p.min().item(), self.drop_p.max().item()
        return f"num_neurons={self.num_neurons}, drop_p in [{lo:.3f}, {hi:.3f}]"


def build_dropout(kind: str, num_neurons: int, p: float) -> nn.Module:
    """Factory used by the model. Keeps the model agnostic to dropout type."""
    kind = kind.lower()
    if kind in ("none", "no", "off"):
        return NoDropout()
    if kind == "uniform":
        return UniformDropout(p)
    # Per-neuron dropout. Static at constant p unless an epoch hook drives its
    # probabilities (config prob_source); during warm-up it equals uniform.
    if kind == "per_neuron":
        return PerNeuronDropout(num_neurons, p)
    raise ValueError(f"unknown dropout kind: {kind!r}")
