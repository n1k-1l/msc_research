"""
Eval-mode dropout sanity checks.

Guards the inverted-dropout convention the whole study relies on: in eval mode a
dropout module is the identity (no masking, no rescaling), so repeated forward
passes are bit-identical; in train mode it is stochastic, so they are not.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dropout import PerNeuronDropout, UniformDropout


def _module(kind):
    return PerNeuronDropout(64, p=0.5) if kind == "per_neuron" else UniformDropout(0.5)


def test_eval_mode_is_identity_and_deterministic():
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    for kind in ("per_neuron", "uniform"):
        m = _module(kind).eval()
        out1 = m(x)
        out2 = m(x)
        assert torch.equal(out1, out2), f"{kind}: eval output not deterministic"
        if kind == "per_neuron":
            # PerNeuronDropout is exactly the identity in eval mode.
            assert torch.equal(out1, x), "per_neuron eval is not the identity"


def test_train_mode_is_stochastic():
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    for kind in ("per_neuron", "uniform"):
        m = _module(kind).train()
        out1 = m(x)
        out2 = m(x)
        assert not torch.equal(out1, out2), f"{kind}: train output is deterministic"


if __name__ == "__main__":
    test_eval_mode_is_identity_and_deterministic()
    test_train_mode_is_stochastic()
    print("ok")
