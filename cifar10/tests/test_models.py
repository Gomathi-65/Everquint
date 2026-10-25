import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import get_loaders  # noqa: E402
from models import CNN, ResidualBlock, build_model, count_params  # noqa: E402
from utils import set_seed  # noqa: E402


@pytest.mark.parametrize("name", ["baseline", "residual"])
def test_output_shape_and_backward(name):
    model = build_model(name)
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_residual_block_shortcuts():
    assert isinstance(ResidualBlock(16, 16).shortcut, torch.nn.Identity)
    blk = ResidualBlock(16, 32)
    assert not isinstance(blk.shortcut, torch.nn.Identity)
    assert blk(torch.randn(2, 16, 8, 8)).shape == (2, 32, 8, 8)


def test_residual_block_skip_carries_signal():
    blk = ResidualBlock(8, 8).eval()
    for p in list(blk.conv1.parameters()) + list(blk.conv2.parameters()):
        torch.nn.init.zeros_(p)
    x = torch.rand(1, 8, 4, 4)  # non-negative input
    assert torch.allclose(blk(x), x, atol=1e-5)  # with zero convs the block returns ReLU(x) == x


def test_eval_mode_is_deterministic_and_dropout_active_in_train():
    m = CNN(dropout=0.5)
    x = torch.randn(8, 3, 32, 32)
    m.eval()
    assert torch.equal(m(x), m(x))
    m.train()
    assert not torch.equal(m(x), m(x))


def test_seed_makes_init_reproducible():
    set_seed(1)
    a = build_model("baseline")
    set_seed(1)
    b = build_model("baseline")
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_param_counts_positive():
    assert count_params(build_model("baseline")) > 0 and count_params(build_model("residual")) > 0


def test_fake_loader_batch_shape():
    tr, va, te = get_loaders(batch_size=32, num_workers=0, fake=True)
    x, y = next(iter(tr))
    assert x.shape == (32, 3, 32, 32) and y.shape == (32,)
    assert len(va.dataset) == 100 and len(te.dataset) == 200
