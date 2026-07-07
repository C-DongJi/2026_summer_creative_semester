"""멀티 해상도 STFT 손실 테스트."""
import torch

from src.training.losses import MultiResolutionSTFTLoss, build_loss


def test_zero_loss_on_identical_input():
    loss_fn = MultiResolutionSTFTLoss(windows=(1024, 512), hop=147)
    x = torch.randn(2, 2, 44100)
    assert loss_fn(x, x).item() == 0.0


def test_positive_loss_on_different_input():
    loss_fn = MultiResolutionSTFTLoss(windows=(1024, 512), hop=147)
    torch.manual_seed(0)
    x = torch.randn(1, 2, 22050)
    y = torch.randn(1, 2, 22050)
    assert loss_fn(x, y).item() > 0.0


def test_gradient_flows():
    loss_fn = MultiResolutionSTFTLoss(windows=(512,), hop=147)
    x = torch.randn(1, 2, 8192, requires_grad=True)
    y = torch.randn(1, 2, 8192)
    loss = loss_fn(x, y)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_build_loss_names():
    assert isinstance(build_loss("multi_stft"), MultiResolutionSTFTLoss)
    assert isinstance(build_loss("l1"), torch.nn.L1Loss)
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)
