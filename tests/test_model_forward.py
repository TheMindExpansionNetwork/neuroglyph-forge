import torch
from neuroglyph_models.tiny_b2q import TinyB2Q


def test_model_forward():
    model = TinyB2Q(n_channels=14, n_classes=4)
    x = torch.randn(8, 14, 25)
    y = model(x)
    assert y.shape == (8, 4)
    assert torch.isfinite(y).all()