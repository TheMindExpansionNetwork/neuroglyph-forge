import torch
import torch.nn as nn
import torch.optim as optim

from neuroglyph_models.tiny_b2q import TinyB2Q


def test_training_smoke():
    model = TinyB2Q(n_channels=14, n_classes=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(32, 14, 25)
    y = torch.randint(0, 2, (32,))

    for _ in range(5):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    assert loss.item() > 0