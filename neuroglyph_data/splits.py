"""Load processed tensors and build train/val splits."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split


class EpochDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_processed_dir(data_dir: Path, task: str | None = None) -> tuple[torch.Tensor, torch.Tensor, str]:
    data_dir = Path(data_dir)
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        task = task or info.get("task", "hand")
        pt_name = info["files"][0]
    else:
        task = task or "hand"
        for candidate in (f"processed_{task}.pt", f"synthetic_{task}.pt"):
            if (data_dir / candidate).exists():
                pt_name = candidate
                break
        else:
            pt_name = f"synthetic_{task}.pt"
            from neuroglyph_data.synthetic import write_synthetic_processed

            write_synthetic_processed(data_dir, task=task)

    blob = torch.load(data_dir / pt_name, weights_only=True)
    return blob["X"], blob["y"], blob.get("task", task)


def train_val_loaders(
    data_dir: Path,
    task: str,
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 0,
):
    X, y, _ = load_processed_dir(data_dir, task=task)
    ds = EpochDataset(X, y)
    n_val = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    from torch.utils.data import DataLoader

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )