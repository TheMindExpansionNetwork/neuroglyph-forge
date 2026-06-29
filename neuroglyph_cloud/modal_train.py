"""
Modal GPU training for NeuroGlyph Forge.

Pattern: https://github.com/modal-labs/modal-examples (06_gpu_and_ml, long-training)

Setup:
  pip install modal
  modal setup   # API token from modal.com

Train on cloud:
  modal run neuroglyph_cloud/modal_train.py --task hand --epochs 40

Fine-tune (upload base ckpt to volume first, or train from scratch on volume data):
  modal run neuroglyph_cloud/modal_train.py --finetune --task hand --epochs 30
"""

from __future__ import annotations

import pathlib

import modal

APP_NAME = "neuroglyph-forge-train"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_dir(PROJECT_ROOT, remote_path="/root/neuroglyph-forge")
    .uv_pip_install(
        "torch~=2.4.0",
        "numpy>=1.26",
        "pydantic>=2.0",
    )
)

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name("neuroglyph-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("neuroglyph-checkpoints", create_if_missing=True)


@app.function(
    gpu="T4",
    timeout=60 * 60,
    volumes={
        "/data": data_vol,
        "/ckpt": ckpt_vol,
    },
)
def train_on_gpu(
    task: str = "hand",
    model: str = "tiny_b2q",
    epochs: int = 40,
    lr: float = 1e-3,
    finetune: bool = False,
    freeze_encoder: bool = False,
    base_name: str = "tiny_b2q_hand.pt",
):
    import sys

    sys.path.insert(0, "/root/neuroglyph-forge")

    import torch
    from neuroglyph_train.engine import finetune_decoder, train_decoder

    data_dir = pathlib.Path("/data/processed")
    if not data_dir.exists() or not list(data_dir.glob("*.pt")):
        from neuroglyph_data.synthetic import write_synthetic_processed

        write_synthetic_processed(data_dir, task=task, n_epochs=800)
        data_vol.commit()

    out_dir = pathlib.Path("/ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available={torch.cuda.is_available()} device={device}", flush=True)

    if finetune:
        base = out_dir / base_name
        if not base.exists():
            base = out_dir / f"finetune_{task}.pt"
        if not base.exists():
            raise FileNotFoundError(
                f"No base checkpoint at {out_dir}; run without --finetune first or upload to volume"
            )
        path = finetune_decoder(
            base_checkpoint=base,
            data_dir=data_dir,
            task=task,
            epochs=epochs,
            lr=lr * 0.1,
            freeze_encoder=freeze_encoder,
            output_dir=out_dir,
            device=device,
        )
    else:
        path = train_decoder(
            task=task,
            data_dir=data_dir,
            model_name=model,
            epochs=epochs,
            lr=lr,
            output_dir=out_dir,
            device=device,
        )

    ckpt_vol.commit()
    return str(path)


@app.local_entrypoint()
def main(
    task: str = "hand",
    model: str = "tiny_b2q",
    epochs: int = 40,
    lr: float = 1e-3,
    finetune: bool = False,
    freeze_encoder: bool = False,
    upload_data: bool = True,
):
    """Upload local processed tensors then launch GPU training."""
    local_processed = PROJECT_ROOT / "data" / "processed"
    if upload_data and local_processed.exists():
        with data_vol.batch_upload() as batch:
            for pt in local_processed.glob("*.pt"):
                batch.put_file(str(pt), f"processed/{pt.name}")
            manifest = local_processed / "manifest.json"
            if manifest.exists():
                batch.put_file(str(manifest), "processed/manifest.json")

    if upload_data:
        local_ckpt = PROJECT_ROOT / "checkpoints"
        if local_ckpt.exists():
            with ckpt_vol.batch_upload() as batch:
                for pt in local_ckpt.glob("*.pt"):
                    batch.put_file(str(pt), pt.name)

    result = train_on_gpu.remote(
        task=task,
        model=model,
        epochs=epochs,
        lr=lr,
        finetune=finetune,
        freeze_encoder=freeze_encoder,
    )
    print("remote checkpoint:", result)