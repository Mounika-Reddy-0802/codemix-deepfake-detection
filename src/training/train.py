"""Stage-1 / Stage-3 training loop: mixed precision, checkpointing, W&B logging.

Config-driven (``configs/train_baseline.yaml``). Heavy imports (torch/transformers/
wandb) are lazy inside :func:`train`, so importing this module is cheap (CI-safe).
Use ``--smoke`` (or ``max_steps`` in the config) to verify the loop learns on a
1% subset before committing GPU hours.

The same config runs on the CPU laptop and on a Colab/Kaggle GPU: the device comes
from :mod:`src.utils.device` (override with ``$DFD_DEVICE`` or ``--device``), AMP
switches itself off when there is no GPU to use it, and clip paths are resolved
against ``$DATA_ROOT`` so the manifest itself never has to be edited.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyper-parameters + paths for one training run."""

    encoder: str = "wav2vec2-base"
    train_manifest: str = "data/manifests/train.csv"
    dev_manifest: str = "data/manifests/dev.csv"
    out_dir: str = "checkpoints/baseline"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    epochs: int = 10
    num_workers: int = 2
    seed: int = 1234
    amp: bool = True
    freeze_feature_extractor: bool = True
    freeze_encoder: bool = False
    max_steps: int | None = None  # smoke-test cap
    subset_frac: float = 1.0  # fraction of the train manifest to use
    device: str | None = None  # None/"auto" -> detect; "cpu"/"cuda" -> force
    data_root: str | None = None  # None -> $DATA_ROOT, else this machine's data dir
    wandb_project: str = "codemix-deepfake-detection"
    wandb_mode: str = "online"


def load_config(path: str) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file (unknown keys ignored)."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    known = {k: raw[k] for k in raw if k in TrainConfig.__dataclass_fields__}
    return TrainConfig(**known)


def _evaluate(model, loader, device) -> dict[str, float]:
    """Run the dev set and return EER/AUC (higher score = bonafide)."""
    import numpy as np
    import torch

    from src.training.metrics import evaluate as eval_metrics

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            wav = batch["waveforms"].to(device)
            logits = model(wav, lengths=batch["lengths"].to(device))
            prob = torch.softmax(logits, dim=1)[:, 1]  # P(bonafide)
            scores.extend(prob.detach().cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
    return eval_metrics(np.array(scores), np.array(labels), n_boot=200)


def train(cfg: TrainConfig) -> dict[str, float]:
    """Train the detector and return the best dev metrics."""
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import AudioManifestDataset, collate_fn
    from src.models.detector import Detector
    from src.utils.device import (
        amp_enabled,
        dataloader_kwargs,
        describe,
        device_family,
        make_grad_scaler,
        resolve_device,
    )
    from src.utils.seed import set_seed

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(describe(device))
    loader_opts = dataloader_kwargs(device, cfg.num_workers)

    train_ds = AudioManifestDataset(cfg.train_manifest, data_root=cfg.data_root)
    if cfg.subset_frac < 1.0:
        n = max(1, int(len(train_ds) * cfg.subset_frac))
        train_ds.df = train_ds.df.sample(n=n, random_state=cfg.seed).reset_index(drop=True)
    dev_ds = AudioManifestDataset(cfg.dev_manifest, data_root=cfg.data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_opts,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_opts,
    )

    model = Detector(
        cfg.encoder,
        freeze_feature_extractor=cfg.freeze_feature_extractor,
        freeze_encoder=cfg.freeze_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = amp_enabled(device, cfg.amp)
    autocast_type = device_family(device)
    scaler = make_grad_scaler(device, enabled=use_amp)

    run = _init_wandb(cfg)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    best_eer = 1.0
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            wav = batch["waveforms"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=autocast_type, enabled=use_amp):
                logits = model(wav, lengths=lengths)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if run is not None:
                run.log({"train/loss": float(loss.item()), "epoch": epoch}, step=step)
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break

        metrics = _evaluate(model, dev_loader, device)
        if run is not None:
            run.log({f"dev/{k}": v for k, v in metrics.items()}, step=step)
        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            torch.save(
                {"model": model.state_dict(), "config": cfg.__dict__, "metrics": metrics},
                Path(cfg.out_dir) / "best.pt",
            )
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break

    if run is not None:
        run.finish()
    return {"best_eer": best_eer}


def _init_wandb(cfg: TrainConfig):
    """Start a W&B run if a key is configured; else return None."""
    import os

    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    return wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.__dict__)


def main() -> None:
    """CLI: ``python -m src.training.train --config configs/train_baseline.yaml [--smoke]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the spoof detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="1%% subset, few steps")
    parser.add_argument(
        "--device",
        default=None,
        help="auto | cpu | cuda[:N] | mps; overrides the config and $DFD_DEVICE",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="this machine's data directory; overrides the config and $DATA_ROOT",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.smoke:
        cfg.subset_frac = 0.01
        cfg.max_steps = 20
        cfg.epochs = 1
        cfg.wandb_mode = "offline"
    result = train(cfg)
    print(f"done: best dev EER = {result['best_eer']:.4f}")


if __name__ == "__main__":
    main()
