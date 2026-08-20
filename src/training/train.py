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

import time
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
    #: Cap on clip length in seconds (None = whole clip). Batches pad to their
    #: longest member, so this is what makes peak VRAM predictable.
    max_seconds: float | None = None
    epochs: int = 10
    num_workers: int = 2
    seed: int = 1234
    amp: bool = True
    freeze_feature_extractor: bool = True
    freeze_encoder: bool = False
    max_steps: int | None = None  # smoke-test cap
    #: Print a progress line every N optimiser steps. The loop used to print only
    #: at start and at the very end, so a run that was working looked identical to
    #: one that had hung -- on this laptop that was 7 minutes of silence.
    log_every: int = 25
    #: Cap the dev pass to N clips (None = the whole set). Used by --smoke so a
    #: quick check does not spend its whole runtime evaluating 24,844 clips.
    dev_subset: int | None = None
    #: Weight the loss by inverse class frequency. ASVspoof LA train is 89.8%
    #: spoof (22,800 vs 2,580), and with a plain cross-entropy the cheapest
    #: solution is to ignore the audio and always answer "spoof": that scores a
    #: loss of ~0.329, which is precisely where the first 3-epoch run sat while
    #: its dev EER wandered around 0.49 -- chance. The model emitted a CONSTANT
    #: P(bonafide)=0.128 for every clip, matching the 0.102 class prior, with
    #: zero separation between the classes. Weighting makes the rare class
    #: expensive enough to be worth learning.
    class_weighted: bool = True
    #: Continue from ``<out_dir>/last.pt`` when it exists. Epoch-level rather than
    #: step-level: an epoch here is ~45 minutes, which is inside the window this
    #: machine usually stays up for.
    resume: bool = False
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


#: Dev-score spread below which the model is considered collapsed. Real runs sit
#: orders of magnitude above this; a collapsed one sits at exactly 0.
COLLAPSE_STD = 1e-4


def _evaluate(model, loader, device) -> dict[str, float]:
    """Run the dev set and return EER/AUC (higher score = bonafide)."""
    import numpy as np
    import torch

    from src.training.metrics import evaluate as eval_metrics

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    # Populated below; a near-zero spread means the model is emitting one number
    # for every clip and has stopped depending on its input at all.
    with torch.no_grad():
        for batch in loader:
            wav = batch["waveforms"].to(device)
            logits = model(wav, lengths=batch["lengths"].to(device))
            prob = torch.softmax(logits, dim=1)[:, 1]  # P(bonafide)
            scores.extend(prob.detach().cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
    score_array = np.array(scores)
    metrics = eval_metrics(score_array, np.array(labels), n_boot=200)
    metrics["score_std"] = float(score_array.std()) if score_array.size else 0.0
    return metrics


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

    train_ds = AudioManifestDataset(
        cfg.train_manifest,
        data_root=cfg.data_root,
        max_seconds=cfg.max_seconds,
        random_crop=True,
        seed=cfg.seed,
    )
    if cfg.subset_frac < 1.0:
        n = max(1, int(len(train_ds) * cfg.subset_frac))
        train_ds.df = train_ds.df.sample(n=n, random_state=cfg.seed).reset_index(drop=True)
    dev_ds = AudioManifestDataset(
        cfg.dev_manifest,
        data_root=cfg.data_root,
        max_seconds=cfg.max_seconds,
        random_crop=False,
        seed=cfg.seed,
    )
    if cfg.dev_subset is not None and cfg.dev_subset < len(dev_ds):
        # Stratified by label so a capped dev set keeps both classes; an
        # all-bonafide sample would make EER meaningless.
        dev_ds.df = (
            dev_ds.df.groupby("label", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, int(cfg.dev_subset * len(g) / len(dev_ds.df))), random_state=cfg.seed
                )
            )
            .reset_index(drop=True)
        )

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
    if cfg.class_weighted:
        counts = train_ds.df["label"].str.lower().value_counts()
        n_spoof = float(counts.get("spoof", 0))
        n_bona = float(counts.get("bonafide", 0))
        total = n_spoof + n_bona
        # Index order follows LABEL_TO_INT: 0 = spoof, 1 = bonafide.
        weights = torch.tensor(
            [total / (2.0 * max(n_spoof, 1.0)), total / (2.0 * max(n_bona, 1.0))],
            dtype=torch.float32,
            device=device,
        )
        print(
            f"class weights: spoof {weights[0]:.3f}, bonafide {weights[1]:.3f} "
            f"(from {int(n_spoof)} spoof / {int(n_bona)} bonafide)",
            flush=True,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    use_amp = amp_enabled(device, cfg.amp)
    autocast_type = device_family(device)
    scaler = make_grad_scaler(device, enabled=use_amp)

    run = _init_wandb(cfg)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    best_eer = 1.0
    step = 0
    start_epoch = 0
    snapshot = Path(cfg.out_dir) / "last.pt"
    if cfg.resume and snapshot.is_file():
        state = torch.load(snapshot, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        step = int(state.get("step", 0))
        best_eer = float(state.get("best_eer", 1.0))
        print(
            f"resumed from {snapshot}: epoch {start_epoch}, step {step}, "
            f"best EER so far {best_eer:.4f}",
            flush=True,
        )
    running = 0.0
    t0 = time.time()
    total_steps = len(train_loader) * cfg.epochs
    print(
        f"training: {len(train_ds)} train / {len(dev_ds)} dev clips, "
        f"batch {cfg.batch_size}, {len(train_loader)} steps/epoch x {cfg.epochs} epochs",
        flush=True,
    )
    for epoch in range(start_epoch, cfg.epochs):
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
            running += float(loss.item())
            if run is not None:
                run.log({"train/loss": float(loss.item()), "epoch": epoch}, step=step)
            if cfg.log_every and step % cfg.log_every == 0:
                rate = step / max(time.time() - t0, 1e-9)
                done = f"{step}/{cfg.max_steps}" if cfg.max_steps else f"{step}/{total_steps}"
                eta = (
                    (total_steps - step) / rate / 60
                    if cfg.max_steps is None and total_steps
                    else 0.0
                )
                print(
                    f"  epoch {epoch + 1}/{cfg.epochs} step {done}  "
                    f"loss {running / cfg.log_every:.4f}  {rate:.2f} steps/s"
                    + (f"  eta {eta:.0f} min" if eta else ""),
                    flush=True,
                )
                running = 0.0
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break

        print(f"  evaluating on {len(dev_ds)} dev clips ...", flush=True)
        metrics = _evaluate(model, dev_loader, device)
        print(
            f"  epoch {epoch + 1}: dev EER {metrics['eer']:.4f}  "
            f"score std {metrics.get('score_std', float('nan')):.4f}  "
            f"({(time.time() - t0) / 60:.1f} min elapsed)",
            flush=True,
        )
        # Abort on a collapsed model rather than spending the remaining epochs on
        # it. A healthy detector spreads its scores; one that has stopped reading
        # its input returns the same number for every clip, which shows up as a
        # standard deviation of essentially zero long before the EER looks odd.
        # The first Stage-1 run sat at std 0.0000 for 210 minutes and finished at
        # EER 0.485 -- chance -- because nothing was watching for this.
        if metrics.get("score_std", 1.0) < COLLAPSE_STD:
            raise RuntimeError(
                f"model collapsed: dev score std {metrics['score_std']:.6f} < {COLLAPSE_STD} "
                f"after epoch {epoch + 1} -- it is emitting a constant regardless of input. "
                "Usually the learning rate is too high for a pretrained encoder "
                "(1e-4 does this to wav2vec2; 1e-5 works). Stopping instead of "
                "spending the remaining epochs."
            )
        if run is not None:
            run.log({f"dev/{k}": v for k, v in metrics.items()}, step=step)
        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            torch.save(
                {"model": model.state_dict(), "config": cfg.__dict__, "metrics": metrics},
                Path(cfg.out_dir) / "best.pt",
            )
        # A resumable snapshot, separate from best.pt. best.pt holds the model
        # worth keeping; this holds everything needed to CONTINUE -- optimizer
        # moments, AMP scaler state, the epoch reached and the best EER so far.
        # Without it a crash in epoch 10 of 15 discards nine epochs of GPU time,
        # which on this machine (0x133 bugchecks roughly every 30-40 min) is not a
        # hypothetical. Written to a temp file and renamed so a crash mid-write
        # cannot leave a truncated checkpoint that fails to load.
        snapshot = Path(cfg.out_dir) / "last.pt"
        staging = snapshot.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "step": step,
                "best_eer": best_eer,
                "config": cfg.__dict__,
            },
            staging,
        )
        staging.replace(snapshot)
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
        "--resume", action="store_true", help="continue from <out_dir>/last.pt if present"
    )
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
    if args.resume:
        cfg.resume = True
    if args.smoke:
        cfg.subset_frac = 0.01
        cfg.max_steps = 20
        cfg.epochs = 1
        cfg.wandb_mode = "offline"
        # Cap the dev pass too. Without this the "quick" check trains 20 steps and
        # then evaluates all 24,844 dev clips, which is most of its runtime and
        # makes a smoke test useless as a smoke test.
        cfg.dev_subset = 200
        cfg.log_every = 5
    result = train(cfg)
    print(f"done: best dev EER = {result['best_eer']:.4f}")


if __name__ == "__main__":
    main()
