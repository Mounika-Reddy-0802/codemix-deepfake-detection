"""AASIST as an outside baseline, scored and fine-tuned on this project's manifests.

AASIST (Jung et al., ICASSP 2022) is a widely used end-to-end anti-spoofing model:
a sinc-filter front end on the raw waveform and graph attention over spectral and
temporal nodes. Its public weights (``third_party/aasist/AASIST.pth``) were trained
on ASVspoof 2019 LA, the same English corpus as our Stage-1 model, so it is scored
two ways:

* **zero-shot** - the published weights, like S1: how far an established English
  detector transfers to code-mixed, phone-line speech;
* **fine-tuned** - started from those weights and trained on exactly the manifest a
  wav2vec2 system used, with the same dev set and seed, so the comparison with S2
  differs only in the architecture.

Scores are the log-odds of bonafide (logit difference). They rank clips exactly as
P(bonafide) does, but a confident network saturates the probability to 0 or 1 in
float32 and ties would blur the EER; the log-odds never tie. The metric code is
``src.training.evaluate``'s own (pooled EER with bootstrap, per-attack EER against the
shared genuine set), so every number is directly comparable. A genuine-only
manifest gets the false-alarm summary of ``src.training.false_alarm`` instead, at
log-odds 0 (= P(bonafide) 0.5, the deployed operating point).

AASIST reads a fixed 64,600 samples (~4 s at 16 kHz): longer clips are cropped
(the head for scoring, a seeded random window for training) and shorter ones are
repeated, as in the reference implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NB_SAMP = 64_600
LOG_ODDS_THRESHOLD = 0.0  # = P(bonafide) 0.5
WEIGHTS = "third_party/aasist/AASIST.pth"
WEIGHTS_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": NB_SAMP,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def fixed_length(wave: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Exactly ``NB_SAMP`` samples: crop (head, or random with ``rng``) or repeat."""
    if len(wave) >= NB_SAMP:
        start = 0 if rng is None else int(rng.integers(0, len(wave) - NB_SAMP + 1))
        return wave[start : start + NB_SAMP]
    if len(wave) == 0:
        return np.zeros(NB_SAMP, dtype=np.float32)
    reps = NB_SAMP // len(wave) + 1
    return np.tile(wave, reps)[:NB_SAMP]


def build(weights: str | None = WEIGHTS, device: str = "cpu"):
    """The AASIST network, with the published (hash-checked) or given weights."""
    import torch

    from src.baselines.aasist_model import Model

    model = Model(MODEL_CONFIG)
    if weights:
        if Path(weights).as_posix().endswith(WEIGHTS):
            digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
            if digest != WEIGHTS_SHA256:
                raise ValueError(f"{weights} is not the published AASIST checkpoint")
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device)


class ClipSet:
    """A manifest as fixed-length waveforms and labels (1 = bonafide)."""

    def __init__(self, df: pd.DataFrame, data_root: str | None, train: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        from src.utils.audio_utils import load_wav
        from src.utils.paths import resolve

        row = self.df.iloc[index]
        wave, _ = load_wav(resolve(row["filepath"], self.data_root), target_sr=16_000)
        rng = np.random.default_rng((self.seed, self.epoch, index)) if self.train else None
        label = int(str(row["label"]).lower() == "bonafide")
        return fixed_length(np.asarray(wave, dtype=np.float32), rng), label


def _loader(dataset, batch_size: int, shuffle: bool, workers: int, seed: int = 0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, generator=generator
    )


def score(
    model,
    df: pd.DataFrame,
    data_root: str | None,
    device: str = "cpu",
    batch_size: int = 32,
    workers: int = 0,
) -> pd.DataFrame:
    """The manifest with a ``score`` column: AASIST's log-odds of bonafide."""
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for waves, _ in _loader(ClipSet(df, data_root), batch_size, False, workers):
            _, logits = model(waves.to(device))
            scores.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
    out = df.reset_index(drop=True).copy()
    out["score"] = np.concatenate(scores) if scores else np.array([])
    return out


def result(scored: pd.DataFrame) -> dict:
    """EER metrics when both classes are present, else the false-alarm summary."""
    from src.training import evaluate, false_alarm

    labels = set(scored["label"].str.lower())
    if labels == {"bonafide"}:
        return {"summary": false_alarm.summarise(scored, threshold=LOG_ODDS_THRESHOLD)}
    return {
        "pooled": evaluate.summarise(scored),
        "per_attack": evaluate.per_attack_eer(scored).to_dict("records"),
    }


def train(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    data_root: str | None,
    out_dir: str,
    init: str | None = WEIGHTS,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 24,
    seed: int = 1234,
    device: str = "cpu",
    workers: int = 0,
    log=print,
) -> float:
    """Fine-tune AASIST; keep the epoch with the lowest dev EER. Returns that EER."""
    import torch

    from src.training.metrics import eer

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(init, device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    counts = train_df["label"].str.lower().value_counts()
    # Class weights as in our training loop: each class counts equally.
    weight = torch.tensor(
        [
            len(train_df) / (2 * counts.get("spoof", 1)),
            len(train_df) / (2 * counts.get("bonafide", 1)),
        ],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    dataset = ClipSet(train_df, data_root, train=True, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(epochs):
        dataset.epoch = epoch
        model.train()
        total = 0.0
        for waves, labels in _loader(dataset, batch_size, True, workers, seed + epoch):
            optimiser.zero_grad()
            _, logits = model(waves.to(device), Freq_aug=True)
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimiser.step()
            total += float(loss) * len(labels)
        scheduler.step()
        scored = score(model, dev_df, data_root, device, batch_size, workers)
        dev = eer(
            scored["score"].to_numpy(),
            (scored["label"].str.lower() == "bonafide").to_numpy().astype(int),
        )
        log(f"epoch {epoch + 1}/{epochs}  loss {total / len(dataset):.4f}  dev EER {dev:.4f}")
        if dev < best:
            best = dev
            torch.save(model.state_dict(), out / "best.pt")
    log(f"best dev EER = {best:.4f}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="AASIST baseline: score or fine-tune")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--weights", default=WEIGHTS)
    s.add_argument("--manifest", required=True)
    s.add_argument("--data-root", default=None)
    s.add_argument("--device", default="cpu")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--scores-out", required=True)
    s.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train-manifest", required=True)
    t.add_argument("--dev-manifest", required=True)
    t.add_argument("--data-root", default=None)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--init", default=WEIGHTS)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=24)
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--device", default="cpu")
    t.add_argument("--num-workers", type=int, default=0)
    t.add_argument("--limit", type=int, default=None, help="first N rows of each manifest (smoke)")
    args = parser.parse_args()

    if args.command == "score":
        df = pd.read_csv(args.manifest, dtype={"speaker": str})
        if args.limit:
            df = df.groupby("label", group_keys=False).head(args.limit)
        model = build(args.weights, args.device)
        scored = score(model, df, args.data_root, args.device, args.batch_size, args.num_workers)
        scored.to_csv(args.scores_out, index=False)
        payload = {
            "model": "aasist",
            "weights": args.weights,
            "manifest": args.manifest,
            **result(scored),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
        print(
            json.dumps(
                {k: v for k, v in payload.items() if k in ("pooled", "summary")}, default=float
            )
        )
    else:
        tr = pd.read_csv(args.train_manifest, dtype={"speaker": str})
        dev = pd.read_csv(args.dev_manifest, dtype={"speaker": str})
        if args.limit:
            tr = tr.groupby("label", group_keys=False).head(args.limit)
            dev = dev.groupby("label", group_keys=False).head(args.limit)
        train(
            tr,
            dev,
            args.data_root,
            args.out_dir,
            args.init,
            args.epochs,
            args.lr,
            args.batch_size,
            args.seed,
            args.device,
            args.num_workers,
            log=lambda m: print(m, flush=True),
        )


if __name__ == "__main__":
    main()
