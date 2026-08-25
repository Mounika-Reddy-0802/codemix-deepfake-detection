"""Score a trained detector on one evaluation manifest, with a per-attack breakdown.

Why the breakdown matters. ASVspoof 2019 LA is built so that the eval partition
holds **13 attacks the model has never seen** (A07-A19), while train and dev share
only A01-A06. A single pooled EER hides exactly the thing the corpus was designed
to expose: a detector can look excellent on familiar attacks and fail completely
on one unfamiliar family, and the pooled number will still look respectable
because 12 other attacks carried it.

So every attack is scored separately against the shared bonafide set, which is how
the ASVspoof organisers report results and how a reader can tell "generalises" from
"memorised the six attacks in training".

Scores are **P(bonafide)**, matching ``LABEL_TO_INT`` (bonafide = 1) and the
training loop's own dev evaluation, so a number here is directly comparable to the
dev EER printed during training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def score_manifest(
    checkpoint: str,
    manifest: str,
    encoder: str = "wav2vec2-base",
    batch_size: int = 16,
    max_seconds: float | None = 4.0,
    device: str | None = None,
    data_root: str | None = None,
    limit: int | None = None,
    partial_path: str | None = None,
    flush_every: int = 50,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Run the model over a manifest and return it with a ``score`` column added.

    Inference only, so no gradients are stored and the batch can be larger than in
    training -- the 6 GB card that needs batch 4 to train handles 16 here.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import AudioManifestDataset, collate_fn
    from src.models.detector import Detector
    from src.utils.device import resolve_device

    resolved = resolve_device(device)
    dataset = AudioManifestDataset(
        manifest, data_root=data_root, max_seconds=max_seconds, random_crop=False
    )
    if limit is not None:
        dataset.df = dataset.df.head(limit).reset_index(drop=True)

    model = Detector(_encoder_id(encoder))
    state = torch.load(checkpoint, map_location=resolved, weights_only=False)
    note = _restore_lora(model, state)
    if note:
        print(f"  {note}", flush=True)
    model = model.to(resolved)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # Resume support. This machine's GPU driver intermittently drops the CUDA
    # context mid-run (a TDR, alongside the 0x133 DPC_WATCHDOG bugchecks), which
    # killed a 71,237-clip pass at clip 1,600 and threw the work away. Partial
    # scores are flushed to disk so a driver blip costs one chunk, not the run --
    # the same reason spoof generation is resumable.
    done: dict[str, float] = {}
    if partial_path and Path(partial_path).is_file():
        prior = pd.read_csv(partial_path)
        done = dict(zip(prior["filepath"].astype(str), prior["score"].astype(float), strict=True))
        print(f"  resuming: {len(done)} clip(s) already scored", flush=True)

    todo = dataset.df[~dataset.df["filepath"].astype(str).isin(done)].reset_index(drop=True)
    if todo.empty:
        out = dataset.df.copy()
        out["score"] = out["filepath"].astype(str).map(done)
        return out

    pending = AudioManifestDataset(
        todo, data_root=data_root, max_seconds=max_seconds, random_crop=False
    )
    loader = DataLoader(
        pending,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(0, int(num_workers)),
        pin_memory=resolved.startswith("cuda"),
    )
    scores: list[float] = []
    paths = todo["filepath"].astype(str).tolist()
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader, 1):
                wav = batch["waveforms"].to(resolved)
                lengths = batch["lengths"].to(resolved)
                prob = torch.softmax(model(wav, lengths=lengths), dim=1)[:, 1]  # P(bonafide)
                scores.extend(prob.detach().cpu().numpy().tolist())
                if i % 50 == 0:
                    print(f"  scored {len(done) + len(scores)}/{len(dataset)} clips", flush=True)
                if partial_path and i % flush_every == 0:
                    _flush(partial_path, done, paths, scores)
    finally:
        # Flush whatever survived even when CUDA dies mid-batch, so the next run
        # picks up from here instead of starting over.
        if partial_path:
            _flush(partial_path, done, paths, scores)

    for path, value in zip(paths[: len(scores)], scores, strict=False):
        done[path] = value
    out = dataset.df.copy()
    out["score"] = out["filepath"].astype(str).map(done)
    missing = int(out["score"].isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} clip(s) never scored -- rerun the same command to resume "
            f"from {partial_path}"
        )
    return out


def _restore_lora(model, state) -> str | None:
    """Re-attach LoRA adapters so a Stage-3 checkpoint loads into a plain Detector.

    ``apply_lora`` renames ``...q_proj.weight`` to ``...q_proj.base.weight`` and adds
    ``lora_A``/``lora_B``, so a Stage-3 ``best.pt`` does not fit the unadapted module
    tree that :func:`score_manifest` builds -- ``load_state_dict`` fails on every
    adapted key. Rebuilding the same wrapping first is what makes the gap-closure
    number scoreable at all. Returns a log line, or ``None`` for a Stage-1 checkpoint.

    The *targets* are read back from the checkpoint's own key names and the rank from
    ``lora_A``'s shape, so a run that adapted a non-default set of projections still
    reloads. ``alpha`` alone cannot be recovered from shapes -- it only ever appears
    as a scaling factor -- so the saved config is authoritative there, and a
    checkpoint without one falls back to the default the module documents.
    """
    from src.models.lora import DEFAULT_TARGETS, apply_lora

    weights = state["model"] if isinstance(state, dict) and "model" in state else state
    suffix = ".lora_A"
    adapted = [k for k in weights if k.endswith(suffix)]
    if not adapted:
        return None

    targets = sorted({k[: -len(suffix)].rsplit(".", 1)[-1] for k in adapted})
    rank = int(weights[adapted[0]].shape[0])
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    alpha = float(cfg.get("lora_alpha", 16.0) or 16.0)
    wrapped = apply_lora(model, tuple(targets) or DEFAULT_TARGETS, r=rank, alpha=alpha, dropout=0.0)
    if wrapped != len(adapted):
        raise RuntimeError(
            f"checkpoint holds {len(adapted)} adapted layer(s) but rebuilding matched "
            f"{wrapped} -- the encoder does not match the one this adapter was trained on"
        )
    return f"lora checkpoint: r={rank} alpha={alpha} restored on {wrapped} layers {tuple(targets)}"


def _flush(partial_path: str, done: dict, paths: list, scores: list) -> None:
    """Write every score obtained so far, so a crash cannot discard them."""
    merged = dict(done)
    for path, value in zip(paths[: len(scores)], scores, strict=False):
        merged[path] = value
    Path(partial_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filepath": list(merged), "score": list(merged.values())}).to_csv(
        partial_path, index=False
    )


def _encoder_id(name: str) -> str:
    """Map the short config name to its Hugging Face id."""
    return {
        "wav2vec2-base": "facebook/wav2vec2-base",
        "wavlm-base": "microsoft/wavlm-base",
        "xlsr-53": "facebook/wav2vec2-large-xlsr-53",
    }.get(name, name)


def per_attack_eer(scored: pd.DataFrame) -> pd.DataFrame:
    """EER for each attack, each scored against the SHARED bonafide set.

    Pairing every attack with the same bonafide pool is what makes the columns
    comparable: an attack's EER then reflects only how detectable that attack is,
    not how many bonafide clips happened to sit beside it.
    """
    from src.training.metrics import eer as compute

    bonafide = scored[scored["label"].str.lower() == "bonafide"]
    rows: list[dict[str, object]] = []
    spoof = scored[scored["label"].str.lower() == "spoof"]
    for attack, group in spoof.groupby("tool"):
        pair = pd.concat([bonafide, group], ignore_index=True)
        labels = (pair["label"].str.lower() == "bonafide").astype(int).to_numpy()
        rows.append(
            {
                "attack": str(attack),
                "spoof_clips": int(len(group)),
                "eer": round(float(compute(pair["score"].to_numpy(), labels)), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("eer", ascending=False).reset_index(drop=True)


def summarise(scored: pd.DataFrame) -> dict:
    """Pooled metrics over the whole manifest."""
    from src.training.metrics import evaluate as eval_metrics

    labels = (scored["label"].str.lower() == "bonafide").astype(int).to_numpy()
    metrics = eval_metrics(scored["score"].to_numpy(), labels, n_boot=200)
    metrics.update(
        clips=int(len(scored)),
        bonafide=int(labels.sum()),
        spoof=int((labels == 0).sum()),
        score_std=round(float(scored["score"].std()), 6),
    )
    return metrics


def main() -> None:
    """CLI: ``python -m src.training.evaluate --checkpoint ... --manifest ...``."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a detector on one manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--encoder", default="wav2vec2-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N clips")
    parser.add_argument("--out", default="experiments/eval_results.json")
    parser.add_argument("--scores-out", default=None, help="also write the per-clip scores CSV")
    parser.add_argument(
        "--partial",
        default=None,
        help="incremental score cache; rerunning with the same path resumes",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for wav decode; >0 keeps the GPU fed on a fast card",
    )
    args = parser.parse_args()

    scored = score_manifest(
        args.checkpoint,
        args.manifest,
        encoder=args.encoder,
        batch_size=args.batch_size,
        max_seconds=args.max_seconds,
        device=args.device,
        data_root=args.data_root,
        limit=args.limit,
        partial_path=args.partial,
        num_workers=args.num_workers,
    )
    pooled = summarise(scored)
    attacks = per_attack_eer(scored)

    print("\n=== pooled ===")
    print(json.dumps(pooled, indent=2))
    print("\n=== per attack (worst first) ===")
    print(attacks.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "manifest": args.manifest,
                "encoder": args.encoder,
                "max_seconds": args.max_seconds,
                "pooled": pooled,
                "per_attack": attacks.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    if args.scores_out:
        scored.to_csv(args.scores_out, index=False)
        print(f"wrote {args.scores_out}")


if __name__ == "__main__":
    main()
