"""Package the final-recipe detector for publication, and load a published package.

The weights are published only once the team and the mentor agree (``docs/MODEL_CARD.md``,
"Release decision"). This module prepares the package so that step is a single upload.

A package is a folder with:

- ``model.safetensors``: the whole state dict. The adapters alone would not be enough:
  they sit on Stage-1's fine-tuned encoder, not on the original wav2vec 2.0 weights.
- ``config.json``: the training config, the live operating point, and how audio is windowed.
- ``README.md``: the model card with Hugging Face metadata.
- ``SHA256SUMS``: a checksum of every other file, checked again on loading.

``safetensors`` stores tensors only, so loading a package runs no pickled code, unlike
``torch.load`` on a checkpoint.

Run::

    python -m src.inference.model_package export --checkpoint final_s1234_best.zip \\
        --out D:/release/codemix-deepfake-detector
    python -m src.inference.model_package verify --checkpoint final_s1234_best.zip \\
        --package D:/release/codemix-deepfake-detector
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.inference.streaming import (
    EMA_ALPHA,
    HOP_SECONDS,
    MIN_RMS_DBFS,
    SAMPLE_RATE,
    TARGET_DBFS,
    WINDOW_SECONDS,
)

FORMAT_VERSION = 1
WEIGHTS = "model.safetensors"
CONFIG = "config.json"
CARD = "README.md"
SUMS = "SHA256SUMS"
LICENSE = "cc-by-nc-sa-4.0"
FRONT_MATTER = f"""---
license: {LICENSE}
language:
- hi
- en
pipeline_tag: audio-classification
base_model: facebook/wav2vec2-base
tags:
- audio-deepfake-detection
- anti-spoofing
- code-mixing
- telephony
- wav2vec2
- lora
---
"""
LOADING = """
## Loading this package

The package is loaded with the project's code, which rebuilds the detector exactly as every
result in the paper was scored:

```python
from src.inference.model_package import load_package
from src.inference.predict import make_score_fn

model, device = load_package("path/to/this/folder")
score = make_score_fn(model, device)  # 4 s of 16 kHz float32 audio -> P(genuine)
```

`config.json` holds the live operating point (threshold and warning ladder) and the
windowing: 4 s windows every 2 s, each level-normalised to -23 dBFS, windows quieter than
-50 dBFS skipped, and the scores smoothed before the threshold is applied.
"""


def model_card(card: str) -> str:
    """The Hugging Face README: metadata, the card without its draft notice, how to load."""
    lines = card.splitlines()
    kept = [line for line in lines if not line.startswith(">")]  # the "draft" blockquote
    body = "\n".join(kept).strip() + "\n"
    return FRONT_MATTER + "\n" + body + LOADING


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sums(folder: Path) -> None:
    """Checksum every file of the package except the checksum file itself."""
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name != SUMS)
    lines = [f"{sha256(p)}  {p.name}" for p in files]
    (folder / SUMS).write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_sums(folder: Path) -> None:
    """Raise if any packaged file is missing or differs from its checksum."""
    for line in (folder / SUMS).read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name}: missing or changed since the package was made")


def _plain(value):
    """JSON-safe copy of a config or metrics value (paths and numpy numbers as plain types)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def package_config(state: dict, operating_point: dict) -> dict:
    """What a user needs besides the weights: training config, operating point, windowing."""
    weights = state["model"]
    return {
        "format_version": FORMAT_VERSION,
        "license": LICENSE,
        "encoder": state.get("config", {}).get("encoder", "wav2vec2-base"),
        "parameters": int(sum(t.numel() for t in weights.values())),
        "output": "softmax over 2 classes; index 1 is P(genuine)",
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "level_dbfs": TARGET_DBFS,
            "skip_below_dbfs": MIN_RMS_DBFS,
            "smoothing_alpha": EMA_ALPHA,
        },
        "operating_point": _plain(operating_point),
        "training": _plain(state.get("config", {})),
        "dev_metrics": _plain(state.get("metrics", {})),
    }


def export(state: dict, out: Path, operating_point: dict, card: str) -> dict:
    """Write a package from a loaded checkpoint ``{"model", "config", "metrics"}``."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu().contiguous() for k, v in state["model"].items()}
    save_file(weights, str(out / WEIGHTS), metadata={"format": "pt"})
    config = package_config(state, operating_point)
    (out / CONFIG).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / CARD).write_text(model_card(card), encoding="utf-8")
    write_sums(out)
    return config


def load_package(folder: str | Path, device: str | None = None, build=None):
    """Load a package for inference: ``(model, device)``, like ``predict.load_detector``.

    ``build`` turns ``(encoder, state)`` into an empty detector; by default the same
    builder every checkpoint in the project is loaded with.
    """
    from safetensors.torch import load_file

    from src.utils.device import resolve_device

    folder = Path(folder)
    check_sums(folder)
    config = json.loads((folder / CONFIG).read_text(encoding="utf-8"))
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unknown package format {config.get('format_version')!r}")
    resolved = resolve_device(device)
    state = {"model": load_file(str(folder / WEIGHTS)), "config": config["training"]}
    if build is None:
        from src.training.evaluate import _restore_lora, build_detector

        model = build_detector(config["encoder"], state)
        _restore_lora(model, state)
    else:
        model = build(config["encoder"], state)
    model.load_state_dict(state["model"])
    return model.to(resolved).eval(), resolved


def verify(checkpoint: str, package: Path, windows: int = 3, seed: int = 0) -> float:
    """Largest score difference between the checkpoint and its package on random windows."""
    import numpy as np

    from src.inference.predict import load_detector, make_score_fn

    rng = np.random.default_rng(seed)
    audio = [
        (0.05 * rng.standard_normal(int(WINDOW_SECONDS * SAMPLE_RATE))).astype(np.float32)
        for _ in range(windows)
    ]
    original = make_score_fn(*load_detector(checkpoint, device="cpu"))
    packaged = make_score_fn(*load_package(package, device="cpu"))
    return max(abs(original(w) - packaged(w)) for w in audio)


def main() -> None:
    """CLI: ``export`` a checkpoint into a package, or ``verify`` a package against it."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Package the detector for publication")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--threshold-file", default="configs/threshold_final.yaml")
    ex.add_argument("--card", default="docs/MODEL_CARD.md")
    ve = sub.add_parser("verify")
    ve.add_argument("--checkpoint", required=True)
    ve.add_argument("--package", required=True)
    args = parser.parse_args()

    if args.cmd == "export":
        import torch

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        operating_point = yaml.safe_load(Path(args.threshold_file).read_text(encoding="utf-8"))
        card = Path(args.card).read_text(encoding="utf-8")
        config = export(state, Path(args.out), operating_point, card)
        print(f"wrote {args.out}: {config['parameters']:,} parameters")
    else:
        diff = verify(args.checkpoint, Path(args.package))
        print(f"largest score difference, checkpoint vs package: {diff:.2e}")
        if diff > 1e-5:
            raise SystemExit("the package does not reproduce the checkpoint")


if __name__ == "__main__":
    main()
