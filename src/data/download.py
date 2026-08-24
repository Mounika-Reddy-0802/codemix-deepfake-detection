"""Hugging Face streaming setup for the eval-only Indic corpora.

IndicSynth, IndicTTS-Deepfake and IndicVoices are **evaluation-only** corpora. Per
the golden dataset rule (see the team rules doc) they are:

- never bulk-downloaded (we only ever *stream* them), and
- never placed in a training manifest.

Streaming (``datasets.load_dataset(..., streaming=True)``) pulls examples on demand
without materialising the whole corpus on disk, which is exactly what the free-tier
storage budget needs. Every loader in this module hard-codes ``streaming=True`` --
there is deliberately no non-streaming code path here.

Heavy imports (``datasets``) happen lazily inside functions so that importing this
module stays cheap and side-effect free (CI imports it without the ML stack).

Auth: gated datasets need a Hugging Face token. Set ``HF_TOKEN`` (or
``HUGGING_FACE_HUB_TOKEN``) in ``.env`` and accept each dataset's terms on its HF
page first. IndicVoices (AI4Bharat) is gated and will 401 until access is granted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    from datasets import IterableDataset

# Environment variables that may hold a HF read token, in priority order.
_HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Marker so the eval-only role travels with the data description everywhere.
EVAL_ONLY = "eval-only"


@dataclass(frozen=True)
class StreamingSource:
    """A Hugging Face corpus we access by streaming only (never full download)."""

    repo_id: str
    licence: str
    #: human label -> HF config name. Empty string means "no config; filter by a
    #: column instead" (verify the actual schema on the dataset's HF page).
    configs: dict[str, str] = field(default_factory=dict)
    role: str = EVAL_ONLY
    notes: str = ""


# --- the three eval-only corpora (verified 21 Jul 2026) ----------------------
INDICSYNTH = StreamingSource(
    repo_id="vdivyasharma/IndicSynth",
    licence="CC-BY-NC-4.0",
    configs={"hindi": "hi", "tamil": "ta"},
    notes=(
        "Monolingual Indic spoof column. NC licence = research use only; note it "
        "in the ethics table. Stream Hindi (~206k) + Tamil (~282k) subsets only, "
        "never the full 4,000 hrs. Verify whether 'hi'/'ta' are HF configs or a "
        "'language' column filter on the dataset page."
    ),
)

INDICTTS_DEEPFAKE = StreamingSource(
    repo_id="SherryT997/IndicTTS-Deepfake-Challenge-Data",
    licence="Research",
    configs={"hindi": "hi", "tamil": "ta", "english": "en"},
    notes=(
        "Monolingual Indic real+fake column. Its public wav2vec2 baselines double "
        "as a sanity comparison."
    ),
)

INDICVOICES = StreamingSource(
    repo_id="ai4bharat/indicvoices",
    licence="Research (AI4Bharat)",
    configs={"tamil": "tamil", "code_switched": ""},
    notes=(
        "Tamil bonafide column + supplementary code-mixed bonafide. GATED: accept "
        "terms on the HF page and use a token. Stream filtered subsets only, never "
        "the full 7,348 hrs."
    ),
)

ALL_SOURCES: dict[str, StreamingSource] = {
    "indicsynth": INDICSYNTH,
    "indictts_deepfake": INDICTTS_DEEPFAKE,
    "indicvoices": INDICVOICES,
}


def hf_token() -> str | None:
    """Return the first HF token found in the environment, or ``None``."""
    for name in _HF_TOKEN_ENVS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _load_streaming(
    repo_id: str,
    *,
    config: str | None = None,
    split: str = "train",
    token: str | None = None,
) -> IterableDataset:
    """Load a HF dataset in streaming mode. ``streaming=True`` is not optional.

    Args:
        repo_id: Hugging Face dataset repo id.
        config: dataset config name, or ``None`` for the default config.
        split: split name (default ``"train"``).
        token: HF token; falls back to the environment when ``None``.

    Returns:
        A streaming ``IterableDataset`` -- examples are fetched on demand.
    """
    from datasets import load_dataset  # lazy: keep module import cheap/CI-safe

    return load_dataset(
        repo_id,
        name=config or None,
        split=split,
        streaming=True,
        token=token or hf_token(),
    )


def stream_indicsynth(language: str, split: str = "train") -> IterableDataset:
    """Stream the IndicSynth Hindi or Tamil subset (eval-only spoof column)."""
    return _stream_from(INDICSYNTH, language, split)


def stream_indictts_deepfake(language: str, split: str = "train") -> IterableDataset:
    """Stream an IndicTTS-Deepfake language subset (eval-only real+fake column)."""
    return _stream_from(INDICTTS_DEEPFAKE, language, split)


def stream_indicvoices(subset: str = "tamil", split: str = "train") -> IterableDataset:
    """Stream an IndicVoices subset (eval-only Tamil / code-mixed bonafide)."""
    return _stream_from(INDICVOICES, subset, split)


def _stream_from(source: StreamingSource, key: str, split: str) -> IterableDataset:
    if key not in source.configs:
        raise KeyError(
            f"{source.repo_id}: unknown subset '{key}'. " f"Known: {sorted(source.configs)}"
        )
    config = source.configs[key] or None
    return _load_streaming(source.repo_id, config=config, split=split)


def preview(dataset: IterableDataset, n: int = 3) -> list[dict[str, Any]]:
    """Materialise the first ``n`` streamed examples (for a quick smoke check)."""
    out: list[dict[str, Any]] = []
    for i, example in enumerate(dataset):
        if i >= n:
            break
        out.append(example)
    return out


def describe_sources() -> str:
    """Return a human-readable summary of every eval-only streaming source."""
    lines = ["Eval-only streaming corpora (never trained on, never bulk-downloaded):"]
    for name, src in ALL_SOURCES.items():
        subsets = ", ".join(src.configs) or "(default)"
        lines.append(f"  - {name}: {src.repo_id} [{src.licence}] subsets: {subsets}")
    return "\n".join(lines)


def main() -> None:
    """CLI: print the configured sources; ``--smoke <name> <subset>`` streams 1 row."""
    import argparse

    parser = argparse.ArgumentParser(description="Eval-only HF streaming setup")
    parser.add_argument(
        "--smoke",
        nargs=2,
        metavar=("SOURCE", "SUBSET"),
        help="stream a single example (needs network + HF token)",
    )
    args = parser.parse_args()

    print(describe_sources())
    if not args.smoke:
        return

    source_name, subset = args.smoke
    if source_name not in ALL_SOURCES:
        raise SystemExit(f"unknown source: {source_name} (known: {sorted(ALL_SOURCES)})")
    ds = _stream_from(ALL_SOURCES[source_name], subset, split="train")
    rows = preview(ds, n=1)
    keys = sorted(rows[0].keys()) if rows else []
    print(f"streamed 1 example from {source_name}/{subset}; feature keys: {keys}")


if __name__ == "__main__":
    main()
