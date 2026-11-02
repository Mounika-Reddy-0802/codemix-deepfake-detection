"""English routing: send English speech to the English-only detector, the rest on.

The adapters were trained for Hindi-English speech, and on English they are much worse
than S1, the English-only detector: over the G.711 line the final adapter reaches 25.95%
EER on English and S1 6.50% (``docs/results/english_final_v1.md``). A deployment that
also takes English calls can put a language gate in front: English goes to S1, anything
else to the Hindi-English detector.

The gate is a spoken-language identifier (SpeechBrain's VoxLingua107 ECAPA model, 107
languages, run on CPU). None of the candidate rules has a parameter to tune; the rule
is chosen among them on development audio only (:func:`select_rule`), before any test
output is read:

- ``top_english``: S1 when the most likely language is English (the first rule);
- ``english_over_hindi``: S1 when P(English) > P(Hindi);
- ``not_indian``: the Hindi-English detector when the most likely language is one of
  the languages of India and its neighbours, S1 for everything else.

The criterion, written down before the development results were read: the highest
worst-case routing accuracy over the development sets (English clean, English over
G.711, Hindi-English over G.711); a tie goes to the earlier, simpler rule. The gate
reads exactly the audio the detectors score: the same loader, the same first 4 s.

The hard case is English spoken by Hindi-English speakers (the language-control set):
the gate sends it to S1, which rejects most of those speakers' genuine English
(``docs/results/language_control_v1.md``). It is reported, not hidden.

Usage::

    python -m src.inference.routing lid --manifest M.csv --data-root DIR --out lid.csv
    python -m src.inference.routing lid --manifest data/manifests/asvspoof_dev.csv \\
        --sample 400 --render g711 --out lid_english_dev_g711.csv

``lid`` is resumable: clips already in ``--out`` are skipped. Torch and SpeechBrain are
imported lazily so the routing rule itself stays importable in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

LID_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
ENGLISH = "en"
HINDI = "hi"
MAX_SECONDS = 4.0
SAMPLE_RATE = 16_000
SAMPLE_SEED = 1234
FLUSH_EVERY = 200
S1_CHECKPOINT = "checkpoints/baseline/best.pt"


def language_code(label: str) -> str:
    """``"en: English"`` -> ``"en"`` (the model's labels carry the name after a colon)."""
    return str(label).split(":", 1)[0].strip()


#: VoxLingua107 codes of the languages of India and its neighbours
INDIAN = frozenset(
    {"as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "pa", "sa", "si", "ta", "te", "ur"}
)

#: candidate rules, simplest first: True means the clip goes to S1
RULES = {
    "top_english": lambda lid: lid["language"].astype(str) == ENGLISH,
    "english_over_hindi": lambda lid: lid["p_en"].astype(float) > lid["p_hi"].astype(float),
    "not_indian": lambda lid: ~lid["language"].astype(str).isin(INDIAN),
}


def is_english(language: str) -> bool:
    """The first rule for a single clip: English goes to S1."""
    return language == ENGLISH


def to_s1(lid: pd.DataFrame, rule: str) -> pd.Series:
    """Which clips a rule sends to S1 (``lid`` has language, p_en and p_hi)."""
    return RULES[rule](lid).astype(bool)


def route(goes_to_s1: pd.Series, english_scores: pd.Series, other_scores: pd.Series) -> pd.Series:
    """The routed system's score: S1's where the gate chose S1, the other model's else."""
    return english_scores.where(goes_to_s1.astype(bool), other_scores)


def select_rule(dev: dict[str, tuple[pd.DataFrame, bool]]) -> tuple[str, pd.DataFrame]:
    """Choose the rule on development sets only.

    ``dev`` maps a set name to its gate output and whether its clips should reach S1.
    Returns the chosen rule and the accuracy table (percent routed correctly).
    """
    table = pd.DataFrame(
        {
            name: {rule: float((to_s1(lid, rule) == wanted).mean()) * 100 for rule in RULES}
            for name, (lid, wanted) in dev.items()
        }
    )
    worst = table.min(axis=1)
    chosen = next(rule for rule in RULES if worst[rule] == worst.max())
    return chosen, table.assign(worst=worst)


class LanguageGate:
    """Spoken-language identification on CPU, loaded on first use."""

    def __init__(self, savedir: str | None = None, threads: int = 4) -> None:
        self.savedir = savedir or str(Path.home() / ".cache/speechbrain" / LID_SOURCE.split("/")[1])
        self.threads = threads
        self._model = None
        self._labels: list[str] = []

    def _load(self) -> None:
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier

        torch.set_num_threads(self.threads)
        self._model = EncoderClassifier.from_hparams(
            source=LID_SOURCE, savedir=self.savedir, run_opts={"device": "cpu"}
        )
        encoder = self._model.hparams.label_encoder
        self._labels = [language_code(x) for x in encoder.decode_ndim(list(range(len(encoder))))]

    def classify(self, waveform: np.ndarray) -> dict[str, float | str]:
        """Most likely language and the posteriors of English and Hindi for one clip."""
        import torch

        if self._model is None:
            self._load()
        with torch.no_grad():
            wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
            log_post = self._model.classify_batch(wav)[0].squeeze(0)
        post = torch.softmax(log_post, dim=-1).numpy()
        return {
            "language": self._labels[int(post.argmax())],
            "p_en": float(post[self._labels.index(ENGLISH)]),
            "p_hi": float(post[self._labels.index(HINDI)]),
        }


def sample_manifest(manifest: pd.DataFrame, n: int, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """A fixed sample, balanced between genuine and fake clips when both exist."""
    per_label = max(1, n // manifest["label"].nunique())
    parts = [
        group.sample(min(per_label, len(group)), random_state=seed)
        for _, group in manifest.groupby("label", sort=True)
    ]
    return pd.concat(parts).reset_index(drop=True)


def render_g711(waveform: np.ndarray, index: int) -> np.ndarray:
    """The project's simulated phone line (G.711 at 20 dB), seeded per clip."""
    from src.data.channel_sim import ChannelConfig, simulate_channel

    return simulate_channel(waveform, ChannelConfig(codec="g711", snr_db=20.0, seed=index))


def identify(
    manifest: pd.DataFrame,
    out_csv: str | Path,
    data_root: str | None = None,
    render: str | None = None,
    gate: LanguageGate | None = None,
) -> pd.DataFrame:
    """Language of every clip's first 4 s, written to ``out_csv`` (resumable)."""
    from src.data.dataset import AudioManifestDataset
    from src.utils.audio_utils import rms_normalize

    out_csv = Path(out_csv)
    done = pd.read_csv(out_csv) if out_csv.is_file() else pd.DataFrame(columns=["filepath"])
    seen = set(done["filepath"].astype(str))
    dataset = AudioManifestDataset(
        manifest, data_root=data_root, max_seconds=MAX_SECONDS, random_crop=False
    )
    gate = gate or LanguageGate()
    rows: list[dict] = []
    for i in range(len(dataset)):
        filepath = str(dataset.df.iloc[i]["filepath"])
        if filepath in seen:
            continue
        waveform = dataset[i].waveform
        if render == "g711":
            # the bundles are level-normalised before the line; do the same here
            waveform = render_g711(rms_normalize(waveform, -23.0), i)
        rows.append({"filepath": filepath, **gate.classify(waveform)})
        if len(rows) % FLUSH_EVERY == 0:
            done = _flush(done, rows, out_csv)
            rows = []
            print(f"  {len(done)}/{len(dataset)} clips", flush=True)
    return _flush(done, rows, out_csv)


def _flush(done: pd.DataFrame, rows: list[dict], out_csv: Path) -> pd.DataFrame:
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".part")
    done.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    return done


def confusion(lid: pd.DataFrame, rule: str = "top_english") -> dict[str, float]:
    """Share of clips a rule sends to S1, and the most common languages the gate hears."""
    languages = lid["language"].astype(str)
    return {
        "clips": int(len(lid)),
        "to_s1_percent": round(float(to_s1(lid, rule).mean()) * 100, 2),
        "top_languages": {k: int(v) for k, v in languages.value_counts().head(5).items()},
    }


def compare(sample: pd.DataFrame, rule: str) -> dict[str, dict]:
    """S1 alone, the Hindi-English detector alone, and the routed system on one sample.

    ``sample`` has ``label``, ``language`` (the gate's decision), ``s1`` and ``other``
    (each model's P(genuine) for the clip). Every system is scored on the same clips.
    """
    from src.reporting.det_curves import clip_metrics

    labels = (sample["label"].astype(str) == "bonafide").to_numpy().astype(int)
    goes = to_s1(sample, rule)
    systems = {
        "s1": sample["s1"],
        "other": sample["other"],
        "routed": route(goes, sample["s1"], sample["other"]),
    }
    out = {
        name: clip_metrics(scores.to_numpy(dtype=float), labels) for name, scores in systems.items()
    }
    out["gate"] = confusion(sample, rule)
    return out


def assemble(
    lid: pd.DataFrame, s1_scores: pd.DataFrame, other_scores: pd.DataFrame
) -> pd.DataFrame:
    """One row per clip the gate saw: label, gate output, S1's and the other model's score.

    Clips are joined on the file name, so score files written with different data roots
    still line up; a clip missing from either score file is an error, not a silent drop.
    """

    def keyed(frame: pd.DataFrame, score: str | None = None) -> pd.DataFrame:
        out = frame.assign(
            clip=frame["filepath"].astype(str).str.replace("\\", "/").str.rsplit("/", n=1).str[-1]
        )
        return out.rename(columns={"score": score}) if score else out

    sample = keyed(lid).merge(keyed(other_scores, "other")[["clip", "label", "other"]], on="clip")
    sample = sample.merge(keyed(s1_scores, "s1")[["clip", "s1"]], on="clip")
    if len(sample) != len(lid):
        raise ValueError(f"{len(lid) - len(sample)} gate rows have no score in one of the files")
    return sample


def score_s1(
    manifest: pd.DataFrame, data_root: str, out_csv: str | Path, checkpoint: str = S1_CHECKPOINT
) -> pd.DataFrame:
    """S1's P(genuine) for every clip of ``manifest`` on CPU, resumable through a partial file."""
    from src.training.evaluate import score_manifest

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = manifest[["filepath", "label"]].reset_index(drop=True)
    tmp_manifest = out_csv.with_suffix(".manifest.csv")
    rows.to_csv(tmp_manifest, index=False)
    scored = score_manifest(
        checkpoint,
        str(tmp_manifest),
        batch_size=8,
        max_seconds=MAX_SECONDS,
        device="cpu",
        data_root=data_root,
        partial_path=str(out_csv.with_suffix(".partial.csv")),
    )
    scored[["filepath", "label", "score"]].to_csv(out_csv, index=False)
    return scored


def main() -> None:
    """CLI: ``lid`` writes the gate's decisions; ``s1`` scores the same clips with S1."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="English routing gate")
    sub = parser.add_subparsers(dest="command", required=True)
    lid = sub.add_parser("lid", help="identify the language of every clip")
    lid.add_argument("--manifest", required=True)
    lid.add_argument("--data-root", default=None)
    lid.add_argument("--out", required=True)
    lid.add_argument("--sample", type=int, default=None, help="fixed balanced sample of N")
    lid.add_argument("--render", choices=["g711"], default=None)
    lid.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    s1 = sub.add_parser("s1", help="S1's scores for the clips a gate file lists")
    s1.add_argument("--lid", required=True, help="output of the lid command")
    s1.add_argument("--manifest", required=True, help="manifest with each clip's label")
    s1.add_argument("--data-root", required=True)
    s1.add_argument("--out", required=True)
    sel = sub.add_parser("select", help="choose the rule on development gate outputs")
    sel.add_argument(
        "--dev",
        nargs="+",
        required=True,
        help="NAME=PATH=s1|other: a dev gate output and where its clips should go",
    )
    sel.add_argument("--out", required=True)
    rep = sub.add_parser("report", help="compare S1, the other model and the routed system")
    rep.add_argument("--rule", required=True, choices=sorted(RULES))
    rep.add_argument(
        "--set",
        nargs="+",
        required=True,
        help="NAME=LID=S1_SCORES=OTHER_SCORES for each test set",
    )
    rep.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.command == "report":
        results = {}
        for item in args.set:
            name, lid_path, s1_path, other_path = item.split("=")
            sample = assemble(pd.read_csv(lid_path), pd.read_csv(s1_path), pd.read_csv(other_path))
            results[name] = compare(sample, args.rule)
            print(
                name,
                json.dumps({k: results[name][k].get("eer") for k in ("s1", "other", "routed")}),
            )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"rule": args.rule, "sets": results}, indent=2), encoding="utf-8"
        )
        return

    if args.command == "select":
        dev = {}
        for item in args.dev:
            name, path, wanted = item.split("=")
            dev[name] = (pd.read_csv(path), wanted == "s1")
        chosen, table = select_rule(dev)
        record = {"chosen": chosen, "percent_routed_correctly": table.round(2).to_dict()}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(table.round(2).to_string())
        print("chosen:", chosen)
        return

    if args.command == "s1":
        lid_rows = pd.read_csv(args.lid)
        labels = pd.read_csv(args.manifest, usecols=["filepath", "label"])
        score_s1(lid_rows[["filepath"]].merge(labels, on="filepath"), args.data_root, args.out)
        return
    manifest = pd.read_csv(args.manifest)
    if args.sample:
        manifest = sample_manifest(manifest, args.sample)
    result = identify(
        manifest, args.out, args.data_root, args.render, LanguageGate(threads=args.threads)
    )
    print(json.dumps(confusion(result), indent=2))


if __name__ == "__main__":
    main()
