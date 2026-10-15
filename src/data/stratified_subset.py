"""A fixed, attack-balanced sample of a scoring manifest.

External review comment 4 asks for the trade-off between Hinglish phone-line
detection and English performance. English has only ever been scored on clean audio,
on all 71,237 ASVspoof 2019 LA evaluation clips. Rendering and scoring all of them
through the telephone channel on a CPU would take most of a day, so the phone-line
comparison is made on a sample drawn once, before any score is seen: a fixed number
of genuine clips and the same number of clips from every attack, with a fixed seed.
Equal attack shares keep a few large attack families from dominating the EER; the
sample is reported as a sample, beside the full-set clean numbers.
"""

from __future__ import annotations

import pandas as pd


def stratified_subset(
    manifest: pd.DataFrame,
    n_bonafide: int = 2000,
    n_per_attack: int = 308,
    attack_column: str = "tool",
    seed: int = 1234,
) -> pd.DataFrame:
    """``n_bonafide`` genuine rows plus ``n_per_attack`` rows from every attack.

    An attack with fewer rows than asked contributes all it has. Order is shuffled
    with the same seed so a truncated run still sees every attack.
    """
    labels = manifest["label"].astype(str).str.lower()
    genuine = manifest[labels == "bonafide"]
    spoof = manifest[labels != "bonafide"]
    if genuine.empty or spoof.empty:
        raise ValueError("the manifest needs both genuine and spoofed rows")
    parts = [genuine.sample(n=min(n_bonafide, len(genuine)), random_state=seed)]
    for _, group in spoof.groupby(spoof[attack_column].astype(str), sort=True):
        parts.append(group.sample(n=min(n_per_attack, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    """CLI: ``python -m src.data.stratified_subset --manifest in.csv --out out.csv``."""
    import argparse

    parser = argparse.ArgumentParser(description="Attack-balanced fixed sample of a manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-bonafide", type=int, default=2000)
    parser.add_argument("--n-per-attack", type=int, default=308)
    parser.add_argument("--attack-column", default="tool")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    subset = stratified_subset(
        pd.read_csv(args.manifest),
        args.n_bonafide,
        args.n_per_attack,
        args.attack_column,
        args.seed,
    )
    subset.to_csv(args.out, index=False)
    counts = subset.groupby(subset[args.attack_column].astype(str)).size()
    print(f"wrote {args.out}: {len(subset)} rows, {subset['speaker'].nunique()} speakers")
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
