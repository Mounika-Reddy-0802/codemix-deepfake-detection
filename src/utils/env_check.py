"""Verify the training/serving environment before a week's work depends on it.

Week 3 (W3-T6, owner M). Three Kaggle accounts, a W&B project, a Hugging Face
token and two gated datasets have to be working before Week 4 generation and
Week 5 training start. Half of that is checkable from code and half is not, and
the useful thing is to be honest about which half is which:

- ``ok`` / ``missing`` / ``error`` -- the machine checked it,
- ``manual`` -- only a human can confirm it (GPU quota on an account this process
  is not logged into, dataset terms accepted in a browser).

A check never prints a secret. Token presence is reported as a length and a
prefix class, never as a value, so the output of this module is safe to paste
into a PR or a log book.

``notebooks/env_check.ipynb`` calls into here rather than duplicating the logic --
notebooks hold no pipeline logic (team rules doc, section 9).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

OK = "ok"
MISSING = "missing"
ERROR = "error"
MANUAL = "manual"

#: Runtime secrets the pipeline expects in .env / Kaggle Secrets.
REQUIRED_ENV_VARS = ("WANDB_API_KEY", "HF_TOKEN")

#: Packages the training and serving paths import.
REQUIRED_PACKAGES = (
    "torch",
    "torchaudio",
    "transformers",
    "librosa",
    "soundfile",
    "pandas",
    "numpy",
)

#: The SSL encoders the three systems are built on.
ENCODER_REPOS = (
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base",
    "facebook/wav2vec2-large-xlsr-53",
)

#: Things no script can confirm from inside one process.
MANUAL_CHECKS = (
    ("kaggle gpu account 1", "30 hrs/week GPU quota confirmed in the Kaggle UI"),
    ("kaggle gpu account 2", "30 hrs/week GPU quota confirmed in the Kaggle UI"),
    ("kaggle gpu account 3", "30 hrs/week GPU quota confirmed in the Kaggle UI"),
    ("hf gated: indicvoices", "dataset terms accepted on the HF page for this token"),
    ("hf gated: indictts-deepfake", "dataset terms accepted on the HF page"),
    ("wandb project created", "project visible at wandb.ai under the team account"),
    ("asvspoof 2019 la downloaded", "S1 anchor corpus present and checksummed"),
    ("colab gpu runtime", "GPU runtime confirmed in a Colab notebook"),
)


@dataclass(frozen=True)
class CheckResult:
    """One environment check and what it found."""

    name: str
    status: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == OK

    @property
    def blocking(self) -> bool:
        """Whether this failure stops Week 4/5 work (manual items are not failures)."""
        return self.status in (MISSING, ERROR)


def describe_secret(value: str | None) -> str:
    """Describe a secret without revealing it: length plus a coarse prefix class."""
    if not value:
        return "not set"
    if value.startswith("hf_"):
        kind = "hf token"
    elif value.startswith(("github_pat_", "ghp_")):
        kind = "github token"
    elif value.startswith("REPLACE"):
        kind = "PLACEHOLDER -- not a real credential"
    else:
        kind = "opaque"
    return f"set ({len(value)} chars, {kind})"


def check_env_var(name: str) -> CheckResult:
    """Whether a required environment variable is set (value never printed)."""
    value = os.environ.get(name)
    detail = describe_secret(value)
    if not value:
        return CheckResult(name, MISSING, detail)
    if value.startswith("REPLACE"):
        return CheckResult(name, ERROR, detail)
    return CheckResult(name, OK, detail)


def check_package(name: str) -> CheckResult:
    """Whether a required package imports, and at what version."""
    import importlib

    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        return CheckResult(name, MISSING, str(exc))
    version = getattr(module, "__version__", "unknown")
    return CheckResult(name, OK, f"version {version}")


def check_gpu() -> CheckResult:
    """Whether this process can see a CUDA device."""
    try:
        import torch
    except ImportError:
        return CheckResult("cuda", MISSING, "torch not installed")
    if not torch.cuda.is_available():
        return CheckResult("cuda", MISSING, "no CUDA device visible to this process")
    return CheckResult("cuda", OK, torch.cuda.get_device_name(0))


def check_hf_identity() -> CheckResult:
    """Whether the HF token actually authenticates (network call)."""
    if not os.environ.get("HF_TOKEN"):
        return CheckResult("hf identity", MISSING, "HF_TOKEN not set")
    try:
        from huggingface_hub import HfApi

        user = HfApi().whoami(token=os.environ["HF_TOKEN"])
    except Exception as exc:  # noqa: BLE001 - offline or bad token, both reportable
        return CheckResult("hf identity", ERROR, f"{type(exc).__name__}: {exc}")
    return CheckResult("hf identity", OK, f"authenticated as {user.get('name', '?')}")


def check_encoder(repo: str) -> CheckResult:
    """Whether an SSL encoder loads (downloads weights on first run)."""
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(repo)
    except Exception as exc:  # noqa: BLE001 - missing weights or no network
        return CheckResult(repo, ERROR, f"{type(exc).__name__}: {exc}")
    hidden = getattr(model.config, "hidden_size", "?")
    return CheckResult(repo, OK, f"loaded, hidden_size={hidden}")


def manual_checks() -> list[CheckResult]:
    """The items a human confirms; always reported, never guessed at."""
    return [CheckResult(name, MANUAL, detail) for name, detail in MANUAL_CHECKS]


def run_all(network: bool = False, encoders: bool = False) -> list[CheckResult]:
    """Run every check. ``network``/``encoders`` opt in to the slow online ones."""
    results = [check_env_var(name) for name in REQUIRED_ENV_VARS]
    results += [check_package(name) for name in REQUIRED_PACKAGES]
    results.append(check_gpu())
    if network:
        results.append(check_hf_identity())
    if encoders:
        results += [check_encoder(repo) for repo in ENCODER_REPOS]
    results += manual_checks()
    return results


def summarise(results: list[CheckResult]) -> dict[str, int]:
    """Counts per status, plus how many blocking failures there are."""
    counts = {OK: 0, MISSING: 0, ERROR: 0, MANUAL: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    counts["blocking"] = sum(1 for r in results if r.blocking)
    return counts


def render(results: list[CheckResult]) -> str:
    """Human-readable report. Safe to paste into a PR or a log book."""
    width = max((len(r.name) for r in results), default=10)
    lines = [f"{'CHECK'.ljust(width)}  STATUS    DETAIL", "-" * (width + 30)]
    for result in results:
        lines.append(f"{result.name.ljust(width)}  {result.status.ljust(8)}  {result.detail}")
    counts = summarise(results)
    lines += [
        "",
        f"ok={counts[OK]}  missing={counts[MISSING]}  error={counts[ERROR]}  "
        f"manual={counts[MANUAL]}",
    ]
    if counts["blocking"]:
        lines.append(f"{counts['blocking']} blocking issue(s) -- fix before Week 4 generation")
    lines.append(f"{counts[MANUAL]} item(s) still need a human to confirm")
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m src.utils.env_check [--network] [--encoders]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify the project environment")
    parser.add_argument("--network", action="store_true", help="check HF authentication")
    parser.add_argument("--encoders", action="store_true", help="load the three SSL encoders")
    args = parser.parse_args()

    print(render(run_all(network=args.network, encoders=args.encoders)))


if __name__ == "__main__":
    main()
