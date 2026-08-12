"""Tests for the environment verification (Week 3, W3-T6, owner M).

Two things matter here. First, the report is pasted into PRs and the log book, so
**no check may ever print a secret**. Second, the checks a human still has to do
must stay visible as `manual` rather than being quietly counted as passes.

No network, no GPU, no model downloads: the online checks are opt-in and not
exercised here.
"""

import src.utils.env_check as ec


# --------------------------------------------------------------------------- #
# Secrets never leak
# --------------------------------------------------------------------------- #
def test_secret_value_is_never_echoed() -> None:
    secret = "hf_abcdef1234567890SECRETVALUE"
    described = ec.describe_secret(secret)
    assert secret not in described
    assert "SECRETVALUE" not in described


def test_secret_description_reports_length_and_class() -> None:
    assert ec.describe_secret("hf_" + "x" * 30) == "set (33 chars, hf token)"
    assert "github token" in ec.describe_secret("ghp_" + "y" * 36)


def test_unset_secret_is_reported_plainly() -> None:
    assert ec.describe_secret(None) == "not set"
    assert ec.describe_secret("") == "not set"


def test_placeholder_credential_is_called_out() -> None:
    assert "PLACEHOLDER" in ec.describe_secret("REPLACE_WITH_YOUR_TOKEN")


def test_env_var_check_hides_the_value(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_supersecrettokenvalue")
    result = ec.check_env_var("HF_TOKEN")
    assert result.status == ec.OK
    assert "supersecret" not in result.detail


def test_missing_env_var_is_missing_not_error(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert ec.check_env_var("WANDB_API_KEY").status == ec.MISSING


def test_placeholder_env_var_is_an_error(monkeypatch) -> None:
    # A placeholder is worse than absent: it looks configured and is not.
    monkeypatch.setenv("HF_TOKEN", "REPLACE_WITH_HF_TOKEN")
    assert ec.check_env_var("HF_TOKEN").status == ec.ERROR


# --------------------------------------------------------------------------- #
# Package checks
# --------------------------------------------------------------------------- #
def test_installed_package_passes() -> None:
    result = ec.check_package("pandas")
    assert result.status == ec.OK
    assert "version" in result.detail


def test_absent_package_is_missing() -> None:
    assert ec.check_package("definitely_not_a_real_package_xyz").status == ec.MISSING


# --------------------------------------------------------------------------- #
# Manual items stay visible
# --------------------------------------------------------------------------- #
def test_manual_checks_cover_all_three_kaggle_accounts() -> None:
    names = [r.name for r in ec.manual_checks()]
    assert sum("kaggle gpu" in n for n in names) == 3


def test_manual_checks_include_the_gated_datasets() -> None:
    names = " ".join(r.name for r in ec.manual_checks())
    assert "indicvoices" in names
    assert "indictts-deepfake" in names


def test_manual_items_are_not_counted_as_passes() -> None:
    results = ec.manual_checks()
    assert not any(r.passed for r in results)
    assert not any(r.blocking for r in results)


def test_manual_items_are_not_blocking() -> None:
    assert ec.CheckResult("x", ec.MANUAL, "").blocking is False
    assert ec.CheckResult("x", ec.MISSING, "").blocking is True
    assert ec.CheckResult("x", ec.ERROR, "").blocking is True


# --------------------------------------------------------------------------- #
# Summary + report
# --------------------------------------------------------------------------- #
def test_summary_counts_each_status() -> None:
    results = [
        ec.CheckResult("a", ec.OK),
        ec.CheckResult("b", ec.MISSING),
        ec.CheckResult("c", ec.ERROR),
        ec.CheckResult("d", ec.MANUAL),
    ]
    counts = ec.summarise(results)
    assert counts[ec.OK] == 1
    assert counts["blocking"] == 2


def test_report_always_names_the_outstanding_manual_work() -> None:
    report = ec.render(ec.manual_checks())
    assert "need a human to confirm" in report


def test_report_flags_blocking_issues() -> None:
    report = ec.render([ec.CheckResult("HF_TOKEN", ec.MISSING, "not set")])
    assert "blocking issue" in report


def test_run_all_stays_offline_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    names = [r.name for r in ec.run_all()]
    assert "hf identity" not in names  # opt-in only
    assert any("kaggle gpu" in n for n in names)
