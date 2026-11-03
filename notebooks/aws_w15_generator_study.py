"""Generator study (P-041) on one AWS GPU machine: train, score, save to S3.

The Kaggle quota was spent until 7 November, so the study runs on AWS (us-east-1, one
g5.2xlarge, A10G). This is the week-15 variant notebook's code with the four study systems,
Chatterbox (CM09) added to the scored sets, and the run pinned to the registration commit
(the hypotheses and decision rule were committed before it). The bootstrap places the Kaggle
datasets under /kaggle/input exactly as Kaggle mounts them, so the setup is unchanged:
codemix-w13-core, codemix-extension-bundle, codemix-week13-clones, codemix-w13-amr,
codemix-w13-amr2 and codemix-w13-cm09.

Results are copied to ``s3://$GENSTUDY_BUCKET`` after every run; afterwards::

    python -m src.reporting.generator_study --results <unzipped final_results>
"""

# ---- 1. settings
import os

SEEDS = [1234, 2025, 7, 42, 99]
SYSTEMS = {  # name -> config (P-041; all four committed with the registration)
    "control": "configs/train_lora_genstudy_control.yaml",
    "fold_a": "configs/train_lora_genstudy_fold_a.yaml",
    "fold_b": "configs/train_lora_genstudy_fold_b.yaml",
    "fold_c": "configs/train_lora_genstudy_fold_c.yaml",
}
EVALS = {  # EER sets, all over the phone line
    "eval_pool": "data/manifests/codemix_eval_channel20.csv",
    "cm04": "data/manifests/score_cm04_norm_channel20.csv",
    "rvc_test": "data/manifests/rvc_holdout_test_channel20.csv",
    "new_speakers": "data/manifests/score_ext_channel20.csv",
    "cm07": "data/manifests/score_cm07_channel20.csv",
    "new_speakers_amr": "data/manifests/score_ext_amr.csv",
    "cm07_amr": "data/manifests/score_cm07_amr.csv",
    "cm04_amr": "data/manifests/score_cm04_norm_amr.csv",
    "cm09": "data/manifests/score_cm09_channel20.csv",  # Chatterbox
}
CONTROL = "data/manifests/mucs_language_control_channel20.csv"  # genuine only: false alarms
ZIP_NAME = "genstudy_results.zip"
REPO_HOST = "github.com/Mounika-Reddy-0802/codemix-deepfake-detection.git"
BRANCH = "week15-krishna-real-calls-and-training"
PIN = "a66b2a862cf6503bc7e32611e8e1caec588e472c"  # the registration commit (rule before results)
BUCKET = os.environ["GENSTUDY_BUCKET"]  # results are copied here after every run
SCORE_ARGS = "--device cuda --batch-size 32 --num-workers 4 --max-seconds 4.0"


# ---- 2. setup
import glob, hashlib, json, os, re, shutil, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working")    # the saved output: the zip and two checkpoints only
TEMP = Path("/kaggle/temp")
try:
    TEMP.mkdir(parents=True, exist_ok=True)
    (TEMP / ".writable").write_text("ok")
except OSError:
    TEMP = Path("/tmp/w13")
    TEMP.mkdir(parents=True, exist_ok=True)
REPO = TEMP / "codemix-deepfake-detection"
OUT = TEMP / "final_results"
ROOT = TEMP / "root_channel"      # both datasets' phone-line clips, linked into one folder
# no run starts after 13 h; the machine also powers itself off at 15 h whatever happens
DEADLINE = time.time() + 13 * 3600
FAILURES = []


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True,
                       cwd=REPO if REPO.exists() else None)
    if r.stdout.strip():
        print(r.stdout[-2500:])
    if r.returncode and r.stderr.strip():
        print(r.stderr[-2500:])
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")
    return r


def run_logged(args):
    """Run a long command, streaming its output, and return all of it."""
    args = [str(a) for a in args]
    print("$", " ".join(args), flush=True)
    proc = subprocess.Popen(args, text=True, errors="replace", cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(args)}\n")
        for line in proc.stdout:
            lines.append(line)
            log.write(line)
            if not any(n in line for n in ("Warning", "warn(", "Some weights", "You should")):
                print(line, end="", flush=True)
    proc.wait()
    return proc.returncode, "".join(lines)


def save_progress():
    """Zip the results so far (never checkpoints) into /kaggle/working."""
    import zipfile
    staging = WORK / (ZIP_NAME + ".tmp")
    with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.suffix != ".pt":
                zf.write(f, f.relative_to(OUT.parent))
    staging.replace(WORK / ZIP_NAME)


def tail(text, n=30):
    return "\n".join([line for line in text.splitlines() if line.strip()][-n:])


def failed(where, detail):
    FAILURES.append({"where": where, "detail": detail})
    print(f"\n!!!!!! FAILED: {where}\n{detail}\n!!!!!! recorded; the run continues\n", flush=True)
    try:
        (OUT / "FAILURES.txt").write_text(
            "\n\n".join(f"== {f['where']}\n{f['detail']}" for f in FAILURES), encoding="utf-8")
        save_progress()
    except Exception as e:
        print("(could not write FAILURES.txt:", e, ")")


def _make_writable(func, path, _):
    os.chmod(path, 0o700)
    func(path)


os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
for attempt in range(1, 4):
    if REPO.exists():
        shutil.rmtree(REPO, onerror=_make_writable)
    if subprocess.run(["git", "clone", "--branch", BRANCH,
                       f"https://{REPO_HOST}", str(REPO)]).returncode == 0 and \
            subprocess.run(["git", "checkout", "-q", PIN], cwd=REPO).returncode == 0:
        break
    print(f"clone attempt {attempt} failed; retrying in 20 s", flush=True)
    time.sleep(20)
else:
    raise SystemExit("Could not clone the repository after 3 attempts. Is Internet switched On?")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                        capture_output=True).stdout.strip()
if COMMIT != PIN:
    raise SystemExit(f"checked out {COMMIT}, not the registration commit {PIN}")
print("running the registration commit", COMMIT)

BASELINE = REPO / "checkpoints" / "baseline" / "best.pt"
pointer = subprocess.run(["git", "cat-file", "-p", "HEAD:checkpoints/baseline/best.pt"],
                         cwd=REPO, text=True, capture_output=True).stdout
want = re.search(r"oid sha256:([0-9a-f]{64})", pointer).group(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


source = None
for candidate in glob.glob("/kaggle/input/**/best.pt", recursive=True):  # hash-checked
    if os.path.getsize(candidate) > 100_000_000 and sha256(candidate) == want:
        shutil.copy(candidate, BASELINE)
        source = candidate
        break
if source is None:
    print("Stage-1 checkpoint not in the inputs; fetching it with git lfs ...")
    sh("git lfs install --local && git lfs pull --include checkpoints/baseline/best.pt", check=False)
    source = "git lfs"
if not BASELINE.is_file() or BASELINE.stat().st_size < 100_000_000 or sha256(BASELINE) != want:
    raise SystemExit("The Stage-1 checkpoint could not be obtained. Attach "
                     "saikrishnareddy9/codemix-w13-core and run again.")
print(f"Stage-1 checkpoint OK (sha256 verified) from {source}")
sh(f"{sys.executable} -m pip install -q 'transformers>=4.57,<5' peft==0.13.2 soundfile==0.12.1 "
   "librosa==0.11.0 pandas pyyaml scikit-learn pytest", check=False)

# One data root: every phone-line clip of the three datasets, linked (not copied) into ROOT/clips.
folders = [p for p in glob.glob("/kaggle/input/**/channel/clips", recursive=True) if os.listdir(p)]
# The week-13 datasets hold only phone-line clips; Kaggle may unpack them as clips/ at
# the dataset's top level rather than channel/clips.
folders += [p for p in glob.glob("/kaggle/input/**/clips", recursive=True)
            if ("week13-clones" in p or "w13-core" in p or "w13-cm09" in p) and "clean" not in p
            and p not in folders and os.listdir(p)]
print("phone-line clip folders:", folders)
if len(folders) < 3:
    raise SystemExit("Attach saikrishnareddy9/codemix-w13-core, "
                     "saikrishnareddy9/codemix-extension-bundle and "
                     "saikrishnareddy9/codemix-week13-clones (Add Input).")
shutil.rmtree(ROOT, ignore_errors=True)
(ROOT / "clips").mkdir(parents=True)
linked = same_name = 0
for folder in folders:
    for name in os.listdir(folder):
        target = ROOT / "clips" / name
        if target.exists():
            same_name += 1
            continue
        os.symlink(os.path.join(folder, name), target)
        linked += 1
print(f"data root {ROOT}: {linked} clips linked ({same_name} names in both datasets, first kept)")

# The AMR-NB copies keep their clips' names, so they go in their own folder: clips/amr.
amr_dirs = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/*.wav", recursive=True)
                   if "w13-amr" in p}, key=lambda d: -len(os.listdir(d)))
if not amr_dirs:
    raise SystemExit("Attach saikrishnareddy9/codemix-w13-amr (Add Input).")
(ROOT / "clips" / "amr").mkdir()
for d in amr_dirs:  # codemix-w13-amr and codemix-w13-amr2, one folder of clips each
    for name in os.listdir(d):
        target = ROOT / "clips" / "amr" / name
        if not target.exists():
            os.symlink(os.path.join(d, name), target)
print(f"AMR-NB clips: {len(os.listdir(ROOT / 'clips' / 'amr'))} from {amr_dirs}")

import pandas as pd
import yaml
from src.data.scoring_manifests import missing_files


manifests = set(EVALS.values()) | {CONTROL}
for config in SYSTEMS.values():
    cfg = yaml.safe_load((REPO / config).read_text())
    manifests |= {cfg["train_manifest"], cfg["dev_manifest"]}
for manifest in sorted(manifests):
    gone = missing_files(pd.read_csv(manifest), str(ROOT))
    print(f"  {manifest:58s} missing {len(gone)}")
    if gone:
        raise SystemExit(f"{manifest} has clips in no attached dataset, e.g. {gone[:3]} -- "
                         "are all three data inputs attached?")
print("every clip every run needs is present")
sh(f"{sys.executable} -m pytest tests/test_splits.py tests/test_config_guard.py "
   "tests/test_final_recipe.py tests/test_codec_mix.py tests/test_generator_study.py "
   "tests/test_generator_study_report.py -q")
for config in SYSTEMS.values():
    sh(f"{sys.executable} -m src.training.config_guard {config}")

# ---- 3. train, score, save
import statistics as st
import zipfile

DEV_LINE = re.compile(r"best dev EER = ([0-9.]+)")
RUNS = []
DONE = False
FIELDS = ["eval_pool", "cm04", "rvc_test", "new_xtts", "new_tortoise", "cm07", "control_fa",
          "new_xtts_amr", "new_tortoise_amr", "cm07_amr", "cm04_amr", "cm09"]


def score(tag, ckpt, manifest):
    """EER JSON of one model on one set, or None (recorded as a failure)."""
    code_, text = run_logged([sys.executable, "-m", "src.training.evaluate", "--checkpoint", ckpt,
                              "--manifest", manifest, "--encoder", "wav2vec2-base",
                              "--data-root", ROOT, *SCORE_ARGS.split(),
                              "--scores-out", OUT / f"{tag}_scores.csv", "--out", OUT / f"{tag}.json"])
    if code_ == 0:
        try:
            return json.loads((OUT / f"{tag}.json").read_text())
        except Exception:
            pass
    failed(f"scoring {tag} (exit {code_})", tail(text))
    return None


def false_alarms(tag, ckpt):
    code_, text = run_logged([sys.executable, "-m", "src.training.false_alarm", "--checkpoint", ckpt,
                              "--manifest", CONTROL, "--encoder", "wav2vec2-base",
                              "--data-root", ROOT, "--device", "cuda", "--batch-size", "32",
                              "--num-workers", "4", "--scores-out", OUT / f"{tag}_scores.csv",
                              "--out", OUT / f"{tag}.json"])
    try:
        return json.loads((OUT / f"{tag}.json").read_text())["summary"]["false_alarm_rate"]
    except Exception:
        failed(f"false alarms {tag} (exit {code_})", tail(text))
        return None


def per_attack(result, attack):
    for row in (result or {}).get("per_attack", []):
        if row["attack"] == attack:
            return row["eer"]
    return None


def summary_text():
    lines = ["STATUS: " + ("FINISHED - every run ran." if DONE else
                           f"IN PROGRESS - {len(RUNS)} run(s) finished so far."),
             "RUN HEALTH: " + ("every step ran without error." if not FAILURES else
                               f"{len(FAILURES)} step(s) FAILED -- see FAILURES.txt"),
             f"commit: {COMMIT}", "",
             "EER (%) per run, phone line; control_fa = genuine clips called fake (%)",
             f"{'run':18s}{'status':>9s}{'dev':>7s}" + "".join(f"{f:>17s}" for f in FIELDS)]
    for r in RUNS:
        dev = "-" if r["dev_eer"] is None else f"{100 * r['dev_eer']:.2f}"
        cells = "".join(f"{'-' if r.get(f) is None else format(100 * r[f], '.2f'):>17s}" for f in FIELDS)
        lines.append(f"{r['name']:18s}{r['status']:>9s}{dev:>7s}{cells}")
    lines.append("\nMEAN +/- SD over seeds (all speakers; the study judges held-out speakers only)")
    for f in FIELDS:
        parts = []
        for system in SYSTEMS:
            v = [100 * r[f] for r in RUNS if r["system"] == system and r.get(f) is not None]
            parts.append(f"{system} " + (f"{st.mean(v):6.2f} +/- {st.stdev(v):5.2f} (n={len(v)})"
                                         if len(v) >= 2 else ("-" if not v else f"{v[0]:6.2f} (n=1)")))
        by_seed = {}
        for r in RUNS:
            if r.get(f) is not None:
                by_seed.setdefault(r["seed"], {})[r["system"]] = 100 * r[f]
        paired = ""  # the paired judgement is src/reporting/generator_study, run afterwards
        lines.append(f"  {f:13s} " + "   ".join(parts) + "   " + paired)
    return "\n".join(lines)


def write_summary():
    try:
        text = summary_text()
    except Exception:
        text = "The summary could not be built:\n" + traceback.format_exc()
    try:
        (OUT / "summary.txt").write_text(text, encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps({
            "mode": "full" if DONE else "in progress", "commit": COMMIT, "branch": BRANCH,
            "seeds": SEEDS, "systems": SYSTEMS, "runs": RUNS, "failures": FAILURES},
            indent=2, default=str))
    except Exception as e:
        print("(could not write the summary:", e, ")", flush=True)
    return text


def to_s3():
    """Copy the results so far, and any kept checkpoint, to the bucket."""
    sh(f"aws s3 cp {WORK / ZIP_NAME} s3://{BUCKET}/{ZIP_NAME} --only-show-errors", check=False)
    sh(f"aws s3 sync {WORK} s3://{BUCKET}/checkpoints --exclude '*' --include '*.pt' "
       "--only-show-errors", check=False)


def run_job(system, seed):
    """Train one system with one seed, score it everywhere, record and save the row."""
    name = f"{system}_s{seed}"
    t0 = time.time()
    row = {"name": name, "system": system, "seed": seed, "dev_eer": None, "status": "failed"}
    try:
        cfg = yaml.safe_load((REPO / SYSTEMS[system]).read_text())
        cfg.update({"seed": int(seed), "out_dir": f"checkpoints/w13/{name}", "wandb_mode": "disabled"})
        cfg_path = REPO / "configs" / f"w13_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        code_, text = run_logged([sys.executable, "-m", "src.training.train", "--config", cfg_path,
                                  "--data-root", ROOT, "--device", "cuda"])
        ckpt = REPO / cfg["out_dir"] / "best.pt"
        if code_ or not ckpt.is_file():
            row["status"] = "collapsed" if "model collapsed" in text else "failed"
            failed(f"training {name} (exit {code_})", tail(text))
            return row
        dev = DEV_LINE.findall(text)
        row["dev_eer"] = float(dev[-1]) if dev else None
        results = {ev: score(f"{name}__{ev}", ckpt, m) for ev, m in EVALS.items()}
        for ev in ("eval_pool", "cm04", "rvc_test", "cm07", "cm07_amr", "cm04_amr", "cm09"):
            row[ev] = (results[ev] or {}).get("pooled", {}).get("eer")
        row["new_xtts"] = per_attack(results["new_speakers"], "xtts_v2")
        row["new_tortoise"] = per_attack(results["new_speakers"], "tortoise")
        row["new_xtts_amr"] = per_attack(results["new_speakers_amr"], "xtts_v2")
        row["new_tortoise_amr"] = per_attack(results["new_speakers_amr"], "tortoise")
        row["control_fa"] = false_alarms(f"{name}__control", ckpt)
        row["status"] = "ok" if all(row.get(f) is not None for f in FIELDS) else "partial"
        if seed == SEEDS[0] or system == "fold_c":  # fold C: the demo model candidates
            shutil.copy(ckpt, WORK / f"{name}_best.pt")
    except Exception:
        failed(f"{name}: unexpected error", traceback.format_exc())
    finally:
        shutil.rmtree(REPO / "checkpoints" / "w13" / name, ignore_errors=True)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        RUNS.append(row)
        (OUT / "runs.json").write_text(json.dumps(RUNS, indent=2))
        write_summary()
        save_progress()
        to_s3()
        shown = "  ".join(f"{f} {100 * row[f]:.2f}" for f in FIELDS if row.get(f) is not None)
        print(f"\n[{name}] {row['status']}  {shown}  ({row['minutes']} min)\n", flush=True)
    return row


try:
    jobs = [(system, seed) for seed in SEEDS for system in SYSTEMS]
    for system, seed in jobs:
        if time.time() > DEADLINE:
            failed("time limit", f"stopped before {system}_s{seed}; finished runs are kept")
            break
        run_job(system, seed)
    else:
        DONE = True
        print("\nALL RUNS FINISHED\n", flush=True)
finally:
    print(write_summary(), flush=True)
    save_progress()
    to_s3()
