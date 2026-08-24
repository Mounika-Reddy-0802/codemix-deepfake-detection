# Data sources, mirrors and integrity

**W3-T2, owner L.** The corpora are tens of gigabytes and are gitignored — they
never enter this repository. What the repository carries instead is this page plus
`docs/data_checksums.json`: where each archive came from, what it hashes to, and
the team mirror a member can pull it from.

That is not bookkeeping for its own sake. "We used MUCS 2021" is an unverifiable
claim unless the exact archive is pinned; and a teammate downloading 7 GB from a
Drive link has no other way to tell a complete archive from one that stopped at
94% and still extracts.

---

## 1. Where each corpus lives

| Corpus | Official source | Licence | Needed by |
|---|---|---|---|
| **ASVspoof 2019 LA** | [Edinburgh DataShare 10283/3336](https://datashare.ed.ac.uk/handle/10283/3336) | Research (ASVspoof) | Week 5 — S1 training |
| **MUCS 2021 Hindi-English** | [OpenSLR 104](https://www.openslr.org/104/) | CC BY-SA 4.0 | **Week 4 — cloning references** |
| **HiACC** (adult subset) | [Zenodo 15551669](https://zenodo.org/records/15551669) | CC BY 4.0 | **Week 4 — code-mixed eval** |
| IndicVoices / IndicSynth / IndicTTS-Deepfake | Hugging Face, streaming only | see `docs/licences.md` | Week 5+ eval |
| AffectDF | arXiv:2608.05507 release | CC BY-NC 4.0 | Week 8 cross-eval |

**Deliberately not downloaded:** ASVspoof 2021 DF (34.5 GB, storage), MUCS
Bengali-English (out of scope). The three HF corpora are *streamed*, never bulk
downloaded — see `src/data/download.py`.

## 2. Team mirrors (Drive)

Downloading MUCS from OpenSLR takes ~1.5 hours per person. Mirroring the
**original, unextracted archives** to Drive once and sharing the link is faster
and — because the hash is pinned below — provably the same data.

> **Mirror the archives, never the extracted folders.** An extracted HiACC copy
> has *not* been through the child-audio quarantine, which only happens during
> `scripts/01_download_data.sh`. Sharing an extracted tree is how child audio
> ends up somewhere it should never be.

Paste links here as they are created:

| Archive | Drive link | Uploaded by |
|---|---|---|
| `Hindi-English_train.tar.gz` | _paste_ | |
| `Hindi-English_test.tar.gz` | _paste_ | |
| HiACC archive | _paste_ | |
| `LA.zip` (ASVspoof 2019) | _paste_ | |

Record them in the machine-readable manifest at the same time:

```bash
python -m src.data.checksums --write \
  --mirror "raw/mucs2021/Hindi-English_train.tar.gz=https://drive.google.com/..."
```

## 3. Recording hashes (whoever has the archives)

```bash
DATA_ROOT=/c/dfdata python -m src.data.checksums --write
```

Hashes every known archive present, merges into `docs/data_checksums.json`, and
prints a markdown table to paste into section 4. Merging matters: no one machine
holds every corpus, so recording MUCS must not erase somebody else's ASVspoof
entry.

## 4. Verifying a copy (everyone else, before using it)

```bash
DATA_ROOT=/c/dfdata python -m src.data.checksums --verify
```

- **size mismatch** → the download was interrupted. Re-fetch; `curl -C -` and
  `wget -c` both resume.
- **hash mismatch, size identical** → the archive was repacked or corrupted. Do
  not use that copy.
- **archive absent** → not a failure. Nobody is expected to hold all of them.

Recorded hashes are in `docs/data_checksums.json`.

## 5. Where the data lives on disk

The repository sits inside a OneDrive-synced folder on at least one machine, and
cloud clients do not honour `.gitignore` — they will happily upload 60 GB. So the
corpora live **outside** the repo, selected with `DATA_ROOT`:

```bash
DATA_ROOT=/c/dfdata bash scripts/01_download_data.sh --run --only mucs
DATA_ROOT=/c/dfdata bash scripts/01_download_data.sh --run --only hiacc
```

`scripts/01_download_data.sh` warns automatically if `DATA_ROOT` points inside a
OneDrive / Dropbox / Google Drive path, and refuses to start a download that
cannot fit on the drive.

## 6. Storage budget

| Corpus | Archive | Extracted | Total |
|---|---|---|---|
| ASVspoof 2019 LA | **7.12 GB** | **7.13 GB** | **14.3 GB** |
| MUCS Hindi-English | ~7.7 GB | ~7.7 GB | ~15 GB |
| HiACC | ~0.5 GB | ~0.5 GB | ~1 GB |
| **All three** | | | **~30 GB** |

> Measured on 19 Aug 2026, not estimated. ASVspoof LA was long recorded here as
> ~23 GB, which is roughly the FULL 2019 release (LA + PA); the **LA partition
> alone** is 7.12 GB. Extraction is 1.00x, not 2x, because the archive holds FLAC
> that is already compressed — so an archive and its extraction coexist happily.
> Planning storage on the old figure over-reserved by ~30 GB.

Archives can be deleted after extraction *once their hash is recorded here* —
the record is what preserves provenance, not the file.
