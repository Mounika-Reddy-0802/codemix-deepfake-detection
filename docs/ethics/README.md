# Ethics & licence sign-off

The mentor must sign off the ethics/licence note **before any spoof generation**
(Week 2+). Voice cloning is the ethically sensitive step, so the gate comes first.
This page states **what the sign-off must cover**; the signed artefact is scanned
into this folder.

## What the sign-off must cover

1. **Licence compliance** — every corpus and tool used within its licence
   (`docs/licences.md`): IndicSynth **CC-BY-NC** (non-commercial), MUCS **BY-SA**
   (share-alike on derivatives), HiACC **BY**, XTTS-v2 **CPML** (non-commercial).
   Confirm the project is **non-commercial academic research**, which all permit.

2. **HiACC child-data exclusion (hard rule)** — the 1,858 HiACC **child**
   utterances are excluded from the pipeline **entirely**: never bonafide, never a
   cloning reference, never in any manifest. Quarantined on extract into
   `data/raw/hiacc/_EXCLUDED_children/`. Mentor confirms this is understood and
   enforced (also checked by `tests/test_splits.py`).

3. **Voice-cloning ethics** — clones are generated **only** from adult speakers in
   licensed research corpora (MUCS / HiACC-adult), **only** to build a detector,
   **never** to impersonate a real, identifiable private individual and **never**
   released as a standalone clone set. No cloning of team members' or third
   parties' voices without consent.

4. **Consent & privacy** — no personal data beyond what the source corpora already
   contain under their own consent/ethics approvals; no attempt to re-identify
   speakers; secrets (`.env`) never committed.

5. **Responsible framing / dual-use** — this is **defensive** detection research.
   The live demo **warns the potential victim, never the caller**, never
   auto-disconnects, and produces no artefact that helps commit fraud. The paper
   states the DLT/production caveats for real deployment in India.

6. **Anti-leakage integrity** — Stage-1 trains on ASVspoof 2019 LA (English) only;
   no eval-only corpus and no Tortoise (held-out attack) audio ever enters a
   training manifest. This underpins the paper's central claim, so it is an ethics
   *and* scientific-integrity item.

7. **Scope of released artefacts** — decide and record what (if anything) is
   released: code yes; generated clones only under source-licence terms (MUCS
   BY-SA), and never the HiACC child data.

## Sign-off artefact

- The mentor reviews this note + `docs/licences.md`, then signs.
- Save the scan here as `docs/ethics/mentor_signoff_<yyyy-mm-dd>.pdf`.
- Record in `docs/progress.md`: date, "ethics sign-off obtained", mentor name.
- **No spoof generation (Week 2 XTTS pilots included) until this exists.**

## Pre-meeting checklist

- [ ] `docs/licences.md` complete and current.
- [ ] Child-exclusion mechanism in place (download script quarantine + test).
- [ ] Cloning scope agreed (adult, licensed corpora, detection-only).
- [ ] Demo alert policy agreed (warn victim only, never caller, no auto-disconnect).
- [ ] Release policy for code/data/clones decided.
