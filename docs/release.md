# Release

What this project publishes, and what it does not.

**Published: the code and the metadata.** The repository holds every line of code,
configuration and test. `python -m src.reporting.release` builds
`codemix_metadata_release.zip`, which adds nothing new but gathers, in one archive
with a `CHECKSUMS.sha256` file, everything needed to audit a number in the paper:

- the manifests (every clip list with its label, speaker, attack, language condition
  and channel; paths are `${DATA_ROOT}` placeholders),
- the generation job tables and the frozen speaker pools,
- the training configurations,
- the per-clip scores and JSON summaries each results table is read from,
- the result write-ups, the datasheet and the licence notes.

The archive takes only files git tracks, from an allowlist of folders and file types,
and refuses any audio, model weight, archive or secret.

**Not published: the audio and the weights.** The clones are cloned speech
of identifiable people (`docs/datasheet.md`), so they stay private; their metadata and
checksums are in the archive. The genuine audio is MUCS 2021 (CC BY-SA 4.0) and HiACC
(CC BY 4.0), downloadable from their publishers, and the project's bundling code
rebuilds the normalised and phone-line versions from them. HiACC child audio is never
used and never appears in any manifest.

**Proposed: the detector weights.** `docs/MODEL_CARD.md` drafts a card for the final
recipe's adapter (intended use, out-of-scope uses, evaluation, limits, licence). Publishing
the weights waits for the team's and the mentor's decision; until then they stay private.
