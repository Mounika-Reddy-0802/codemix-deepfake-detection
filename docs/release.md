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

**Not published: the audio, and the deployed and final-recipe weights.** The English-only
detector (S1, `checkpoints/baseline/best.pt`) and two early adapters
(`checkpoints/lora_codemix*/`) are tracked in the repository through Git LFS, so anyone who
can clone the repository can fetch them. The clones are cloned speech
of identifiable people (`docs/datasheet.md`), so they stay private; their metadata and
checksums are in the archive. The genuine audio is MUCS 2021 (CC BY-SA 4.0) and HiACC
(CC BY 4.0), downloadable from their publishers, and the project's bundling code
rebuilds the normalised and phone-line versions from them. HiACC child audio is never
used and never appears in any manifest.

**Proposed: the detector weights.** `docs/MODEL_CARD.md` drafts a card for the final
recipe's adapter (intended use, out-of-scope uses, evaluation, limits, licence). Publishing
the final-recipe weights waits for the team's and the mentor's decision; until then they stay
private.

The package is ready for that decision. `python -m src.inference.model_package export`
turns a checkpoint into a folder holding `model.safetensors` (the whole state dict: the
adapters sit on Stage-1's fine-tuned encoder, so they cannot be published alone),
`config.json` (training config, dev metrics, the live operating point and the windowing),
`README.md` (the model card with Hugging Face metadata and the draft notice removed) and
`SHA256SUMS`, which `load_package` checks before loading. safetensors holds tensors only, so
loading the package runs no pickled code. `verify` loads the checkpoint and the package
side by side; for `final_s1234` the scores are identical (largest difference 0.0).

If the team and the mentor agree, publishing is two steps:

1. Fill in the account and the licence they chose (`LICENSE` in
   `src/inference/model_package.py`) and export the package again, then run `verify`. The
   package's `README.md` is the model card as it stood at export time, so an export made
   before the card's last change (the real-call results) is out of date.
2. Upload the folder to a new model repository on Hugging Face, for example with
   `huggingface-cli upload <account>/codemix-deepfake-detector <package folder> .`, and
   replace "The trained weights are not yet public" in both papers' availability
   statements with the link.
