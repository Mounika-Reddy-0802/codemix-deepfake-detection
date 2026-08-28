# Progress log

One line per finished task: `date | task | owner | branch | summary`.

| Date | Task | Owner | Branch | Summary |
|------|------|-------|--------|---------|
| 2026-07-31 | W1-T3 | M | main | repo bootstrap: structure, gitignore, requirements, ci, hooks, docs |
| 2026-07-31 | W1-T1 | L | week1/lahari | download+verify script with hiacc child quarantine; hingcos email template |
| 2026-07-31 | W1-T2 | L | week1/lahari | hf streaming loaders for eval-only indic corpora + licence register |
| 2026-07-31 | W1-T7 | ALL | week1/lahari | related-work + ethics templates (split across the three week branches) |
| 2026-07-31 | W1-T4 | M | week1/mounika | env check notebook: loads wav2vec2-base/xlsr-53/wavlm-base + forward pass + wandb init |
| 2026-07-31 | W1-T5 | SK | week1/saikrishna | free aiortc webrtc harness with 16khz pcm tap + shared audio-source contract |
| 2026-07-31 | W1-T6 | SK | week1/saikrishna | twilio conference + media streams design note with 4 alert layers and week-6 timing |
| 2026-08-03 | W2-T1 | L | week2/lahari | preprocessing (resample/vad/loudness/segment) + g711+snr channel simulation chain |
| 2026-08-03 | W2-T2 | M | week2/mounika | dataset+collate, speaker-disjoint splits + anti-leakage checks, metrics (eer/auc/f1/bootstrap) |
| 2026-08-03 | W2-T3 | SK | week2/saikrishna | adult-speaker selection + xtts-v2 clone driver with metadata logging and held-out-tool firewall |
| 2026-08-10 | W3-M | M | week3/mounika | stage-1 training scaffolding: ssl encoder + attentive pooling + mlp head + amp training loop |
| 2026-08-10 | W3-L | L | week3/lahari | xtts-v2 generation at scale: clone-job assembly + per-file metadata + generation stats |
| 2026-08-10 | W3-SK | SK | week3/saikrishna | held-out tortoise tts driver for unseen-attack split with tool+pool firewall |
| 2026-08-12 | W3-T5 | SK | week3-krishna-xtts-rvc-pilot | ethics gate blocks all generation, no override; pilot protocol written, 0 clips generated (gate closed) |
| 2026-08-12 | W3-T4 | SK | week3-krishna-xtts-rvc-pilot | three-way speaker pool carve + sha256 freeze/verify; pools not yet frozen (needs shortlist) |
| 2026-08-12 | W3-T3 | M | week3-mounika-affectdf-taxonomy | affectdf related-work anchor read from the pdf + precise gap statement; attack taxonomy in table-1 format |
| 2026-08-12 | W3-T6 | M | week3-mounika-affectdf-taxonomy | env verification (ok=3 missing=7 manual=8), secrets never echoed; readme resource links |
| 2026-08-12 | W3-T2 | L | week3-lahari-preprocess-mucs-hiacc | fixed quarantine bypass in the preprocessing walk; quarantine audit + report; channel-sim listening test builder |
| 2026-08-12 | W3-T4 | L | week3-lahari-preprocess-mucs-hiacc | snr/duration speaker ranking + shortlist that is never padded; awaiting downloads |
| 2026-08-12 | W3-T2 | L | dev | mucs + hiacc downloaded to C:/dfdata; quarantine bug found on real data: 1858 child files were live, now excluded; 3318 adult reachable |
| 2026-08-12 | W3-T2 | L | dev | corpus-aware indexing: mucs kaldi tables (52825 utts / 520 spk) + hiacc prefix ids (3318 / 24 spk); 20-clip listening set rendered |
| 2026-08-12 | W3-T4 | L | dev | speaker ranking over 520 mucs speakers, top 50 shortlisted; pools NOT frozen pending the team listening pass |
| 2026-08-13 | W3-T4 | SK | dev | speaker pools FROZEN: 25 train / 10 adaptation / 15 eval, sha256 f57e0d85dbd3c8f5b96ad6af59edae7218104b0167807eae6433314947063e7b |
| 2026-08-13 | W3-T2 | L | dev | channel sim verified objectively: 20/20 band-limited (hf energy 0.6% clean -> 0.0001% channel), snr 17.6 db, corr 0.99 |
| 2026-08-13 | W3-T2 | ALL | dev | listening test passed: all three rated 20 pairs, telephony 4.0/5, intelligible 4.0/5; channel protocol verified for g711 at 20 db |
| 2026-08-13 | W3-T5 | SK | dev | xtts pilot pack ready: 20 jobs, 4 script/tag cells x 5 train-pool speakers, refs cut; generation needs a kaggle gpu (no coqui tts on py3.13, no cuda here) |
| 2026-08-13 | W3-T5 | SK | dev | colab notebook for the xtts pilot; platform moved kaggle -> colab, week-5 compute plan flagged for revision |
| 2026-08-17 | W3-T2 | L | week3-lahari-preprocess-mucs-hiacc | quarantine re-verified on the gpu laptop: same 3318/1858/24 as dev, 0 outside quarantine, no CH* id in any manifest; audit now refuses to overwrite the hand-written report; AD63 undocumented, not usable as a reference |
| 2026-08-17 | W3-T6 | M | week3-mounika-affectdf-taxonomy | gpu laptop setup hardened: xtts weights pre-fetched from hf with the hash.md5 re-download trap documented, transfer bundle gitignored so the signed ethics note cannot be committed |
| 2026-08-17 | W3-T5 | SK | week3-krishna-xtts-rvc-pilot | devanagari -> latin transliteration (P-014): mucs is 98.2% devanagari-bearing so hinglish had to be produced, not filtered; 0 unmapped chars across 56143 transcripts |
| 2026-08-17 | W3-T5 | SK | week3-krishna-xtts-rvc-pilot | pilot packs reproducible across machines (P-016): whole-second durations made the old tie-break filesystem-dependent, so the two laptops swapped speakers 4 and 5 in every cell |
| 2026-08-17 | W3-T5 | SK | week3-krishna-xtts-rvc-pilot | xtts char limit now applied to the romanised text (P-015): romanisation inflates length ~13%, which put 14 of 20 pilot transcripts over the 150-char hindi cap |
| 2026-08-17 | W3-T5 | SK | week3-krishna-xtts-rvc-pilot | script a/b generated on gpu (rtx 3050): 20 romanised + 20 devanagari, matched speaker/sentence/tag, 0 failures; blind sheet + answer key for 3 raters |
| 2026-08-17 | W3-T5 | SK | week3-krishna-xtts-rvc-pilot | a/b pre-screen: median 13.9 vs 14.4 chars/sec (devanagari vs romanised), so romanisation does not slow or truncate speech; 1 clip of 40 silently produced 0.83 s from 150 chars -- week-4 needs a duration auto-filter |
| 2026-08-19 | W4-T1 | L | week3-krishna-xtts-rvc-pilot | xtts generation at scale complete: 4,000 clips over 25 train-pool speakers (160 each), 1,404 transcripts cross-paired, 6.91 h of audio, 0 generator exceptions |
| 2026-08-19 | W4-T6 | L | week3-krishna-xtts-rvc-pilot | quality screen caught 116 clips (2.9%) that raised nothing: stalled 3-6 c/s, truncated 31-92 c/s, or under 1 s |
| 2026-08-19 | W4-T6 | L | week3-krishna-xtts-rvc-pilot | 95 of those 116 came from speaker 987461 alone, whose reference was 25 s at 89% silence (2.65 s of voice) |
| 2026-08-19 | W4-T6 | L | week3-krishna-xtts-rvc-pilot | references now ranked by effective speech seconds, not duration; 987461 re-cut 2.65 s -> 12.34 s, 5 of 25 refs were under threshold |
| 2026-08-19 | W4-T1 | L | week3-krishna-xtts-rvc-pilot | after re-cut + regeneration: 3,998 of 4,000 usable (99.95%, was 97.1%); the 2 rejects are borderline slow, both speakers keep 159 |
| 2026-08-19 | W3-T6 | L | week3-krishna-xtts-rvc-pilot | asvspoof 2019 la downloaded and extracted: 7.12 GB zip (exact byte match), 122,299 flac, sha256 208a7e4e recorded then zip deleted |
| 2026-08-19 | W3-T6 | L | week3-krishna-xtts-rvc-pilot | cm protocols verified against published figures: train 25380/20spk, dev 24844/20spk, eval 71237/67spk, zero speaker overlap between splits |
| 2026-08-19 | W3-T6 | L | week3-krishna-xtts-rvc-pilot | corrected the asvspoof size in 3 docs: it is 7.12 GB, not the ~23-24 GB recorded (that is the full LA+PA release); extraction is 1.00x since flac is already compressed |
| 2026-08-17 | W3-T6 | M | week3-krishna-xtts-rvc-pilot | gpu laptop environment: dependency pins made installable (librosa/hub/websockets), xtts weights pre-fetched from hf with the hash.md5 trap documented, transfer bundle gitignored so the signed note cannot be committed |
| 2026-08-23 | W3-T6 | M | week3-krishna-xtts-rvc-pilot | commit-msg hook hardened and actually installed via core.hooksPath: strips trailers, rejects attribution words, verified on 3 live messages |
| 2026-08-24 | W7-T4-prep | SK | week3-krishna-xtts-rvc-pilot | lora adapters wired into the training loop, zero-init verified against the real wav2vec2 detector (1.13% trainable, bit-identical to stage-1 before any step) |
| 2026-08-24 | W7-T4-prep | SK | week3-krishna-xtts-rvc-pilot | pool-firewalled generation + manifest pipeline for the adaptation/eval pools (train pool was the only one built so far); college-pc transfer guide for the rtx 4500 |
| 2026-08-28 | W4-T1 | SK | week3-krishna-xtts-rvc-pilot | stage-1 baseline trained + evaluated on a kaggle gpu: pooled eval EER 0.5843%, AUC 0.9998, F1 0.9723 over 71,237 clips; readme and docs/STAGE1_ASVSPOOF_RESULTS.md updated to match |
