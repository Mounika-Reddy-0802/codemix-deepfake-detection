# Bibliography verification report (paper/refs.bib)

Checked 2026-10-11. 38 entries: the 10 existing ones (keys unchanged) and 28 new ones.
Every DOI in the file was resolved against Crossref (the metadata behind doi.org) in a
final pass. All 29 DOIs resolve, and the title and page range on each record match the
.bib entry.

**Access notes.** DBLP (dblp.org and its mirrors), OpenReview and IEEE Xplore blocked
automated access during this session (bot challenge or empty page), so none of them
could be used. I used these instead:

- **ISCA papers:** the ISCA Archive page (`citation_doi`, `citation_firstpage`,
  `citation_lastpage` meta tags) plus the Crossref record.
- **IEEE and ACM papers:** the Crossref record that IEEE or ACM deposited, plus the arXiv
  record (comment, journal_ref or DOI field) as a second source.
- **ICLR papers:** the iclr.cc poster page, proceedings.iclr.cc, or ML Anthology
  (mlanthology.org).
- **Other sources:** ACL Anthology `.bib` exports, the NeurIPS proceedings `.bib` export,
  the RFC Editor / IETF Datatracker, the ITU-T Recommendation page, CEUR-WS, Open Library
  (ISBN) and the GitHub API.

**BibTeX style caveat.** `paper/main.tex` uses `IEEEtran.bst`. To my knowledge that
style does not print the `doi`, `eprint`, `archivePrefix`, `primaryClass` or `urldate`
fields. The `note` field (arXiv id, access date) and the `url` field are what reach the
rendered reference list. If the reviewer wants DOIs printed, check the compiled output.

---

## 1. Entry table

### Existing entries

| Key | Status | What was added / changed | Source URL(s) |
|---|---|---|---|
| todisco2019asvspoof | verified | DOI 10.21437/Interspeech.2019-2249; publisher ISCA. Author forms aligned to ISCA ("Sahidullah, Md.", "Kinnunen, Tomi H."). Pages 1008--1012 confirmed. | https://www.isca-archive.org/interspeech_2019/todisco19_interspeech.html ; https://api.crossref.org/works/10.21437/Interspeech.2019-2249 |
| baevski2020wav2vec | verified | volume 33; pages 12449--12460; publisher Curran Associates; proceedings URL. Booktitle normalised. **Author corrected: "Zhou, Henry" → "Zhou, Yuhao"** (form used in the NeurIPS proceedings). No DOI exists for NeurIPS 2020. | https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Bibtex.bib ; https://proceedings.neurips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html ; https://arxiv.org/abs/2006.11477 |
| hu2022lora | verified | Booktitle normalised to {Proc. ICLR}; OpenReview URL added. ICLR 2022 has no pages and no DOI. | https://iclr.cc/virtual/2022/poster/6319 ; https://mlanthology.org/iclr/2022/hu2022iclr-lora/ ; https://arxiv.org/abs/2106.09685 |
| okabe2018attentive | verified | DOI 10.21437/Interspeech.2018-993; publisher ISCA. Pages confirmed. | https://www.isca-archive.org/interspeech_2018/okabe18_interspeech.html |
| tak2022wav2vec | verified | pages 112--119; DOI 10.21437/Odyssey.2022-16; publisher ISCA. Booktitle normalised to {Proc. Odyssey}. | https://www.isca-archive.org/odyssey_2022/tak22_odyssey.html ; https://arxiv.org/abs/2202.12233 |
| diwan2021mucs | verified | DOI 10.21437/Interspeech.2021-1339; publisher ISCA. Title, 20-author list and pages confirmed. | https://www.isca-archive.org/interspeech_2021/diwan21_interspeech.html |
| casanova2024xtts | verified | DOI 10.21437/Interspeech.2024-2016; publisher ISCA. Pages confirmed. | https://www.isca-archive.org/interspeech_2024/casanova24_interspeech.html ; https://arxiv.org/abs/2406.04904 |
| betker2023tortoise | verified (arXiv only) | Converted to arXiv @misc with eprint 2305.07243, archivePrefix, primaryClass cs.SD and URL. No published version found. | https://arxiv.org/abs/2305.07243 (export.arxiv.org API) |
| rvc2023 | verified (software) | url field, urldate 2026-10 and note "Accessed: 2026-10"; howpublished = GitHub repository. Repo created 2023-03-27 (supports year 2023). No commit hash added. | https://api.github.com/repos/RVC-Project/Retrieval-based-Voice-Conversion-WebUI |
| affectdf2026 | verified (arXiv only) | arXiv @misc fields (eprint 2608.05507, eess.AS) and URL. **Author form corrected: "Suresh Chandra, Shreeram" → "Chandra, Shreeram Suresh".** Still v1 only; no published version found. | https://arxiv.org/abs/2608.05507 ; https://www.isca-archive.org/odyssey_2024/chandra24_odyssey.html (name form) |

### New entries

| Key | Status | What was added | Source URL(s) |
|---|---|---|---|
| liu2023asvspoof2021 | verified | IEEE/ACM TASLP 31:2507--2522 (2023), DOI 10.1109/TASLP.2023.3285283 | https://api.crossref.org/works/10.1109/TASLP.2023.3285283 ; https://arxiv.org/abs/2210.02437 (DOI field) ; https://ieeexplore.ieee.org/document/10155166/ |
| tak2021rawnet2 | verified | ICASSP 2021, 6369--6373, DOI 10.1109/ICASSP39728.2021.9414234 | Crossref record ; https://arxiv.org/abs/2011.01108 ("Accepted to ICASSP 2021") |
| jung2022aasist | verified | ICASSP 2022, 6367--6371, DOI 10.1109/ICASSP43922.2022.9747766 | Crossref record ; https://arxiv.org/abs/2110.01200 |
| tak2022rawboost | verified | ICASSP 2022, 6382--6386, DOI 10.1109/ICASSP43922.2022.9746213 | Crossref record ; https://arxiv.org/abs/2111.04433 ("Accepted to IEEE ICASSP 2022") |
| muller2022generalize | verified | Interspeech 2022, 2783--2787, DOI 10.21437/Interspeech.2022-108 | https://www.isca-archive.org/interspeech_2022/muller22_interspeech.html |
| yi2022add | verified | ICASSP 2022, 9216--9220, DOI 10.1109/ICASSP43922.2022.9746939; 17-author list from the published record | Crossref record ; https://arxiv.org/abs/2202.08433 |
| muller2024mlaad | verified | **IJCNN 2024** (venue confirmed), 1--7, DOI 10.1109/IJCNN60899.2024.10650962 | Crossref record ; https://arxiv.org/abs/2401.09512 ("IJCNN 2024") |
| conneau2021xlsr | verified | Interspeech 2021, 2426--2430, DOI 10.21437/Interspeech.2021-329 | https://www.isca-archive.org/interspeech_2021/conneau21_interspeech.html |
| babu2022xlsr | verified | Interspeech 2022, 2278--2282, DOI 10.21437/Interspeech.2022-143 | https://www.isca-archive.org/interspeech_2022/babu22_interspeech.html |
| chen2022wavlm | verified | IEEE JSTSP 16(6):1505--1518, Oct 2022, DOI 10.1109/JSTSP.2022.3188113 | Crossref record ; https://arxiv.org/abs/2110.13900 (DOI field) |
| pasad2021layerwise | verified | ASRU 2021, 914--921, DOI 10.1109/ASRU51503.2021.9688093 | Crossref record ; https://arxiv.org/abs/2107.04734 ("Accepted to ASRU 2021") |
| zhang2024sls | verified | ACM MM 2024 (32nd), 6765--6773, ACM New York, DOI 10.1145/3664647.3681345 | https://api.crossref.org/works/10.1145/3664647.3681345 |
| loshchilov2019adamw | verified | ICLR 2019, OpenReview URL (no pages or DOI exist) | https://mlanthology.org/iclr/2019/loshchilov2019iclr-decoupled/ ; https://arxiv.org/abs/1711.05101 ("Published as a conference paper at ICLR 2019") |
| itu1988g711 | verified | ITU-T Rec. G.711 (11/88), "Pulse code modulation (PCM) of voice frequencies", approved 1988-11-25 | https://www.itu.int/rec/T-REC-G.711-198811-I/en |
| valin2012opus | verified | RFC 6716, IETF, Sept 2012, DOI 10.17487/RFC6716; full author names from the RFC text | https://datatracker.ietf.org/doc/rfc6716/bibtex/ ; https://www.rfc-editor.org/rfc/rfc6716.txt ; https://www.rfc-editor.org/info/rfc6716 |
| efron1993bootstrap | verified | Chapman & Hall, New York, 1993; series Monographs on Statistics and Applied Probability vol. 57; ISBN 0-412-04231-2; DOI 10.1007/978-1-4899-4541-9 (its print ISBN is the same Chapman & Hall ISBN) | https://openlibrary.org/isbn/9780412042317.json ; https://api.crossref.org/works/10.1007/978-1-4899-4541-9 |
| das2021companding | verified | ICASSP 2021, 6349--6353, DOI 10.1109/ICASSP39728.2021.9413501 | Crossref record |
| chen2021channel | verified | ASVspoof 2021 Workshop, 75--82, DOI 10.21437/ASVSPOOF.2021-12 | https://www.isca-archive.org/asvspoof_2021/chen21_asvspoof.html ; https://arxiv.org/abs/2107.12018 |
| tsutsumi2026channel | verified | Odyssey 2026, 9--14, DOI 10.21437/Odyssey.2026-2 | https://www.isca-archive.org/odyssey_2026/tsutsumi26_odyssey.html |
| delgado2026presentation | verified | ICASSP 2026, 16242--16246, DOI 10.1109/ICASSP55912.2026.11460707 | Crossref record ; https://arxiv.org/abs/2509.26471 ("ICASSP 2026") |
| wang2023lora | verified | IJCAI 2023 DADA Workshop, CEUR-WS Vol-3597, 101--106; **7 authors in the published version (adds Xinrui Yan)** | https://ceur-ws.org/Vol-3597/ ; https://ceur-ws.org/Vol-3597/paper17.pdf ; https://arxiv.org/abs/2306.05617 |
| wu2024adapter | verified | Interspeech 2024, 2095--2099, DOI 10.21437/Interspeech.2024-253 | https://www.isca-archive.org/interspeech_2024/wu24c_interspeech.html |
| oiso2024prompt | verified | Interspeech 2024, 2710--2714, DOI 10.21437/Interspeech.2024-81 | https://www.isca-archive.org/interspeech_2024/oiso24_interspeech.html |
| laakkonen2026metalora | verified | ICASSP 2026, 19092--19096, DOI 10.1109/ICASSP55912.2026.11463170 | Crossref record ; https://arxiv.org/abs/2502.10838 |
| sharma2025indicsynth | verified | ACL 2025 (long), 22037--22060, Vienna, DOI 10.18653/v1/2025.acl-long.1070 | https://aclanthology.org/2025.acl-long.1070.bib |
| huang2025speechfake | verified | ACL 2025 (long), 9985--9998, Vienna, DOI 10.18653/v1/2025.acl-long.493 | https://aclanthology.org/2025.acl-long.493.bib |
| kuckreja2026habibi | verified | ICLR 2026, 28903--28926, proceedings URL | https://proceedings.iclr.cc/paper_files/paper/2026/hash/3122aaa22b2fe83f9cead1a696f65ceb-Abstract-Conference.html ; https://arxiv.org/abs/2505.22581 |
| girish2026indiccodecfake | verified | Findings of ACL 2026, 43476--43489, San Diego, DOI 10.18653/v1/2026.findings-acl.2159 | https://aclanthology.org/2026.findings-acl.2159.bib |

---

## 2. Unverified fields

**No field in the .bib is unverified.** Fields I deliberately left out because no source
exists or I could not reach one:

- **hu2022lora, loshchilov2019adamw:** pages and DOI. ICLR 2019 and 2022 have neither;
  proceedings.iclr.cc only covers 2024 onwards.
- **baevski2020wav2vec:** DOI. NeurIPS 2020 does not issue DOIs. I also omitted the
  proceedings editors, which the NeurIPS .bib lists, to keep the entry short.
- **betker2023tortoise, affectdf2026:** venue, because both are arXiv-only.
- **itu1988g711:** DOI. The ITU page shows none.
- **Conference addresses and months:** omitted for ISCA, IEEE and ICLR entries. Only the
  ACL entries (from ACL Anthology) and zhang2024sls (from Crossref) carry an address.
- **rvc2023:** commit hash or release version, per instructions.
- **Unchecked on publisher pages:** IEEE Xplore pages could not be fetched, so the IEEE
  and ACM entries rest on the Crossref record those publishers deposited, plus arXiv.
  DBLP and OpenReview could not be fetched either, so neither was used.

---

## 3. What each new reference is, and where it goes

**Related work: anti-spoofing benchmarks and detectors**

- **liu2023asvspoof2021:** The ASVspoof 2021 journal overview of the LA, PA and DF tracks. Its LA track transmits audio through telephony codecs, which makes it the closest benchmark precedent for our channel condition. Supports the benchmark paragraph and the channel-robustness discussion.
- **tak2021rawnet2:** RawNet2, an end-to-end raw-waveform countermeasure and an ASVspoof 2021 baseline. Supports related work on detectors and gives context for the AffectDF numbers we quote.
- **jung2022aasist:** AASIST, a spectro-temporal graph-attention countermeasure and the standard non-SSL baseline. Supports related work on detectors; AffectDF reports its collapse.
- **tak2022rawboost:** RawBoost, the waveform augmentation used with the wav2vec 2.0 front-end in tak2022wav2vec. Supports related work on SSL detectors and augmentation versus channel matching.
- **muller2022generalize:** The In-the-Wild dataset, showing that detectors trained on ASVspoof fail on real-world audio. Supports related work on generalisation, our closest precedent for the generalisation-gap question.
- **yi2022add:** The ADD 2022 challenge, which covers low-quality and partially fake Mandarin audio. Supports the benchmark paragraph and shows a non-English challenge setting.
- **zhang2024sls:** XLS-R with a sensitive-layer-selection classifier, a strong SSL-based detector that uses multiple layers. Supports related work on SSL detectors and contrasts with our single-encoder plus attentive-pooling design.

**Related work: self-supervised encoders** (also the encoder ablation)

- **conneau2021xlsr:** XLSR-53, cross-lingual wav2vec 2.0 pre-training. Supports SSL related work and the XLSR-53 encoder ablation (`configs/ablation_encoder_xlsr53_lora.yaml`).
- **babu2022xlsr:** XLS-R, a 128-language wav2vec 2.0 model and the usual front-end in SSL anti-spoofing (tak2022wav2vec, zhang2024sls). Supports SSL related work.
- **chen2022wavlm:** WavLM, the denoising-plus-masked-prediction SSL model listed as the secondary encoder in the related-work notes. Supports SSL related work and any WavLM ablation.
- **pasad2021layerwise:** A layer-wise probing study of wav2vec 2.0. Supports the results claim that linguistic content emerges in the middle and upper layers (results.tex, layer analysis).

**Related work: codec, channel and telephony robustness**

- **das2021companding:** μ-law/A-law companding as augmentation for logical-access detection. This is the G.711 operation we apply, used there as augmentation rather than as a matched condition. Supports codec/channel related work and the contrast with our channel-matched protocol.
- **chen2021channel:** A channel-robust ASVspoof 2021 system trained with landline, cellular and VoIP codec augmentation. Supports codec/channel related work.
- **tsutsumi2026channel:** Evaluates countermeasures on real phone-call, VoIP and face-to-face recordings, finds near-chance EER under channel mismatch, and fixes it by domain adaptation with in-domain degraded speech. **This is the closest prior work to our channel-matched protocol** and must be discussed when answering the reviewer.
- **delgado2026presentation:** Argues that detectors fail in deployment because training audio is not passed through the phone presentation channel, and builds call-injected and loudspeaker-replayed training data. Supports codec/channel related work and the live-call (system) section.

**Related work: parameter-efficient adaptation**

- **wang2023lora:** LoRA applied to wav2vec2 for fake-audio detection, matching full fine-tuning with far fewer parameters. **This is direct prior use of LoRA on a wav2vec2 detector**, so our novelty is the code-mixed, channel-matched setting and the native-training ceiling, not LoRA itself.
- **wu2024adapter:** Adapter learning inside a pre-trained SSL front-end for robust spoof detection. Supports the parameter-efficient adaptation paragraph.
- **oiso2024prompt:** Prompt tuning for test-time domain adaptation of a detector with a small target set. Supports the parameter-efficient adaptation paragraph and is the closest in spirit to our Stage-3 domain adaptation.
- **laakkonen2026metalora:** Meta-learned LoRA for generalisable speech deepfake detection. Supports the parameter-efficient adaptation paragraph as the most recent LoRA-for-detection work.

**Related work: multilingual, Indic and code-switched detection**

- **muller2024mlaad:** MLAAD, a TTS spoof dataset in many languages. It is monolingual per utterance, which is the multilingual-shift precedent the gap statement contrasts with.
- **sharma2025indicsynth:** IndicSynth, synthetic speech in 12 Indian languages. It is one of our evaluation-only corpora (project dataset rules, §6) and the main Indic deepfake dataset.
- **huang2025speechfake:** SpeechFake, a large multilingual deepfake dataset. Supports the multilingual-shift paragraph.
- **kuckreja2026habibi:** ArEnAV, an Arabic–English **audio-visual** deepfake dataset with **intra-utterance code-switching**. Supports the multilingual/code-switched paragraph and is the nearest code-switching precedent (see the caution below).
- **girish2026indiccodecfake:** Indic-CodecFake, neural-audio-codec deepfakes in Indic languages, showing that English-trained detectors do not transfer. Supports the Indic paragraph.

**Methods**

- **loshchilov2019adamw:** The AdamW optimiser (methodology.tex: "All systems use AdamW").
- **itu1988g711:** G.711 μ-law PCM, the codec in the channel-matched render (methodology.tex §channel; system.tex).
- **valin2012opus:** The Opus codec, used in the browser carrier path (system.tex).
- **efron1993bootstrap:** The bootstrap confidence intervals over clips (methodology.tex; results.tex table caption).

**Caution for the reviewer response (scope of the novelty claim).** related_work.tex
currently says "We also found no prior work measuring detector generalisation to
code-mixed speech over a telephone channel." That narrow claim still holds. However:

- **tsutsumi2026channel and delgado2026presentation** already study detection under real
  telephony channel mismatch, and channel-domain adaptation.
- **kuckreja2026habibi** already builds intra-utterance code-switched deepfakes (Arabic–English,
  audio-visual).
- **wang2023lora and laakkonen2026metalora** already apply LoRA to SSL deepfake detectors.

The answer to the reviewer should therefore place the contribution in the combination:

- Hindi–English intra-utterance code-mixing, audio-only;
- a matched G.711 telephone condition, in which train and test share the channel;
- speaker-disjoint LoRA adaptation, measured against a native full-fine-tuning ceiling;
- a held-out unseen attack (Tortoise).

None of these four parts is new on its own. related_work.tex also says it does not review
"detection for languages other than English and Hindi–English". The reviewer is asking
for exactly that review, so the multilingual entries above are there to fill it.

---

## 4. Discrepancies found in the existing entries

1. **baevski2020wav2vec:** The second author was "Zhou, Henry", which is the arXiv form.
   The NeurIPS proceedings list him as **"Zhou, Yuhao"**, and the .bib now uses the
   proceedings form. The entry also had no volume, pages or publisher. The booktitle had
   a "(NeurIPS)" suffix, which I removed to match the requested venue style.
2. **affectdf2026:** The author "Suresh Chandra, Shreeram" put "Suresh Chandra" as the
   family name. His ISCA-indexed paper (Odyssey 2024, `chandra24_odyssey`) and Crossref
   both split it as **family "Chandra", given "Shreeram Suresh"**, and the related-work
   notes also list him as "Chandra". Changed to "Chandra, Shreeram Suresh".
   - AffectDF is still arXiv v1 (6 Aug 2026), with no journal reference and no Crossref
     record.
   - The same group's peer-reviewed predecessor exists: Mahapatra et al., "Can Emotion
     Fool Anti-spoofing?", Interspeech 2025, pp. 5628--5632, DOI
     10.21437/Interspeech.2025-1234. It is verified but not added. Cite it alongside
     AffectDF if a peer-reviewed anchor is wanted.
3. **todisco2019asvspoof:** Minor. ISCA and Crossref give "Sahidullah, Md." and
   "Kinnunen, Tomi H."; the old entry had "Md" and "Tomi". Aligned to ISCA. Pages
   1008--1012 were already correct; the DOI was missing.
4. **tak2022wav2vec:** Pages and DOI were missing (now 112--119,
   10.21437/Odyssey.2022-16). The published title capitalises "Wav2vec 2.0"; I kept the
   lowercase "{wav2vec} 2.0" brand form, which is cosmetic only.
5. **hu2022lora:** The booktitle was the long form; normalised to {Proc. ICLR}. Nothing
   else was wrong.
6. **betker2023tortoise:** The arXiv title is sentence case ("Better speech synthesis
   through scaling"); the entry keeps title case, which is cosmetic. The current arXiv
   version is v2 (23 May 2023).
7. **okabe2018attentive, diwan2021mucs, casanova2024xtts:** Titles, authors and pages
   were all correct; only the DOI and publisher were missing.
8. **rvc2023:** The URL was inside `howpublished`; it is now a proper `url` field with an
   access date.

**Discrepancies in new entries, resolved in favour of the published record:**

- **yi2022add (ADD 2022):** The IEEE/Crossref published record lists **17 authors**. The
  arXiv v3 (revised 2024) lists 20, adding Xiaohui Zhang, Zheng Lian and Bin Liu. The
  .bib uses the published list.
- **wang2023lora:** The CEUR-WS published version has **7 authors** and adds Xinrui Yan;
  arXiv has 6. The .bib uses CEUR.
- **muller2022generalize:** ISCA spells "Diekmann, Franziska" and "Müller, Nicolas"; arXiv
  spells "Dieckmann" and "Nicolas M. Müller". The .bib uses ISCA.
- **tak2022rawboost:** The IEEE metadata title reads "Rawboost"; the .bib keeps the
  authors' "RawBoost".

**Verified but not added (optional):**

- CodecFake, Wu, Tseng and Lee, Interspeech 2024, pp. 1770--1774, DOI
  10.21437/Interspeech.2024-2093. It is about neural-codec *synthesis* attacks rather
  than transmission codecs, so it was left out to avoid conflating the two meanings of
  "codec".
- "Are audio DeepFake detection models polyglots?" (arXiv 2412.17924) and HAV-DF (arXiv
  2411.15457) have no peer-reviewed venue that I could find, so both were dropped.

## Week 14 additions (checked 2026-10-25 against the arXiv abstract pages)

- **xue2026rtcfake:** "RTCFake: Speech Deepfake Detection in Real-Time Communication",
  Xue, Yi, Huang, Ren, Chen, Fan, Su, Zhang, Cai; arXiv 2604.23742 (26 April 2026), comment
  "Accepted by ACL 2026". No page numbers or DOI published yet, so none are given.
- **phukan2026seacf:** "Bridging the SEA Gap: An Initial Benchmark for Neural Audio
  Codec-Synthesized Speech Deepfakes in South-East Asian Languages", Chetia Phukan, Girish,
  Akhtar, Buduru; arXiv 2606.15968 (14 June 2026), comment "Accepted to IJCAI-ECAI 2026".
- **hoang2026vietprism:** "VietPrism: A large-scale Vietnamese speech and deepfake corpus
  with diverse dialects and code-switching", Hoang, Le; arXiv 2609.30005 (24 September 2026),
  a preprint submitted to ICASSP 2027. An exception to the no-preprint rule above, because it
  is the closest code-switching deepfake corpus and the novelty statement must account for
  it; it is labelled as a preprint in the bibliography.

**Verified but not added:** ML-ITW (Li et al., arXiv 2603.05852, a 14-language in-the-wild
benchmark, preprint; multilingual corpora are already represented by MLAAD and SpeechFake)
and BabelFake (Segna et al., arXiv 2610.06339, audio-visual, five European languages, no
Hindi or code-switching).
