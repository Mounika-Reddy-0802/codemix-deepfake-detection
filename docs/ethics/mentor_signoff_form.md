# Ethics & Data-Use Sign-Off — Faculty Mentor Approval

**Project:** Quantifying and Closing the Code-Mixing Generalisation Gap in Audio
Deepfake Detection, with a Live Call-Detection Demonstrator

**Programme:** B.Tech CSE (Data Science), Semester 7 — Capstone Project
**Team:** M. Lahari (D006) · S. Mounika Reddy (D032) · K. Sai Krishna Reddy (D034)
**Repository:** github.com/Mounika-Reddy-0802/codemix-deepfake-detection
**Approval sought for:** the **revised (v2)** project scope, dated 12 August 2026

---

## 1. What the project does, in one paragraph

We are building a **detector** that identifies synthetic (AI-cloned) speech in
Hindi–English code-mixed telephone calls — the kind used in voice-based fraud in
India. Existing detectors are trained almost entirely on English and collapse on
code-mixed telephony audio; we measure that failure, reduce it, and demonstrate it
on a live call. To train and honestly test such a detector, **the project must
itself generate synthetic speech**. That generation is the ethically sensitive
activity, and it is what this form seeks approval for.

**No synthetic speech of any kind has been generated to date.** The code that
would generate it is written but is blocked in software until this form is signed
and uploaded (see Section 5).

---

## 2. What changed since the original plan — this is what needs fresh approval

| Item | Originally approved plan | Revised (v2) scope |
|---|---|---|
| Volume of generated speech | ~1,500–2,500 clips | **~4,000+ clips** |
| Attack families | XTTS-v2 (TTS) + Tortoise (held-out) | **+ RVC voice conversion** (10–15 per-speaker models) |
| External evaluation | none | **AffectDF test subset** (public benchmark, CC BY-NC 4.0, research use) |
| Everything else | — | unchanged |

The scientific reason for the increase: a detector trained on too little synthetic
speech, or on a single generator, learns the quirks of that one tool rather than
the general signature of synthesis. More volume and a second, structurally
different attack family (voice conversion vs text-to-speech) is what makes the
result trustworthy rather than an artefact.

---

## 3. Data sources and their licences

All corpora are **public research datasets**. No data is collected from any
individual by this project.

| Corpus | Licence | How we use it |
|---|---|---|
| ASVspoof 2019 LA | Research (ASVspoof licence) | Training the English baseline |
| MUCS 2021 Hindi-English (OpenSLR 104) | CC BY-SA 4.0 | Code-mixed speech; voice-cloning references |
| HiACC — **adult subset only** | CC BY 4.0 | Code-mixed evaluation speech |
| IndicVoices (AI4Bharat) | Research, gated | Hindi / Tamil evaluation |
| IndicSynth | **CC BY-NC 4.0** | Evaluation only |
| IndicTTS-Deepfake | Research | Evaluation only |
| AffectDF | **CC BY-NC 4.0** | External evaluation only |

**Generation tools:** Coqui XTTS-v2 (CPML — **non-commercial**), Tortoise-TTS
(Apache-2.0), RVC (open source).

Two licences — IndicSynth and XTTS-v2 — permit **non-commercial use only**. This
project is non-commercial academic research, which both permit. The team commits
to keeping it so.

---

## 4. The rules the team commits to

**4.1 — HiACC child audio is excluded entirely. This is absolute.**
The HiACC corpus contains 1,858 utterances from **children**. These are never used
for anything: not as genuine speech, not as a voice-cloning reference, not in any
data listing. They are quarantined into a separate folder at download time and the
pipeline refuses to read that folder.

> *Disclosure, in the interest of honesty:* on 12 August 2026 the team found that
> although the child folder was being quarantined, a later processing step would
> still have read it. **No child audio was ever actually processed** — the corpus
> has not been downloaded yet. The defect was fixed, an automated audit tool was
> written, and tests now make the exclusion impossible to reopen silently. We
> report this because a rule that is only checked by memory is not a rule.

**4.2 — Voice cloning is limited to adult speakers in licensed research corpora,**
solely to build detection training data.

**4.3 — No real, identifiable private individual will be cloned.** No team
member's voice, no third party's voice, no public figure's voice — not for testing,
not for the demonstration.

**4.4 — Generated clones are never released as a standalone clone set.** Any
release of derived data respects the source licence (MUCS is share-alike). The
detector code and trained model may be released; the raw synthetic speech will not
be published in a form usable as a cloning resource.

**4.5 — No attempt will be made to re-identify any speaker** in any corpus.

**4.6 — The demonstration is defensive.** The live call demo warns the **person
receiving** the call. It never signals the caller, never disconnects a call
automatically, and produces no output that would help someone commit fraud.

**4.7 — Non-commercial academic use only**, as required by the licences above.

**4.8 — Scientific integrity.** The English baseline is trained on ASVspoof 2019
only; evaluation-only corpora and the held-out attack tool never enter training
data. This is what makes the paper's central claim measurable rather than
circular, so it is an integrity commitment as much as a technical one.

**4.9 — Generated audio and credentials stay on team-controlled storage** and are
never committed to the public repository.

**4.10 — Every corpus and tool will be cited** as its licence requires, and the
licence restrictions above will be stated explicitly in the published paper.

---

## 5. How these rules are enforced in software, not just promised

| Rule | Enforcement |
|---|---|
| No generation before this signature | An **ethics gate** in the code refuses to load any voice-cloning model until a signed copy of this form exists in the repository. It has **no override switch** and cannot be bypassed by a setting. |
| Child-audio exclusion | Automated tests + an audit tool that reports any child-looking file and refuses to declare the check passed without a human signature. |
| No leakage between training and testing | An automated test suite that runs before every training job. |
| Speaker groups fixed in advance | Speaker lists are frozen and fingerprinted (SHA-256); any later change is detected. |

---

## 6. Declaration by the student team

We have read the rules in Section 4 and will abide by them for the duration of the
project. We understand that generating synthetic speech carries a risk of misuse
and that our obligation is to keep this work defensive, non-commercial, and
properly attributed.

| Name | Roll No. | Signature | Date |
|---|---|---|---|
| M. Lahari | D006 | | |
| S. Mounika Reddy | D032 | | |
| K. Sai Krishna Reddy | D034 | | |

---

## 7. Faculty mentor approval

I have reviewed the revised scope in Section 2, the data sources and licences in
Section 3, and the commitments in Section 4. I approve the team to proceed with
the generation of synthetic speech for the purpose of building and evaluating a
detector, subject to any conditions noted below.

**Decision** (tick one):

&nbsp;&nbsp;&nbsp; ☐ Approved &nbsp;&nbsp;&nbsp;&nbsp; ☐ Approved with the conditions below &nbsp;&nbsp;&nbsp;&nbsp; ☐ Not approved

**Conditions / remarks:**

<br>

_______________________________________________________________________________

<br>

_______________________________________________________________________________

<br>

_______________________________________________________________________________

<br>

| | |
|---|---|
| **Name** | ______________________________________ |
| **Designation** | ______________________________________ |
| **Department** | ______________________________________ |
| **Signature** | ______________________________________ |
| **Date** | ______________________________________ |

<br>

*Institution stamp (if required):*

<br><br><br>

---

*This form and the supporting licence register (`docs/licences.md`) are maintained
in the project repository. The signed copy is scanned and stored at
`docs/ethics/mentor_signoff_<yyyy-mm-dd>.pdf`.*
