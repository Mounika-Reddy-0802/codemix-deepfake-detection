# Real phone-call test: participant consent and mentor approval

**Project:** Quantifying and Closing the Code-Mixing Generalisation Gap in Audio
Deepfake Detection, with a Live Call-Detection Demonstrator

**Addendum to:** the mentor's ethics and data-use sign-off for the v2 scope (12 August
2026). That sign-off covered public corpora only and stated that no data would be
collected from any individual. This test records and clones the voices of the three
team members, so it needs each member's written consent and the mentor's fresh approval.

---

## 1. What will happen

About 30 phone calls between two phones over a normal mobile network, recorded on the
receiving phone (protocol: `docs/real_calls/PROTOCOL.md`):

- **Genuine calls.** You speak on a call, reading a prepared script or talking freely.
- **Cloned calls.** From about 20 seconds of your reading, two voice-cloning tools
  (XTTS-v2 and Chatterbox) make synthetic speech in your voice saying the prepared
  scripts, and that speech is played into a call. Script B is written like a fraud call
  (an urgent request for money and a one-time code), because that is the situation the
  detector is for.

## 2. What is recorded and made

- Your reference recordings (about 20 seconds, and two readings of about a minute each).
- The recordings of the calls you take part in.
- Synthetic speech in your voice (about 24 short sentences per tool, joined into calls).

## 3. How it is kept and used

1. **Evaluation only.** The recordings and clones are used only to test the detector. They
   are never used to train any model; the project's code rejects any training data that
   contains them.
2. **Private.** They stay on team laptops and in one private Kaggle dataset used to make
   the clones. They are never committed to the project repository, never shared, and
   never published.
3. **What is published.** Only scores and verdicts per call, under the codes P1, P2 and
   P3. Your name never appears next to your code anywhere except on this form.
4. **Deleted.** All recordings and clones, and the Kaggle dataset, are deleted within 30
   days of the project's final evaluation.
5. **Never misused.** The clones are never played to anyone outside a test call between
   the three of us, and never used to contact any other person.

## 4. Your choices

- You may withdraw at any time before the results are submitted for publication. Your
  recordings, your clones and your calls' results are then deleted and left out.
- You may agree to the genuine calls but not to cloning (tick below).

---

## Participant consent

I have read this form. I agree to take part as described, on the conditions in Section 3.

| Code | Name | Genuine calls | Cloning of my voice | Signature | Date |
|---|---|---|---|---|---|
| P1 | | yes / no | yes / no | | |
| P2 | | yes / no | yes / no | | |
| P3 | | yes / no | yes / no | | |

## Mentor approval

I approve the real phone-call test described above: recording calls between the three
consenting team members, cloning their voices for evaluation only, and reporting the
anonymised results in the project report and paper.

Name: ______________________  Signature: ______________________  Date: __________

---

*Keep the signed copy only on team machines: scan it to
`docs/ethics/real_call_consent_<date>.pdf`. PDFs and images in `docs/ethics/` are kept out
of git by `.gitignore`.*
