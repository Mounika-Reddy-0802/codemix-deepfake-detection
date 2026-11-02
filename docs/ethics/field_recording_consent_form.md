# Field recordings for the voice check: participant consent and mentor approval

**Project:** Quantifying and Closing the Code-Mixing Generalisation Gap in Audio
Deepfake Detection, with a Live Call-Detection Demonstrator

**Addendum to:** the real phone-call test consent
([`real_call_consent_form.md`](real_call_consent_form.md)). That form covered recording
calls between the three team members and cloning their voices for test calls between them.
This test records the same three people on the laptop microphone and plays the clones
already made of their voices in two new ways (from a phone into the laptop, and into a call
on speaker), so it needs each member's written consent and the mentor's fresh approval.

---

## 1. What will happen

About 35 short recordings per person, in two sessions (protocol:
`docs/field_validation/PROTOCOL.md`):

- **Genuine recordings.** You read sentences from the real-call scripts and talk freely for
  about 15 seconds at a time into the laptop microphone; you record ten of the same lines
  on your own phone and play them back into the laptop; in the second session you speak on
  a phone call whose receiving phone is on speaker next to the laptop, with music playing
  in the room.
- **Cloned recordings.** The XTTS-v2 and Chatterbox clones of your voice made for the
  real-call test are played from a phone into the laptop microphone, and into a call on
  speaker. No new clone of your voice is made.

## 2. What is recorded and made

- About 35 recordings of about 15 seconds each, as the laptop heard them (about 10
  minutes in all), and ten recordings on your own phone.
- A log of each recording's code (for example `P1_F01`), its label and the voice check's
  verdict.

## 3. How it is kept and used

1. **Evaluation only.** The recordings are used only to measure the voice check. They are
   never used to train or tune any model; the project's code rejects any training data
   whose source is `field_validation`.
2. **Private.** They stay on team laptops in `data/field_validation/`. They are never
   committed to the project repository, never uploaded, never shared and never published.
3. **What is published.** Only scores and verdicts per recording, under the codes P1, P2
   and P3. Your name never appears next to your code anywhere except on this form.
4. **Deleted.** All recordings and the log are deleted within 30 days of the project's final
   evaluation, with the real-call recordings.
5. **Never misused.** The clones are played only in a room where the three of us are
   present, into the laptop or into a call between our own phones; never to anyone else and
   never into a call to anyone outside the team.

## 4. Your choices

- You may withdraw at any time before the results are submitted for publication. Your
  recordings and their results are then deleted and left out.
- You may agree to the genuine recordings but not to replaying the clones of your voice
  (tick below).

---

## Participant consent

I have read this form. I agree to take part as described, on the conditions in Section 3.

| Code | Name | Genuine recordings | Replaying the clones of my voice | Signature | Date |
|---|---|---|---|---|---|
| P1 | | yes / no | yes / no | | |
| P2 | | yes / no | yes / no | | |
| P3 | | yes / no | yes / no | | |

## Mentor approval

I approve the field recordings described above: recording the three consenting team
members on the laptop microphone and over calls between their own phones, replaying the
clones already made of their voices for evaluation only, and reporting the anonymised
results in the project report and paper.

Name: ______________________  Signature: ______________________  Date: __________

---

*Keep the signed copy only on team machines: scan it to
`docs/ethics/field_recording_consent_<date>.pdf`. PDFs and images in `docs/ethics/` are kept
out of git by `.gitignore`.*
