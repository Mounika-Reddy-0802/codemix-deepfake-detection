# Real phone-call test

Written by `python -m src.inference.real_calls report`. Calls over a mobile
network, recorded on the receiving phone (protocol, and what happened on the day:
`docs/real_calls/PROTOCOL.md`). *recorded*: as the receiving phone recorded it;
*g711*: after an 8 kHz G.711 round trip, as the live system's carrier delivers it.
Counts are small; read every percentage with its number of calls.

Calls with a recording: 30.

## ssl_aasist_recorded (threshold 0.540, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 50.0% [18.8, 81.2] | 0.0% [0.0, 39.0] | 18 |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 11 |
| genuine_live | none | 12 | 0.0% [0.0, 24.3] | 0.0% [0.0, 24.3] | - |
| genuine_replay | none | 6 | 0.0% [0.0, 39.0] | 0.0% [0.0, 39.0] | - |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 3.85% | [158, 205] |
| genuine_replay | chatterbox | 43.12% | [158, 162] |
| genuine_live | xtts_v2 | 1.97% | [352, 205] |
| genuine_live | chatterbox | 29.59% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.816 | 0.809 | 0.880 |
| clone_replay | xtts_v2 | 0.146 | 0.057 | 0.041 |
| genuine_live | none | 0.969 | 0.976 | 0.975 |
| genuine_replay | none | 0.933 | 0.928 | 0.934 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 0 | 0 |
| P2 | clone_replay | chatterbox | 2 | 1 | 0 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 0 | 0 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 2 | 0 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 0 | 0 |
| P3 | genuine_replay | none | 2 | 0 | 0 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 66.7% | 0.0% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 0.0% | 0.0% |
| genuine_replay | none | 6 | 0.0% | 0.0% |

## ssl_aasist_g711 (threshold 0.540, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 50.0% [18.8, 81.2] | 0.0% [0.0, 39.0] | 18 |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 10 |
| genuine_live | none | 12 | 0.0% [0.0, 24.3] | 0.0% [0.0, 24.3] | - |
| genuine_replay | none | 6 | 0.0% [0.0, 39.0] | 0.0% [0.0, 39.0] | - |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 3.85% | [158, 205] |
| genuine_replay | chatterbox | 40.62% | [158, 162] |
| genuine_live | xtts_v2 | 2.36% | [352, 205] |
| genuine_live | chatterbox | 30.18% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.807 | 0.797 | 0.873 |
| clone_replay | xtts_v2 | 0.110 | 0.067 | 0.040 |
| genuine_live | none | 0.961 | 0.972 | 0.970 |
| genuine_replay | none | 0.928 | 0.931 | 0.927 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 0 | 0 |
| P2 | clone_replay | chatterbox | 2 | 1 | 0 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 0 | 0 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 2 | 0 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 0 | 0 |
| P3 | genuine_replay | none | 2 | 0 | 0 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 66.7% | 0.0% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 0.0% | 0.0% |
| genuine_replay | none | 6 | 0.0% | 0.0% |

