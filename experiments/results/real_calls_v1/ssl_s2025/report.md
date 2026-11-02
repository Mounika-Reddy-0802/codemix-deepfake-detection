# Real phone-call test

Written by `python -m src.inference.real_calls report`. Calls over a mobile
network, recorded on the receiving phone (protocol, and what happened on the day:
`docs/real_calls/PROTOCOL.md`). *recorded*: as the receiving phone recorded it;
*g711*: after an 8 kHz G.711 round trip, as the live system's carrier delivers it.
Counts are small; read every percentage with its number of calls.

Calls with a recording: 30.

## ssl_aasist_recorded (threshold 0.334, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 0.0% [0.0, 39.0] | 0.0% [0.0, 39.0] | - |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 11 |
| genuine_live | none | 12 | 0.0% [0.0, 24.3] | 0.0% [0.0, 24.3] | - |
| genuine_replay | none | 6 | 0.0% [0.0, 39.0] | 0.0% [0.0, 39.0] | - |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 4.41% | [158, 205] |
| genuine_replay | chatterbox | 41.87% | [158, 162] |
| genuine_live | xtts_v2 | 1.97% | [352, 205] |
| genuine_live | chatterbox | 29.73% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.799 | 0.847 | 0.888 |
| clone_replay | xtts_v2 | 0.133 | 0.040 | 0.029 |
| genuine_live | none | 0.952 | 0.962 | 0.976 |
| genuine_replay | none | 0.872 | 0.900 | 0.933 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 0 | 0 |
| P2 | clone_replay | chatterbox | 2 | 0 | 0 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 0 | 0 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 0 | 0 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 0 | 0 |
| P3 | genuine_replay | none | 2 | 0 | 0 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 0.0% | 0.0% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 0.0% | 0.0% |
| genuine_replay | none | 6 | 0.0% | 0.0% |

## ssl_aasist_g711 (threshold 0.334, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 0.0% [0.0, 39.0] | 0.0% [0.0, 39.0] | - |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 11 |
| genuine_live | none | 12 | 0.0% [0.0, 24.3] | 0.0% [0.0, 24.3] | - |
| genuine_replay | none | 6 | 16.7% [3.0, 56.4] | 0.0% [0.0, 39.0] | 8 |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 3.85% | [158, 205] |
| genuine_replay | chatterbox | 41.25% | [158, 162] |
| genuine_live | xtts_v2 | 2.36% | [352, 205] |
| genuine_live | chatterbox | 30.18% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.782 | 0.834 | 0.881 |
| clone_replay | xtts_v2 | 0.117 | 0.050 | 0.029 |
| genuine_live | none | 0.945 | 0.957 | 0.971 |
| genuine_replay | none | 0.866 | 0.897 | 0.927 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 1 | 0 |
| P2 | clone_replay | chatterbox | 2 | 0 | 0 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 0 | 0 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 0 | 0 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 0 | 0 |
| P3 | genuine_replay | none | 2 | 0 | 0 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 0.0% | 0.0% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 0.0% | 0.0% |
| genuine_replay | none | 6 | 16.7% | 0.0% |

