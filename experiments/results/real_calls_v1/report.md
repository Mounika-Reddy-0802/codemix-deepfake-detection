# Real phone-call test

Written by `python -m src.inference.real_calls report`. Calls over a mobile
network, recorded on the receiving phone (protocol, and what happened on the day:
`docs/real_calls/PROTOCOL.md`). *recorded*: as the receiving phone recorded it;
*g711*: after an 8 kHz G.711 round trip, as the live system's carrier delivers it.
Counts are small; read every percentage with its number of calls.

Calls with a recording: 30.

## deployed_recorded (threshold 0.500, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 7 |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 9 |
| genuine_live | none | 12 | 66.7% [39.1, 86.2] | 66.7% [39.1, 86.2] | 6 |
| genuine_replay | none | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 9 |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 34.72% | [158, 205] |
| genuine_replay | chatterbox | 66.56% | [158, 162] |
| genuine_live | xtts_v2 | 16.0% | [352, 205] |
| genuine_live | chatterbox | 36.39% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.004 | 0.000 | 0.036 |
| clone_replay | xtts_v2 | 0.000 | 0.000 | 0.000 |
| genuine_live | none | 0.309 | 0.325 | 0.335 |
| genuine_replay | none | 0.000 | 0.000 | 0.005 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 2 | 2 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 2 | 2 |
| P2 | clone_replay | chatterbox | 2 | 2 | 2 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 4 | 4 |
| P2 | genuine_replay | none | 2 | 2 | 2 |
| P3 | clone_replay | chatterbox | 2 | 2 | 2 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 4 | 4 |
| P3 | genuine_replay | none | 2 | 2 | 2 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 100.0% | 100.0% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 66.7% | 66.7% |
| genuine_replay | none | 6 | 100.0% | 100.0% |

## deployed_g711 (threshold 0.500, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 7 |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 9 |
| genuine_live | none | 12 | 66.7% [39.1, 86.2] | 66.7% [39.1, 86.2] | 6 |
| genuine_replay | none | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 9 |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 34.97% | [158, 205] |
| genuine_replay | chatterbox | 70.94% | [158, 162] |
| genuine_live | xtts_v2 | 19.42% | [352, 205] |
| genuine_live | chatterbox | 45.12% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.049 | 0.004 | 0.103 |
| clone_replay | xtts_v2 | 0.000 | 0.000 | 0.000 |
| genuine_live | none | 0.379 | 0.357 | 0.346 |
| genuine_replay | none | 0.000 | 0.000 | 0.010 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 2 | 2 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 2 | 2 |
| P2 | clone_replay | chatterbox | 2 | 2 | 2 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 4 | 4 |
| P2 | genuine_replay | none | 2 | 2 | 2 |
| P3 | clone_replay | chatterbox | 2 | 2 | 2 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 4 | 4 |
| P3 | genuine_replay | none | 2 | 2 | 2 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 100.0% | 100.0% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 66.7% | 66.7% |
| genuine_replay | none | 6 | 100.0% | 100.0% |

## final_recorded (threshold 0.687, ladder [3, 5])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 33.3% [9.7, 70.0] | 0.0% [0.0, 39.0] | 36 |
| clone_replay | xtts_v2 | 6 | 83.3% [43.6, 97.0] | 83.3% [43.6, 97.0] | 18 |
| genuine_live | none | 12 | 33.3% [13.8, 60.9] | 16.7% [4.7, 44.8] | 8 |
| genuine_replay | none | 6 | 16.7% [3.0, 56.4] | 16.7% [3.0, 56.4] | 14 |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 15.96% | [158, 205] |
| genuine_replay | chatterbox | 49.37% | [158, 162] |
| genuine_live | xtts_v2 | 17.06% | [352, 205] |
| genuine_live | chatterbox | 42.46% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.841 | 0.919 | 0.930 |
| clone_replay | xtts_v2 | 0.613 | 0.514 | 0.349 |
| genuine_live | none | 0.702 | 0.886 | 0.962 |
| genuine_replay | none | 0.907 | 0.884 | 0.952 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 0 | 0 |
| P2 | clone_replay | chatterbox | 2 | 1 | 0 |
| P2 | clone_replay | xtts_v2 | 2 | 1 | 1 |
| P2 | genuine_live | none | 4 | 3 | 2 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 1 | 0 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 1 | 0 |
| P3 | genuine_replay | none | 2 | 1 | 1 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 16.7% | 0.0% |
| clone_replay | xtts_v2 | 6 | 83.3% | 83.3% |
| genuine_live | none | 12 | 33.3% | 16.7% |
| genuine_replay | none | 6 | 16.7% | 16.7% |

## final_g711 (threshold 0.687, ladder [3, 5])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 16.7% [3.0, 56.4] | 0.0% [0.0, 39.0] | 8 |
| clone_replay | xtts_v2 | 6 | 83.3% [43.6, 97.0] | 83.3% [43.6, 97.0] | 18 |
| genuine_live | none | 12 | 25.0% [8.9, 53.2] | 8.3% [1.5, 35.4] | 8 |
| genuine_replay | none | 6 | 16.7% [3.0, 56.4] | 16.7% [3.0, 56.4] | 16 |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 19.01% | [158, 205] |
| genuine_replay | chatterbox | 50.0% | [158, 162] |
| genuine_live | xtts_v2 | 14.7% | [352, 205] |
| genuine_live | chatterbox | 31.37% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.897 | 0.967 | 0.954 |
| clone_replay | xtts_v2 | 0.591 | 0.560 | 0.411 |
| genuine_live | none | 0.788 | 0.911 | 0.982 |
| genuine_replay | none | 0.924 | 0.900 | 0.960 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 0 | 0 |
| P2 | clone_replay | chatterbox | 2 | 1 | 0 |
| P2 | clone_replay | xtts_v2 | 2 | 1 | 1 |
| P2 | genuine_live | none | 4 | 3 | 1 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 0 | 0 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 0 | 0 |
| P3 | genuine_replay | none | 2 | 1 | 1 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 0.0% | 0.0% |
| clone_replay | xtts_v2 | 6 | 83.3% | 83.3% |
| genuine_live | none | 12 | 8.3% | 8.3% |
| genuine_replay | none | 6 | 16.7% | 16.7% |

## ssl_aasist_recorded (threshold 0.580, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 50.0% [18.8, 81.2] | 33.3% [9.7, 70.0] | 12 |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 10 |
| genuine_live | none | 12 | 0.0% [0.0, 24.3] | 0.0% [0.0, 24.3] | - |
| genuine_replay | none | 6 | 16.7% [3.0, 56.4] | 16.7% [3.0, 56.4] | 40 |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 4.41% | [158, 205] |
| genuine_replay | chatterbox | 36.56% | [158, 162] |
| genuine_live | xtts_v2 | 3.41% | [352, 205] |
| genuine_live | chatterbox | 32.69% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.834 | 0.771 | 0.872 |
| clone_replay | xtts_v2 | 0.158 | 0.073 | 0.054 |
| genuine_live | none | 0.986 | 0.989 | 0.974 |
| genuine_replay | none | 0.993 | 0.957 | 0.934 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 0 | 0 |
| P2 | clone_replay | chatterbox | 2 | 1 | 1 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 0 | 0 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 2 | 1 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 0 | 0 |
| P3 | genuine_replay | none | 2 | 1 | 1 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 66.7% | 33.3% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 0.0% | 0.0% |
| genuine_replay | none | 6 | 16.7% | 16.7% |

## ssl_aasist_g711 (threshold 0.580, ladder [2, 4])

| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |
|---|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 50.0% [18.8, 81.2] | 16.7% [3.0, 56.4] | 14 |
| clone_replay | xtts_v2 | 6 | 100.0% [61.0, 100.0] | 100.0% [61.0, 100.0] | 10 |
| genuine_live | none | 12 | 8.3% [1.5, 35.4] | 0.0% [0.0, 24.3] | 48 |
| genuine_replay | none | 6 | 16.7% [3.0, 56.4] | 16.7% [3.0, 56.4] | 40 |

| Genuine side | Tool | Window EER | Windows (genuine, clone) |
|---|---|---|---|
| genuine_replay | xtts_v2 | 4.41% | [158, 205] |
| genuine_replay | chatterbox | 36.56% | [158, 162] |
| genuine_live | xtts_v2 | 3.41% | [352, 205] |
| genuine_live | chatterbox | 32.69% | [352, 162] |

Mean window score (P(genuine)) by time in the call:

| Calls | Tool | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|---|
| clone_replay | chatterbox | 0.832 | 0.760 | 0.866 |
| clone_replay | xtts_v2 | 0.104 | 0.083 | 0.051 |
| genuine_live | none | 0.977 | 0.987 | 0.969 |
| genuine_replay | none | 0.990 | 0.965 | 0.927 |

Per speaker (calls warned and taken to 'likely cloned'):

| Speaker | Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|---|
| P1 | clone_replay | chatterbox | 2 | 0 | 0 |
| P1 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P1 | genuine_live | none | 4 | 0 | 0 |
| P1 | genuine_replay | none | 2 | 0 | 0 |
| P2 | clone_replay | chatterbox | 2 | 1 | 1 |
| P2 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P2 | genuine_live | none | 4 | 1 | 0 |
| P2 | genuine_replay | none | 2 | 0 | 0 |
| P3 | clone_replay | chatterbox | 2 | 2 | 0 |
| P3 | clone_replay | xtts_v2 | 2 | 2 | 2 |
| P3 | genuine_live | none | 4 | 0 | 0 |
| P3 | genuine_replay | none | 2 | 1 | 1 |

Diagnostic, chooses nothing: the same calls with each first window treated as
silence (how much of a verdict rests on the quiet before the first word):

| Calls | Tool | n | Caution | Likely cloned |
|---|---|---|---|---|
| clone_replay | chatterbox | 6 | 66.7% | 16.7% |
| clone_replay | xtts_v2 | 6 | 100.0% | 100.0% |
| genuine_live | none | 12 | 8.3% | 0.0% |
| genuine_replay | none | 6 | 16.7% | 16.7% |

