# Early-warning study: a stricter speech gate for the first seconds of a call

Gate: windows ending in the first 10 s of a call and quieter than -29.87 dBFS are treated as silence (the level below which 1% of the dev speech windows fall; chosen on dev only).

## deployed_recorded (83 of 120 windows in the first seconds gated)

| Calls | Tool | n | Warned | Likely cloned | Median first caution (s) | Warned calls |
|---|---|---|---|---|---|---|
| clone_replay (without gate) | chatterbox | 6 | 6 | 6 | 7 | C25, C26, C27, C28, C29, C30 |
| clone_replay (without gate) | xtts_v2 | 6 | 6 | 6 | 9 | C19, C20, C21, C22, C23, C24 |
| genuine_live (without gate) | none | 12 | 8 | 8 | 6 | C05, C06, C07, C08, C09, C10, C11, C12 |
| genuine_replay (without gate) | none | 6 | 6 | 6 | 9 | C13, C14, C15, C16, C17, C18 |
| clone_replay (with gate) | chatterbox | 6 | 6 | 6 | 14 | C25, C26, C27, C28, C29, C30 |
| clone_replay (with gate) | xtts_v2 | 6 | 6 | 6 | 14 | C19, C20, C21, C22, C23, C24 |
| genuine_live (with gate) | none | 12 | 8 | 8 | 8 | C05, C06, C07, C08, C09, C10, C11, C12 |
| genuine_replay (with gate) | none | 6 | 6 | 6 | 14 | C13, C14, C15, C16, C17, C18 |

## deployed_g711 (83 of 120 windows in the first seconds gated)

| Calls | Tool | n | Warned | Likely cloned | Median first caution (s) | Warned calls |
|---|---|---|---|---|---|---|
| clone_replay (without gate) | chatterbox | 6 | 6 | 6 | 7 | C25, C26, C27, C28, C29, C30 |
| clone_replay (without gate) | xtts_v2 | 6 | 6 | 6 | 9 | C19, C20, C21, C22, C23, C24 |
| genuine_live (without gate) | none | 12 | 8 | 8 | 6 | C05, C06, C07, C08, C09, C10, C11, C12 |
| genuine_replay (without gate) | none | 6 | 6 | 6 | 9 | C13, C14, C15, C16, C17, C18 |
| clone_replay (with gate) | chatterbox | 6 | 6 | 6 | 14 | C25, C26, C27, C28, C29, C30 |
| clone_replay (with gate) | xtts_v2 | 6 | 6 | 6 | 14 | C19, C20, C21, C22, C23, C24 |
| genuine_live (with gate) | none | 12 | 8 | 8 | 8 | C05, C06, C07, C08, C09, C10, C11, C12 |
| genuine_replay (with gate) | none | 6 | 6 | 6 | 14 | C13, C14, C15, C16, C17, C18 |

## final_recorded (83 of 120 windows in the first seconds gated)

| Calls | Tool | n | Warned | Likely cloned | Median first caution (s) | Warned calls |
|---|---|---|---|---|---|---|
| clone_replay (without gate) | chatterbox | 6 | 2 | 0 | 36 | C27, C30 |
| clone_replay (without gate) | xtts_v2 | 6 | 5 | 5 | 18 | C19, C20, C21, C23, C24 |
| genuine_live (without gate) | none | 12 | 4 | 2 | 8 | C05, C06, C08, C10 |
| genuine_replay (without gate) | none | 6 | 1 | 1 | 14 | C17 |
| clone_replay (with gate) | chatterbox | 6 | 1 | 0 | 64 | C30 |
| clone_replay (with gate) | xtts_v2 | 6 | 6 | 5 | 18 | C19, C20, C21, C22, C23, C24 |
| genuine_live (with gate) | none | 12 | 5 | 1 | 10 | C05, C06, C07, C08, C10 |
| genuine_replay (with gate) | none | 6 | 1 | 1 | 18 | C17 |

## final_g711 (83 of 120 windows in the first seconds gated)

| Calls | Tool | n | Warned | Likely cloned | Median first caution (s) | Warned calls |
|---|---|---|---|---|---|---|
| clone_replay (without gate) | chatterbox | 6 | 1 | 0 | 8 | C27 |
| clone_replay (without gate) | xtts_v2 | 6 | 5 | 5 | 18 | C19, C20, C21, C23, C24 |
| genuine_live (without gate) | none | 12 | 3 | 1 | 8 | C05, C06, C08 |
| genuine_replay (without gate) | none | 6 | 1 | 1 | 16 | C17 |
| clone_replay (with gate) | chatterbox | 6 | 0 | 0 | - | - |
| clone_replay (with gate) | xtts_v2 | 6 | 5 | 5 | 20 | C19, C20, C21, C23, C24 |
| genuine_live (with gate) | none | 12 | 1 | 0 | 16 | C05 |
| genuine_replay (with gate) | none | 6 | 1 | 0 | 18 | C17 |

## ssl_aasist_recorded (83 of 120 windows in the first seconds gated)

| Calls | Tool | n | Warned | Likely cloned | Median first caution (s) | Warned calls |
|---|---|---|---|---|---|---|
| clone_replay (without gate) | chatterbox | 6 | 3 | 2 | 12 | C28, C29, C30 |
| clone_replay (without gate) | xtts_v2 | 6 | 6 | 6 | 10 | C19, C20, C21, C22, C23, C24 |
| genuine_live (without gate) | none | 12 | 0 | 0 | - | - |
| genuine_replay (without gate) | none | 6 | 1 | 1 | 40 | C18 |
| clone_replay (with gate) | chatterbox | 6 | 4 | 1 | 14 | C27, C28, C29, C30 |
| clone_replay (with gate) | xtts_v2 | 6 | 6 | 6 | 14 | C19, C20, C21, C22, C23, C24 |
| genuine_live (with gate) | none | 12 | 0 | 0 | - | - |
| genuine_replay (with gate) | none | 6 | 1 | 1 | 40 | C18 |

## ssl_aasist_g711 (83 of 120 windows in the first seconds gated)

| Calls | Tool | n | Warned | Likely cloned | Median first caution (s) | Warned calls |
|---|---|---|---|---|---|---|
| clone_replay (without gate) | chatterbox | 6 | 3 | 1 | 14 | C28, C29, C30 |
| clone_replay (without gate) | xtts_v2 | 6 | 6 | 6 | 10 | C19, C20, C21, C22, C23, C24 |
| genuine_live (without gate) | none | 12 | 1 | 0 | 48 | C08 |
| genuine_replay (without gate) | none | 6 | 1 | 1 | 40 | C18 |
| clone_replay (with gate) | chatterbox | 6 | 4 | 1 | 14 | C27, C28, C29, C30 |
| clone_replay (with gate) | xtts_v2 | 6 | 6 | 6 | 14 | C19, C20, C21, C22, C23, C24 |
| genuine_live (with gate) | none | 12 | 1 | 0 | 48 | C08 |
| genuine_replay (with gate) | none | 6 | 1 | 1 | 40 | C18 |
