# W10 -- normalised-bundle adapters, with and without RVC

Run mode: **full run**  
Generated: 2026-09-13 11:23 UTC  
Branch: `week10-mounika-unseen-attack-scoring`

EER % (lower is better), each adapter scored in its own condition:

```
set                     cm04  eval_pool  rvc_test
adapter                                          
lora_norm_channel      28.16       4.41     29.56
lora_norm_clean         5.44       1.46     23.91
lora_norm_rvc_channel  30.99       2.75      5.01
lora_norm_rvc_clean    12.01       1.99      5.01
```

Floors from `bundle_normalisation_v1.md`: 5.17% clean, 10.01% channel-matched.
Read every EER against those floors, not against 50%.

Pooled JSONs and per-clip scores are in `results/w10/`; the four adapter checkpoints are in the private dataset `codemix-w10-results`.
