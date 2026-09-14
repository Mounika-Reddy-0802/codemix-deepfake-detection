# S3 native training and English scoring

Run mode: **full run**. Branch: `week10-mounika-unseen-attack-scoring`.

EER % (lower is better). Code-mixed sets are scored in each model's own condition.

```
set                     cm04  english  eval_pool  rvc_test
model                                                     
lora_norm_channel        NaN     2.55        NaN       NaN
lora_norm_clean          NaN    15.83        NaN       NaN
lora_norm_rvc_channel    NaN     2.42        NaN       NaN
lora_norm_rvc_clean      NaN     3.75        NaN       NaN
s3_native_channel      13.71    55.83       1.92     32.15
s3_native_clean        27.72    56.42       1.46     45.07
s3_native_rvc_channel  31.67    52.25       1.79      5.98
s3_native_rvc_clean    21.39    50.90       0.83      8.24
```

Floors: `docs/results/w10_norm_rvc.md`. Per-clip scores and pooled JSONs sit beside this file.
