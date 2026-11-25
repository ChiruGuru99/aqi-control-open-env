# Policy audit (after reward fixes)

Single-script audit over seeds 0-4. Values are terminal grader score mean ± population standard deviation.

| Task | noop | grap2 | grap4 | threshold | max_everything | anti_threshold | budget_aware | random | lagged_threshold |
|---|---|---|---|---|---|---|---|---|---|
| easy | 0.1500 ± 0.0000 | 0.5246 ± 0.0000 | 0.7385 ± 0.0000 | 0.8401 ± 0.0000 | 0.8207 ± 0.0000 | 0.0770 ± 0.0000 | 0.8401 ± 0.0000 | 0.6115 ± 0.0000 | 0.8675 ± 0.0000 |
| medium | 0.3000 ± 0.0000 | 0.6255 ± 0.0000 | 0.8166 ± 0.0000 | 0.8892 ± 0.0000 | 0.8805 ± 0.0000 | 0.2521 ± 0.0000 | 0.8892 ± 0.0000 | 0.6276 ± 0.0000 | 0.9081 ± 0.0000 |
| hard | 0.3000 ± 0.0000 | 0.4854 ± 0.0141 | 0.2401 ± 0.0075 | 0.4972 ± 0.0101 | 0.3215 ± 0.0223 | 0.2715 ± 0.0010 | 0.5465 ± 0.0116 | 0.2401 ± 0.0075 | 0.4432 ± 0.0137 |
| expert | 0.2646 ± 0.0024 | 0.3521 ± 0.0086 | 0.2909 ± 0.0021 | 0.4061 ± 0.0146 | 0.2906 ± 0.0022 | 0.2393 ± 0.0049 | 0.3311 ± 0.0072 | 0.3776 ± 0.0084 | 0.3533 ± 0.0158 |
| crisis | 0.3108 ± 0.0037 | 0.4193 ± 0.0081 | 0.3363 ± 0.0060 | 0.4852 ± 0.0161 | 0.3507 ± 0.0052 | 0.2949 ± 0.0041 | 0.3502 ± 0.0102 | 0.3687 ± 0.0050 | 0.4143 ± 0.0170 |

# Lagged-threshold trace (easy, seed 0)

| Day | Observed PM2.5 | Policy | Stage | pm25_base | pm25_post | Cost |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 142.0 | threshold | 2 | 142.0 | 134.1 | 10.00 |
| 1 | 142.0 | lagged_threshold | 2 | 142.0 | 134.1 | 10.00 |
| 2 | 142.0 | threshold | 2 | 158.0 | 149.2 | 10.00 |
| 2 | 142.0 | lagged_threshold | 2 | 158.0 | 149.2 | 10.00 |
| 3 | 158.0 | threshold | 2 | 183.8 | 173.4 | 10.00 |
| 3 | 158.0 | lagged_threshold | 2 | 183.8 | 173.4 | 10.00 |
| 4 | 175.0 | threshold | 2 | 207.9 | 196.3 | 10.00 |
| 4 | 175.0 | lagged_threshold | 2 | 207.9 | 196.3 | 10.00 |
| 5 | 198.0 | threshold | 2 | 205.4 | 193.7 | 10.00 |
| 5 | 198.0 | lagged_threshold | 2 | 205.4 | 193.7 | 10.00 |
| 6 | 210.0 | threshold | 2 | 172.3 | 162.6 | 10.00 |
| 6 | 210.0 | lagged_threshold | 2 | 172.3 | 162.6 | 10.00 |
| 7 | 185.0 | threshold | 2 | 165.0 | 155.7 | 10.00 |
| 7 | 185.0 | lagged_threshold | 2 | 165.0 | 155.7 | 10.00 |
| 8 | 165.0 | threshold | 2 | 186.9 | 176.4 | 10.00 |
| 8 | 165.0 | lagged_threshold | 2 | 186.9 | 176.4 | 10.00 |
| 9 | 178.0 | threshold | 2 | 231.0 | 218.1 | 10.00 |
| 9 | 178.0 | lagged_threshold | 2 | 231.0 | 218.1 | 10.00 |
| 10 | 220.0 | threshold | 2 | 294.0 | 277.6 | 10.00 |
| 10 | 220.0 | lagged_threshold | 2 | 294.0 | 277.6 | 10.00 |

Days 1-10 are identical. The policies first diverge on day 11: `threshold` sees raw 245.0 and chooses Stage 2, while `lagged_threshold` reads the prior post-intervention 277.6 and chooses Stage 3. This post-intervention threshold crossing causes stage oscillation and changes fatigue; the observation's raw and post-intervention fields are internally correct.
