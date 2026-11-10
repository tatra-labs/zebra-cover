| model | method | puzzles solved | cell accuracy | 2x3 | 3x3 | 3x4 | 4x4 | 4x5 | 5x5 | mean rounds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | plain | 0.0% | 30.6% | 0% | 0% | 0% | 0% | 0% | 0% | 1.00 |
| Qwen3-0.6B | nec | 0.0% | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 1.00 |
| Qwen3-0.6B | nec+feedback | 0.0% | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 4.00 |
| Qwen3-0.6B | nec-lenient | 0.0% | 0.0% | 0% | 0% | 0% | 0% | 0% | 0% | 1.00 |
| Qwen3-1.7B | plain | 13.3% | 43.7% | 80% | 0% | 0% | 0% | 0% | 0% | 1.00 |
| Qwen3-1.7B | nec | 3.3% | 3.3% | 20% | 0% | 0% | 0% | 0% | 0% | 1.00 |
| Qwen3-1.7B | nec+feedback | 3.3% | 5.6% | 20% | 0% | 0% | 0% | 0% | 0% | 3.73 |
| Qwen3-1.7B | nec-lenient | 23.3% | 28.8% | 20% | 40% | 40% | 20% | 20% | 0% | 1.00 |

First-round verdicts of the exact-cover loop:

| model | unique | under | over | count | invalid | none |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0 | 0 | 0 | 0 | 0 | 30 |
| Qwen3-1.7B | 2 | 3 | 1 | 0 | 24 | 0 |
