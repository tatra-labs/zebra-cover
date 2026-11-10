# Where the encodings go wrong

Produced by `python experiments/diagnose.py` from the stored replies in `results/runs/`.

## Qwen3-0.6B

30 puzzles, first round.

- clue calls that mean the same as the gold call for that clue: **0/252 (0%)**
- the same, after mapping near-miss tool names to the closest real tool: 0/252 (0%)
- of the wrong ones, the right relation with left and right swapped: 0
- by puzzle size: 2x3 0/12, 3x3 0/24, 3x4 0/28, 4x4 0/48, 4x5 0/56, 5x5 0/84
- feedback-round replies with no `init_zebra` call (so the whole reply is rejected): 78/90

Re-scoring the stored replies with a lenient reader (tool-name aliases, and a feedback round with no `init_zebra` reuses the previous round's): first round solves **0/30** (2x3 0, 3x3 0, 3x4 0, 4x4 0, 4x5 0, 5x5 0), any round solves 0/30. The first-round number is exact for a one-shot run with that reader, since the first reply does not depend on feedback. The any-round number is not a rerun: with a different verdict the model would have been shown different feedback.

Puzzles whose first reply has at least one error of each kind:

| kind | puzzles |
|---|---:|
| init_zebra missing, repeated or malformed | 30 |

## Qwen3-1.7B

30 puzzles, first round.

- clue calls that mean the same as the gold call for that clue: **135/252 (54%)**
- the same, after mapping near-miss tool names to the closest real tool: 201/252 (80%)
- of the wrong ones, the right relation with left and right swapped: 5
- by puzzle size: 2x3 8/12, 3x3 12/24, 3x4 17/28, 4x4 33/48, 4x5 18/56, 5x5 47/84
- feedback-round replies with no `init_zebra` call (so the whole reply is rejected): 12/82

Re-scoring the stored replies with a lenient reader (tool-name aliases, and a feedback round with no `init_zebra` reuses the previous round's): first round solves **7/30** (2x3 1, 3x3 2, 3x4 2, 4x4 1, 4x5 1, 5x5 0), any round solves 7/30. The first-round number is exact for a one-shot run with that reader, since the first reply does not depend on feedback. The any-round number is not a rerun: with a different verdict the model would have been shown different feedback.

Puzzles whose first reply has at least one error of each kind:

| kind | puzzles |
|---|---:|
| unknown tool name | 23 |
| attribute not declared in init_zebra | 5 |
