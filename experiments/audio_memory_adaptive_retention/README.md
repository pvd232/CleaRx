# Adaptive audio retention

This pilot asks whether a learned policy can allocate a fixed acoustic-position
budget better than assigning two positions to every item.

The A100 runner preserves every transcript embedding and measures 0, 2, and 4
appended acoustic positions with fusion strength fixed at 1. It fits on RAVDESS
actors 01--04, evaluates on actors 05--06, and gives four positions to half of
the held-out items and zero to the rest. That adaptive arm has the same total
budget as the two-position fixed arm.

The saved result is exploratory. Each recording and transcript condition is one
allocation unit; the experiment does not yet allocate positions among several
segments inside one prompt.
