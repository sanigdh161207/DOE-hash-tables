# Scaling Experiment Empirical Interpretation Report

## Empirical Log-Log Slopes
| Strategy | Empirical Log-Log Slope | Theoretical Complexity | Observed Behavior |
|---|---|---|---|
| baseline | 0.915 | O(1) | Near-polynomial scaling |
| cosine | 0.907 | O(N * M) | Near-polynomial scaling |
| weighted_cosine | 1.004 | O(N * M) | Near-polynomial scaling |
| sparse | 0.362 | O(NNZ) | Near-polynomial scaling |

## Discussion of Deviations
Empirical slope measures combined fit and recommendation latency across sizes 10 to 1000 users.
Deviations from strict theoretical Big-O arise from CPU cache locality, NumPy vector C-level optimizations, and fixed memory allocation overhead at lower sample bounds.