# Baseline comparison (frozen synth_v2)

## Headline metrics
| Method | Split | N acc | Open acc | Open MAE |
|---|---|---|---|---|
| KI-HMM v5a | test_x1 | 0.951 | 0.8527 | 0.1830 |
| KI-HMM v5a | test_x2 | 0.990 | 0.7547 | 0.3163 |
| KI-HMM v5a | test_x4 | 1.000 | 0.5080 | 0.6184 |
| SD-HMM | test_x1 | 0.401 | 0.6402 | 0.4980 |
| SD-HMM | test_x2 | 0.167 | 0.3124 | 1.1931 |
| SD-HMM | test_x4 | 0.234 | 0.2952 | 1.1253 |
| VND-HMM | test_x1 | 0.417 | 0.6367 | 0.5032 |
| VND-HMM | test_x2 | 0.167 | 0.2760 | 1.1999 |
| VND-HMM | test_x4 | 0.232 | 0.2819 | 1.1481 |
| Deep-Channel | test_x1 | 0.786 | 0.5290 | 0.5440 |
| Deep-Channel | test_x2 | 0.333 | 0.4344 | 0.7200 |
| Deep-Channel | test_x4 | 0.182 | 0.3172 | 1.0727 |

## Per-N breakdown (noise x1)
| True N | Metric | KI-HMM v5a | SD-HMM | VND-HMM | Deep-Channel |
|---|---|---|---|---|---|
| 1 | N acc | 1.000 | 0.304 | 0.362 | 1.000 |
| 1 | open acc | 0.991 | 0.707 | 0.718 | 0.885 |
| 2 | N acc | 1.000 | 0.849 | 0.814 | 1.000 |
| 2 | open acc | 0.964 | 0.863 | 0.848 | 0.656 |
| 3 | N acc | 0.973 | 0.575 | 0.562 | 0.863 |
| 3 | open acc | 0.881 | 0.738 | 0.743 | 0.497 |
| 4 | N acc | 0.860 | 0.209 | 0.267 | 0.174 |
| 4 | open acc | 0.725 | 0.495 | 0.480 | 0.408 |
| 5 | N acc | 0.929 | 0.000 | 0.014 | 0.986 |
| 5 | open acc | 0.705 | 0.378 | 0.378 | 0.203 |

## Capability matrix
| Method | N | open count | 7-state counts | rates | neural | CFTR |
|---|---|---|---|---|---|---|
| KI-HMM v5a | yes | yes | yes | yes (4 identifiable dirs) | hybrid | yes |
| SD-HMM | yes (BIC) | yes (Viterbi) | no | effective birth-death only | no | no |
| VND-HMM | yes (BIC) | yes (Viterbi) | no | transition probs only | no | no |
| Deep-Channel | max-openings heuristic | yes (per sample) | no | no | yes (RCNN) | no |
