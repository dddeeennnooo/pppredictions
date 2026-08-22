# BTTS model experiments

All experiments use chronological selection/validation data and the same untouched
2025/26 holdout of 1,752 matches. Accuracy is the requested primary objective;
log loss and Brier score guard against improving accuracy with unusable probabilities.

| Branch | Experiment | Accuracy | Log loss | Brier | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `experiment/probability-calibration` | Platt calibration | 55.94% | 0.6874 | 0.2472 | Include |
| `experiment/dixon-coles-goal-model` | Time-decayed Dixon-Coles | 53.25% | 0.6973 | 0.2518 | Benchmark only |
| `experiment/no-quota-value-selection` | Validation threshold, no quota | 54.17% | 0.7258 | — | Opt in |
| `experiment/market-benchmark` | Closing-market Poisson proxy | 56.39% | 0.6822 | 0.2446 | Benchmark only |
| `experiment/market-residual` | Opening-market residual blend | 55.48% | 0.6863 | 0.2463 | Opt in |
| `experiment/rolling-evaluation` | Proper scores and block bootstrap | — | — | — | Include |

The closing-market result is not a deployable early prediction: closing prices contain
information that arrives after the opening snapshot. It is the benchmark the model
should try to approach and eventually beat using information available at prediction
time.

The combined branch is `experiment/combined-accuracy`. Its default preserves the
highest holdout accuracy and adds probability calibration. Other policies remain
explicit options:

```powershell
# Highest tested accuracy; calibrated probabilities
python -m src.main train-weekly-btts

# Resolve every match with a validation-selected threshold
python -m src.main train-weekly-btts --decision-policy threshold

# Allow opening-market residual blends into validation selection
python -m src.main train-weekly-btts --include-market-residual

# Independent goal-distribution benchmark
python -m src.main train-dixon-coles-btts

# Closing-market benchmark and uncertainty audit
python -m src.main benchmark-btts-market
python -m src.main audit-weekly-btts
```

The latest default audit reports a +2.05 percentage-point accuracy lift over the
always-BTTS-Yes baseline, with a 95% match-week block-bootstrap interval from -0.80
to +4.79 points. The interval crossing zero means another untouched season is needed
before treating the lift as stable.
