# claims-frequency-model

Predicts how often a motor insurance policy will claim, using the public
**freMTPL2freq** dataset: 678,013 French motor third-party liability policies,
a standard benchmark in actuarial machine learning. The workflow is the one an
actuarial pricing team uses for claim frequency: check the data in SQL, model
claims per policy-year with exposure as the weight, then compare models on
held-out policies.

```
OpenML ──► SQLite "policies" ──► SQL quality checks + capped "model_table" ──► train / test (75 / 25)
                                                                                 │
              baseline · Poisson GLM · gradient boosting (Poisson loss) ◄────────┘
                                   │
                       deviance · Gini · calibration ──► output/results.md, lorenz.png, calibration.png
```

## Results (held-out 169,504 policies)

| Model | Poisson deviance ↓ | Gini ↑ | Predicted / observed claims |
|---|---|---|---|
| Baseline (portfolio mean) | 0.6258 | −0.010 | 0.995 |
| Poisson GLM | 0.5962 | 0.274 | 0.992 |
| Gradient boosting (Poisson loss) | **0.5743** | **0.330** | 0.996 |

- Both models beat the baseline. Gradient boosting separates risks best: its
  riskiest 10% of exposure has 7.6× the claim frequency of its safest 10%
  (0.309 vs 0.041 claims per policy-year) and carries about 31% of all claims.
- It stays calibrated. Predicted frequency is within 0.004 of observed in
  every decile except the top two (within 0.007), and total predicted claims
  are within 0.5% of actual.
- The GLM is not far behind and is much easier to explain to a rating
  committee, since each factor is a multiplier. In practice I would show both:
  the GLM as the rating structure and the boosted model as a check on what it misses.

![Lorenz curves](output/lorenz.png)
![Calibration by decile](output/calibration.png)

## Modelling choices

| Question | Choice | Why |
|---|---|---|
| Target | `claims / exposure`, weighted by `exposure` | A policy on the books for 3 months should count as a quarter of a policy-year |
| Loss | Poisson deviance | Claim counts are non-negative and rare; squared error suits neither |
| Bad records | Caps in SQL: claims at 4, exposure at 1 year, bonus-malus at 150 | 1,224 rows show > 1 year of exposure and 9 show 5–16 claims, which look like recording errors |
| GLM features | Ages binned into 10 quantile bands, `log(density)`, one-hot categoricals | A GLM is linear in log-frequency, so non-linear effects have to be built in |
| Boosted features | Raw numerics plus native categorical splits | Trees find thresholds and interactions on their own |
| Ranking metric | Exposure-weighted Gini from the Lorenz curve | Measures how well the model orders policies from safe to risky, which is what pricing needs |

The SQL step also produces a quick view of the strongest rating factor
(`output/results.md`): frequency rises from 0.082 at bonus-malus 50–59 to 0.312 at 100+.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v      # 6 tests: SQL prep, metrics, and a GLM recovering a known 3x rate
python run.py              # downloads ~10 MB from OpenML on first run, ~20 s on a laptop
```

## Limits and next steps

- Frequency only. A full pure premium also needs claim severity (a Gamma or
  Tweedie model on freMTPL2sev).
- One train/test split. Cross-validated deviance and a confidence band on the
  Gini would make the model comparison firmer.
- The boosted model's hyperparameters were set by hand. A small tuning run and
  partial-dependence plots would be the next things to add.
