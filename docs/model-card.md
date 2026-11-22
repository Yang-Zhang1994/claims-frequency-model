# Model card: claim-frequency models (freMTPL2)

**Task.** Predict expected claims per policy-year for French motor third-party liability policies.

**Data.** freMTPL2freq (OpenML 41214): 678,013 policies, one year of observation. Capped in SQL:
claims at 4, exposure at 1 year, vehicle power at 12, vehicle age at 20, driver age at 90,
bonus-malus at 150. 75/25 train/test split (random_state 42); the network also holds out 10% of
training data for early stopping.

**Models.**
- Poisson GLM (scikit-learn): one-hot categoricals, age bands, log density. Multiplicative, so each
  factor's effect is explainable.
- Gradient boosting with Poisson loss (scikit-learn). Best held-out deviance and Gini.
- Neural net (PyTorch): embeddings for area, brand, fuel and region, plus scaled numerics; Poisson loss
  with log exposure as an offset; early stopping; total rescaled to match training claims.
- CANN (PyTorch): the same network learning a correction on top of the GLM's prediction.

**Intended use.** Illustration and comparison of frequency models; ranking and pricing analysis.

**Not for.** Pricing real policies, decisions about real people, or any market other than the one in
the data (France, one historical year). The data has no claim severity, so these models say nothing
about cost. Driver age and region are rating factors here; in a real deployment their use would need
legal and fairness review for the jurisdiction.
