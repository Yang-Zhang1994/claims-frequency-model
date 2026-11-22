# claims-frequency-model

Predicts how often a motor insurance policy will claim, using the public
**freMTPL2freq** dataset: 678,013 French motor third-party liability policies,
a standard benchmark in actuarial machine learning. The workflow is the one an
actuarial pricing team uses for claim frequency: check the data in SQL, model
claims per policy-year with exposure as the weight, then compare models on
held-out policies. Two neural models in PyTorch sit alongside the scikit-learn ones, and an MCP server
lets an AI agent score and explain policies with them.

```
OpenML ──► SQLite "policies" ──► SQL quality checks + capped "model_table" ──► train / test (75 / 25)
                                                                                 │
   baseline · Poisson GLM · gradient boosting (scikit-learn) · neural net · GLM + neural (PyTorch) ◄──┘
                                   │
                       deviance · Gini · calibration ──► output/results.md, lorenz.png, calibration.png
                                   │
                       MCP server (score, explain, compare, monitor) ◄── Claude Code or any MCP client
```

## Results (held-out 169,504 policies)

| Model | Poisson deviance ↓ | Gini ↑ | Predicted / observed claims |
|---|---|---|---|
| Baseline (portfolio mean) | 0.6258 | −0.010 | 0.995 |
| Poisson GLM | 0.5962 | 0.274 | 0.992 |
| Gradient boosting (Poisson loss) | **0.5743** | **0.330** | 0.996 |
| Neural net (PyTorch, embeddings) | 0.5859 | 0.296 | 0.990 |
| GLM + neural correction (CANN, PyTorch) | 0.5865 | 0.299 | 0.989 |

- Both models beat the baseline. Gradient boosting separates risks best: its
  riskiest 10% of exposure has 7.6× the claim frequency of its safest 10%
  (0.309 vs 0.041 claims per policy-year) and carries about 31% of all claims.
- It stays calibrated. Predicted frequency is within 0.004 of observed in
  every decile except the top two (within 0.007), and total predicted claims
  are within 0.5% of actual.
- The GLM is not far behind and is much easier to explain to a rating
  committee, since each factor is a multiplier. In practice I would show both:
  the GLM as the rating structure and the boosted model as a check on what it misses.
- The neural nets beat the GLM (Gini 0.296–0.299 vs 0.274) but not gradient boosting. On tabular
  data of this size that is the usual result; the network's value here is the comparison, and that
  it learns smooth age and density effects without hand-made bands.

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

## Neural models (PyTorch)

`claims_model/neural.py` wraps a PyTorch network in the same `fit(X, frequency, sample_weight=exposure)`
/ `predict` interface as the scikit-learn models, so `run.py` evaluates all five the same way.

| Choice | Detail |
|---|---|
| Inputs | Learned embeddings for area, brand, fuel and region (an unseen category maps to its own slot); standardized vehicle power, vehicle age, driver age, bonus-malus and log density |
| Network | Two hidden layers (32, 16, tanh) giving log-frequency |
| Exposure | `log(exposure)` is an offset, so the loss is Poisson on claim counts, the same weighting the other models get |
| Starting point | The output layer starts at zero, so training begins at the portfolio mean (or, for CANN, exactly at the GLM) |
| Training | Adam, batches of 2,048, early stopping on 10% of the training policies, fixed seed (two runs give identical results) |
| Overall level | Early stopping left predictions 2–4% high in total, so one factor rescales them to match the training claims (the balance property a Poisson GLM has by construction). Computed on training data only |

CANN (combined actuarial neural network) starts from the fitted GLM's prediction and learns only a
correction, which keeps the GLM's structure as the backbone. Here it matched the plain network.

On synthetic data with known effects, the network recovers a 2× bonus-malus effect from 80,000 policies
but not from 20,000 (early stopping on a validation split with ~150 claims is too noisy), while the GLM
recovers it from 20,000. The test suite checks both.

## Agent access through MCP

`claims_model/mcp_server.py` is a [Model Context Protocol](https://modelcontextprotocol.io) server
(official Python SDK) that exposes the trained models as tools. The agent never sees the training data.

| Tool | What it returns |
|---|---|
| `score_policy(policy)` | Claims per year from all four models, the policy's risk decile on the held-out book, and its ratio to a typical policy |
| `explain_policy(policy)` | The GLM's multiplier for each rating factor, from a typical policy to this one, ranked. They multiply out exactly to the GLM's prediction |
| `compare_models()` | The held-out scores table |
| `monitoring_status()` | The latest drift and actual/expected report |
| resource `claims://model-card` | [Model card](docs/model-card.md): data, models, intended use and what not to use them for |

Inputs are typed and range-checked (a 12-year-old driver or region `R99` is rejected, naming the field),
and values above the training caps are capped exactly as the SQL prep caps the training data.

`.mcp.json` registers the server with Claude Code. [docs/agent-session/transcript.md](docs/agent-session/transcript.md)
is a recorded session: asked which of three new policies is riskiest and whether to trust the model,
Claude Code scored and explained all three, checked monitoring, and answered. Its answer also flagged
two real weaknesses: the networks disagree with the other models on one policy (0.288 vs ~0.18), and
the GLM gives young drivers an apparent discount because bonus-malus 100 already carries their risk.

## Monitoring the model after deployment

`monitor.py` treats the trained gradient-boosting model as if it were in production and scores three
simulated quarters of new policies. It writes `output/monitoring_report.json`, which a weekly GitHub
Actions job refreshes and commits.

| Check | What it catches | Alert rule |
|---|---|---|
| **Population stability index (PSI)** per input feature, against the training data | The book of business changing (younger drivers, a new region, a new vehicle brand) | ≥ 0.10 watch, ≥ 0.25 alert |
| **Actual / expected claims** once claims are in | The model's overall level going stale, e.g. claim frequency trending up | outside 0.90–1.10 |
| **Gini** on the new batch | The model ranking risks worse than at validation | below 80% of the validation Gini |

Poisson deviance is reported too, but not used for alerts: it rises on any riskier book even when the model is right.

| Simulated batch | What changed | Result |
|---|---|---|
| `steady` | Nothing (more of the same policies) | **ok**: A/E 0.988, every PSI < 0.001 |
| `mix_shift` | Younger, higher bonus-malus book | **watch**: PSI 0.62 on driver age and 0.42 on bonus-malus, but A/E stays 0.981, so the model still prices the new mix correctly |
| `claims_trend` | Same policies, 20% more claims | **alert**: inputs look normal (PSI < 0.001) but A/E is 1.204, so it needs retraining |

The point of keeping both checks: input drift alone does not mean the model is wrong (`mix_shift`), and a
model can go stale while every input looks normal (`claims_trend`).

### Weekly alert email (low-code, n8n)

[`automation/n8n/model-health-alert.json`](automation/n8n/model-health-alert.json) is an n8n workflow that
turns the report into an email for whoever owns the model:

```
Every Monday 10:00 ─► Fetch health report (HTTP) ─► Needs attention? ── no ─► Healthy, no alert
                                                         │ yes
                       One item per batch ─► Keep flagged batches ─► Build HTML table ─► Compose alert ─► Send email
```

It uses only built-in nodes (Schedule, HTTP Request, If, Split Out, Filter, HTML, Edit Fields, Send Email).
To use it, import the file in n8n (*Workflows → Import from file*), attach an SMTP credential to **Send email**,
set the recipient, and activate it.

`automation/n8n/test_n8n_workflow.py` runs the workflow in the n8n CLI against two local fixture reports.
On the alert report it checks that the email subject reads `Claims model: ALERT (2 batch(es) flagged)` and
that the table lists only the two flagged batches. On a healthy report it checks that the flow stops
without building an email. CI runs it on every push. The email node is removed for these runs, because
CI has no SMTP account.

```bash
npm install -g n8n
python -m unittest discover -s automation/n8n -v
```

[`docs/power-automate-alert.md`](docs/power-automate-alert.md) builds the same flow in Power Automate.

## Snowflake version of the pipeline

`snowflake/` runs the same data prep in Snowflake: `setup.sql` creates an auto-suspending XS warehouse,
database and schemas; `load_to_snowflake.py` loads the raw 678,013 policies with `write_pandas`, builds
`ANALYTICS.MODEL_TABLE` and the data-quality and bonus-malus views from `transform.sql`, then
**reconciles** row counts, claim totals and every quality check against the local SQLite pipeline and fails
if any differ.

```bash
pip install -r snowflake/requirements.txt
export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PRIVATE_KEY_PATH=~/.ssh/snowflake_key.p8
python snowflake/load_to_snowflake.py
```

Key-pair setup (one time): `openssl genrsa 2048 | openssl pkcs8 -topk8 -nocrypt -out ~/.ssh/snowflake_key.p8`,
`openssl rsa -in ~/.ssh/snowflake_key.p8 -pubout`, then in a Snowflake worksheet
`ALTER USER <you> SET RSA_PUBLIC_KEY='<key body without the BEGIN/END lines>';`.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v      # 24 tests, ~5 s: SQL prep, metrics, PSI, A/E, the networks and the MCP server
python run.py              # downloads ~10 MB from OpenML on first run, ~20 s on a laptop (all five models)
python monitor.py          # scores the simulated batches, writes output/monitoring_report.json
claude                     # in this folder, Claude Code picks up .mcp.json and can call the claims tools
```

The new tests (`test_neural_and_mcp.py`) use synthetic policies with known effects: the network
recovers them, matches training claims in total, gives identical results for the same seed, handles
unseen categories, and CANN with no training equals the GLM. The MCP tests connect a real MCP client
to the server in-process and check every tool, the model card, the multipliers multiplying out, and
input validation.

## Limits and next steps

- Frequency only. A full pure premium also needs claim severity (a Gamma or
  Tweedie model on freMTPL2sev).
- One train/test split. Cross-validated deviance and a confidence band on the
  Gini would make the model comparison firmer.
- The networks' size and learning rate were set by hand, and they disagree with the other models on
  some policies; a small tuning run with cross-validation, and an ensemble, would be next.
- The boosted model's hyperparameters were set by hand. A small tuning run and
  partial-dependence plots would be the next things to add.
- Monitoring batches are simulated from held-out policies; a real deployment would feed each quarter's
  new business and claims into the same checks.
