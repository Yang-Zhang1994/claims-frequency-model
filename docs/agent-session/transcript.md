# Agent session: Claude Code using the claims MCP server

Recorded on 2026-11-22 with `claude -p` (Claude Code 2.1.308, model claude-opus-5-5), connected only to this
repo's MCP server through `.mcp.json` and allowed only its four tools. Unedited except for formatting; tool
results are trimmed.

## Prompt

An underwriter asks: three new motor policies came in. Which is riskiest, why, and should we trust the
model right now? Use only the claims tools; don't guess numbers.

1. Area E, brand B12, Regular, region R11, 7 hp, 1-year-old car, driver 19, bonus-malus 100, density 9000/km²
2. Area C, brand B1, Diesel, region R24, 6 hp, 8-year-old car, driver 52, bonus-malus 50, density 400/km²
3. Area D, brand B2, Regular, region R82, 9 hp, 3-year-old car, driver 34, bonus-malus 85, density 2500/km²

## Tool calls

1. `score_policy` `{"policy": {"area": "E", "veh_brand": "B12", "veh_gas": "Regular", "region": "R11", "veh_power": 7, "veh_age": 1, "driv_age": 19, "bonus_malus": 100, "density": 9000}}`
   - returned: `{"claims_per_year":{"Poisson GLM":0.2324,"Gradient boosting":0.2401,"Neural net (PyTorch)":0.2073,"GLM + neural correction (CANN)":0.2146},"risk_decile":10,"vs_typical_policy":2.82,"typical_policy_glm_frequency":0.1174}`
2. `score_policy` `{"policy": {"area": "C", "veh_brand": "B1", "veh_gas": "Diesel", "region": "R24", "veh_power": 6, "veh_age": 8, "driv_age": 52, "bonus_malus": 50, "density": 400}}`
   - returned: `{"claims_per_year":{"Poisson GLM":0.0907,"Gradient boosting":0.0941,"Neural net (PyTorch)":0.0853,"GLM + neural correction (CANN)":0.0934},"risk_decile":8,"vs_typical_policy":1.1,"typical_policy_glm_frequency":0.1174}`
3. `score_policy` `{"policy": {"area": "D", "veh_brand": "B2", "veh_gas": "Regular", "region": "R82", "veh_power": 9, "veh_age": 3, "driv_age": 34, "bonus_malus": 85, "density": 2500}}`
   - returned: `{"claims_per_year":{"Poisson GLM":0.1588,"Gradient boosting":0.1844,"Neural net (PyTorch)":0.2883,"GLM + neural correction (CANN)":0.1804},"risk_decile":10,"vs_typical_policy":2.16,"typical_policy_glm_frequency":0.1174}`
4. `explain_policy` `{"policy": {"area": "E", "veh_brand": "B12", "veh_gas": "Regular", "region": "R11", "veh_power": 7, "veh_age": 1, "driv_age": 19, "bonus_malus": 100, "density": 9000}}`
   - returned: `{"typical_frequency":0.1174,"policy_frequency":0.2324,"factors":{"bonus_malus":{"value":100,"typical":50,"multiplier":3.289},"driv_age":{"value":19,"typical":44,"multiplier":0.693},"veh_age":{"value":1,"typical":6,"multiplier":0.838},"regio…`
5. `explain_policy` `{"policy": {"area": "C", "veh_brand": "B1", "veh_gas": "Diesel", "region": "R24", "veh_power": 6, "veh_age": 8, "driv_age": 52, "bonus_malus": 50, "density": 400}}`
   - returned: `{"typical_frequency":0.1174,"policy_frequency":0.0907,"factors":{"veh_brand":{"value":"B1","typical":"B12","multiplier":0.881},"veh_gas":{"value":"Diesel","typical":"Regular","multiplier":0.94},"veh_age":{"value":8,"typical":6,"multiplier":…`
6. `explain_policy` `{"policy": {"area": "D", "veh_brand": "B2", "veh_gas": "Regular", "region": "R82", "veh_power": 9, "veh_age": 3, "driv_age": 34, "bonus_malus": 85, "density": 2500}}`
   - returned: `{"typical_frequency":0.1174,"policy_frequency":0.1588,"factors":{"bonus_malus":{"value":85,"typical":50,"multiplier":2.301},"driv_age":{"value":34,"typical":44,"multiplier":0.663},"veh_age":{"value":3,"typical":6,"multiplier":0.875},"veh_br…`
7. `monitoring_status` `{}`
   - returned: `{"generated_at":"2026-11-20T02:36:26+00:00","overall_status":"alert","summary":[{"batch":"steady","status":"ok","action":"No action needed.","actual_to_expected":0.988,"drifted_features":"none"},{"batch":"mix_shift","status":"watch","action…`
8. `compare_models` `{}`
   - returned: `{"result":[{"model":"Baseline (portfolio mean)","fit_seconds":0,"poisson_deviance":0.6258,"gini":-0.0101,"predicted_to_observed_claims":0.9951},{"model":"Poisson GLM","fit_seconds":0.8,"poisson_deviance":0.5962,"gini":0.2742,"predicted_to_o…`

## Answer

## Policy #1 is the riskiest: the 19-year-old driver in Area E

All numbers below come from the claims tools (`score_policy`, `explain_policy`, `compare_models`, `monitoring_status`).

| | Gradient boosting (best model) | GLM | Neural net | CANN | Risk decile | vs. typical policy (GLM) |
|---|---|---|---|---|---|---|
| **#1** E / B12 / age 19 / BM 100 | **0.240** | 0.232 | 0.207 | 0.215 | **10** | **2.82×** |
| #3 D / B2 / age 34 / BM 85 | 0.184 | 0.159 | 0.288 | 0.180 | 10 | 2.16× |
| #2 C / B1 / age 52 / BM 50 | 0.094 | 0.091 | 0.085 | 0.093 | 8 | 1.10× |

Frequencies are expected claims per policy-year. The "typical" policy's GLM frequency is 0.117.

**Why #1 is riskiest.** Bonus-malus drives almost all of it. A coefficient of 100 (no claims-free record yet) multiplies the typical frequency by **3.29×**. Urban density adds a little (9,000/km² gives 1.11×; Area E gives 1.03×). Region R11 (0.89×) and the 1-year-old car (0.84×) pull it down.

**#3 is close behind, for the same reason.** A bonus-malus of 85 gives 2.30×, offset by driver age 34 (0.66×), brand B2 (0.88×) and a 3-year-old car (0.88×).

**#2 is close to a typical policy.** It has the best possible bonus-malus (50), and every other factor is near 1×, with small discounts for brand B1 (0.88×) and diesel (0.94×). Note that the gradient boosting model still puts it in decile 8, because the "typical" reference policy is itself fairly high on the book's scale. So #2 is average-to-moderate, not low risk.

## Should we trust the model right now? Only partly

**The ranking: yes, mostly.** On held-out data, gradient boosting is the best model (deviance 0.574, Gini 0.330, predicted/observed claims 0.996). Three of the four models agree on the order #1 > #3 > #2.

**The overall level: no, not without caution.** Monitoring status is **ALERT** (report generated 2026-11-20):
- **claims_trend batch:** actual claims are 1.204× expected, with no input drift. Claims are running about 20% above what the model predicts, which points to a real trend the model hasn't learned. The recommended action is to investigate and consider retraining. If this reflects the live book, all three figures above are likely understated by about that much.
- **mix_shift batch:** status is "watch". The driver-age and bonus-malus mix has shifted, but predictions still match claims (1.0 would be a perfect match; this batch is 0.98).
- **steady batch:** status is OK (0.99).

The batch names look like test scenarios rather than real production segments. Someone should confirm which one reflects the current book before relying on the 20% figure.

**Two more caveats:**
1. **The models disagree on #3.** The neural net gives 0.288, about 1.6× the other models. Treat #3's risk as less certain and worth a manual look.
2. **The GLM's driver-age effect looks backwards for #1.** It gives the 19-year-old a 0.69× *discount* compared with a 44-year-old. The likely cause is that young drivers almost always start at bonus-malus 100, so the 3.29× bonus-malus factor is carrying the young-driver risk. The total is plausible, but I wouldn't show the age line by itself to anyone as a justification.

**My recommendation:** rank #1 > #3 > #2 and refer #1 for young-driver pricing. Until the claims_trend alert is resolved, treat the absolute frequencies as possibly understated by about 20%.

_10 turns, $0.29._
