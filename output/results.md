# Results

## Data quality (raw table, SQL)

- rows: 678,013
- duplicate_policy_ids: 0
- null_cells: 0
- non_positive_exposure: 0
- exposure_over_one_year: 1,224
- claims_over_four: 9

## Observed frequency by bonus-malus band (SQL)

| bonus_malus_band   |   policies |   exposure_years |   claims |   claim_frequency |
|:-------------------|-----------:|-----------------:|---------:|------------------:|
| 50-59              |     461239 |           264962 |    21808 |            0.0823 |
| 60-79              |     118032 |            53777 |     6920 |            0.1287 |
| 80-99              |      71418 |            29809 |     4265 |            0.1431 |
| 100+               |      27324 |             9811 |     3063 |            0.3122 |

## Held-out scores (508,509 train / 169,504 test policies)

Lower Poisson deviance is better; higher Gini separates risks better; predicted/observed claims near 1.0 means the total is right.

| model                            |   fit_seconds |   poisson_deviance |    gini |   predicted_to_observed_claims |
|:---------------------------------|--------------:|-------------------:|--------:|-------------------------------:|
| Baseline (portfolio mean)        |           0   |             0.6258 | -0.0101 |                         0.9951 |
| Poisson GLM                      |           0.8 |             0.5962 |  0.2742 |                         0.9921 |
| Gradient boosting (Poisson loss) |           2.1 |             0.5743 |  0.3301 |                         0.9956 |
| Neural net (PyTorch, embeddings) |           4.6 |             0.5859 |  0.296  |                         0.9898 |
| GLM + neural correction (CANN)   |           5.9 |             0.5865 |  0.2991 |                         0.9888 |

## Calibration by decile (gradient boosting)

|   decile |   exposure |   observed_freq |   predicted_freq |
|---------:|-----------:|----------------:|-----------------:|
|        1 |    8953.43 |          0.0405 |           0.0403 |
|        2 |    8953.39 |          0.056  |           0.0537 |
|        3 |    8953.87 |          0.062  |           0.0624 |
|        4 |    8953.44 |          0.0667 |           0.07   |
|        5 |    8954.13 |          0.0726 |           0.0767 |
|        6 |    8953.12 |          0.0838 |           0.083  |
|        7 |    8954.67 |          0.0892 |           0.0898 |
|        8 |    8953.69 |          0.0971 |           0.1    |
|        9 |    8952.81 |          0.1335 |           0.1279 |
|       10 |    8954.67 |          0.3086 |           0.3017 |
