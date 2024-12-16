# Estimation model of subjective hazard rate in dots reversal

This is the reproduction for the results in Christopher M Glaze et al. 2015 eLife

[Normative evidence accumulation in unpredictable environments](https://elifesciences.org/articles/08825)

## Calculate subjective parameters

- Hazard Rate (λ)
λ indicates subjective hazard rate, assigned at each different environment with a certain range of λ.
- k

k indicates the ratio of the evidence to prior expectancy, common in all different λ environments.

```bash
python calc_dr_params_demo.py 0
```

If you use docker environment,

```bash
docker compose exec workspace python calc_dr_params_demo.py 0
```
