# Teacher Demand Forecasting in Portugal, 2025-2040

A six-stage econometric pipeline projecting the recruitment gap between
teacher demand and teacher supply in Portugal's public school system through
2040, built on a NUTS III panel assembled from DGEEC and INE public data.

TODO: one or two sentences stating the headline result (national gap /
recruitment need, central scenario) once you've confirmed the number you
want to lead with.

## Pipeline

The project is organized as a sequential pipeline, each stage reading the
previous stage's output:

1. **Data build** (`data/04_code/`, 11 scripts) — assembles a NUTS III panel
   from raw DGEEC (teacher counts by school, graduate counts) and INE
   (population estimates and projections, ageing index, renewal index,
   student enrollment, teacher counts) sources into a master panel.
2. **Students** (`models/01_students/`) — forecasts student enrollment by
   NUTS III and education cycle.
3. **Demand** (`models/02_demand/`) — estimates demand elasticities via a
   two-way fixed effects panel regression with wild cluster bootstrap
   inference, backtests several candidate forecasting specifications
   (including hybrid and growth-ratio variants) against each other using
   rolling-origin cross-validation, selects a winning specification per
   cell subject to statistical gating criteria, and benchmarks the
   resulting national aggregate against the DGEEC + Nova SBE (2025) study.
4. **Supply** (`models/03_supply/`) — models the existing teacher stock
   forward using cohort-based exit hazards by age band, entry channels
   into the profession, and regional raking to NUTS II/III.
5. **Gap** (`models/04_gap/`) — reconciles demand and supply into a
   recruitment gap, with NUTS III-level reconciliation and a decomposition
   by entry channel.
6. **Uncertainty** (`models/05_uncertainty/`) — quantifies uncertainty via
   Monte Carlo simulation (4,000 draws) with a correlation parameter
   between demand-side and supply-side shocks, producing deficit
   probability and fan charts plus a variance decomposition.
7. **Validation** (`models/06_validation/`) — validates the pipeline via
   hindcasting (MASE by age band) and internal consistency checks
   (identity, cross-block, spatial, and reconciliation tolerances).

Diagram-generation code for the pipeline structure is in `report/codigo/`.

## Data

All source data are public:
- DGEEC: teacher counts by school (annual files, 2011/12-2024/25) and
  graduate counts (1996/97-2024/25)
- INE: population estimates, population projections, ageing index,
  renewal index, student enrollment, and teacher counts

Raw, processed, and analysis-ready versions are included under `data/`.

## Repository structure

```
data/
  01_raw/         raw DGEEC and INE files, as downloaded
  02_processed/   cleaned, NUTS III-harmonized tables
  03_analysis/    analysis-ready panels
  04_code/        11 build scripts producing the master panel
models/
  00_data/        pipeline input snapshot
  01_students/    student enrollment forecast
  02_demand/      teacher demand model (elasticities, bake-off, selection)
  03_supply/      teacher supply model (hazards, entries, Monte Carlo)
  04_gap/         demand-supply reconciliation and gap
  05_uncertainty/ Monte Carlo uncertainty quantification
  06_validation/  hindcasting and consistency checks
report/
  codigo/         pipeline diagram generation scripts
```

## Reproducing the results

Python (numpy, pandas, matplotlib). Each stage under `models/*/code/` reads
its inputs from the preceding stage and writes results and figures to its
own `results/` and `images/` folders. Run in order:

```
python data/04_code/01_build_students.py
...
python data/04_code/11_final_join.py
python models/01_students/code/forecast_students.py
python models/02_demand/code/demand_pipeline.py
python models/03_supply/code/supply_pipeline.py
python models/04_gap/code/gap_model_and_map.py
python models/05_uncertainty/code/uncertainty_model.py
python models/06_validation/code/validation.py
```

TODO: confirm this is the exact intended run order and add a
`requirements.txt` (none is currently in the repo).

## Limitations

TODO: state the limitations you've actually identified, e.g. around the
demand elasticity gating, the flat vs. estimated exit hazard assumption
(`h60_mode`), or the 2020 exclusion from backtesting.

## Notes on development

Developed with the assistance of AI coding tools; methodology, model
choices, and validation are my own.

## Related

- Blog post: [The Causal Layer](https://joaovieira22.github.io/)

## License

MIT
