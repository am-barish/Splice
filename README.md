# SPLICE — Scalable Source Selection for Machine Learning Tasks

Reference implementation for the paper *Scalable Source Selection for Machine
Learning Tasks*.

| Variant | Marginal-gain oracle | `--method` |
|---|---|---|
| SPLICE | exact retraining | `normal` |
| SPLICE-DP | data profiles + meta-learned regressor | `dp` |
| SPLICE-Grad | source/target gradient alignment | `gradmatch` |

---

## Install

```bash
git clone <repo-url> && cd Splice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. All commands are run from the repository root. Runtimes in the
paper were measured on an Apple MacBook M3 Pro with 36 GB RAM.

---

## Layout

```
core/
  context.py            ExperimentContext: sources, splits, caches, surrogate state
  splice.py             search (splicing, fixed_support, get_best_subset)
  metrics.py            profit function, fairness terms, marginal-gain oracles
  experiment.py         run_experiment driver shared by all algorithms
  splice_ext.py         split-aware patches applied at import
surrogates/
  dataprofiles.py       SPLICE-DP profiles, tabular
  dataprofiles_unstructured.py   SPLICE-DP profiles, image and text
  gradientmatch.py      SPLICE-Grad gradient alignment
baselines/
  classical.py          Single, AllSrc, Random, Greedy
  grasp.py              GRASP
  dsdm.py               DsDM
  coresets.py           point-level coresets (random, k-Center, GradMatch)
data_prep/
  acs.py                ACS Folktables context builder
  amazon.py             Amazon Reviews sources
  wilds.py              Wilds camera-trap sources
  wildcam.py            camera-trap Dataset wrapper
  loaders.py            shared dataset loading and source construction
  splits.py             canonical split seeds and validation fractions
  valsize_splits.py     nested stratified validation draws
  warm_caches.py        pre-builds Wilds / Amazon feature caches
  regen_meta_history.py rebuilds Splice-DP artifacts from the val split
experiments/
  repeated_splits.py    repeated-split driver
  stabilize.py          oracle training-config patches and probes
acs_cache/              ACS parquet, surrogate and gradient artifacts (shipped)
target_splits/          the 30 ACS target splits (shipped)
```

---

## Data

- **ACS Folktables** — three datasets (ACSIncome, ACSPublicCoverage,
  ACSTravelTime), 15 U.S. states as candidate sources and Puerto Rico as the
  held-out target, 30/70 validation/test split, base seed `12345`. Contexts are
  built by `data_prep/acs.py`.
- **Wilds** — camera-trap images from 30 locations; features extracted once
  with a frozen ImageNet ResNet-18 (512-dim) and cached. Splice-DP artifacts:
  ```bash
  python -m data_prep.regen_meta_history --family wilds --iters 5
  ```
- **Amazon Reviews** — 24 categories as sources, 3 as target;
  `reviews_raw.parquet` plus LSA features `reviews_features_lsa150.parquet`.
  Splice-DP artifacts:
  ```bash
  python -m data_prep.regen_meta_history --family amazon --iters 5
  ```
  Both write `history_list_<family>_val.pkl` and
  `<Family>_target_profile_val.pkl` to the repository root.
- **SynPubCov** — 1,000 synthetic sources of 5,000 examples each, drawn from
  ACSPublicCoverage by stratified resampling within label strata with a
  per-source positive rate spread around the pool base rate
  (`repartition_stratified_skew` in `data_prep/acs.py`).

Wilds and Amazon use base seed `42`.

---

## Running the repeated-split experiment

```bash
python -m experiments.repeated_splits \
    --dataset ACSPublicCoverage --family acs \
    --method normal --algos splice \
    --n-splits 30 --smax 7 --split-seed-base 1000 \
    --cache-dir acs_cache
```

Repeat with `--method dp` and `--method gradmatch` for the surrogate rows, and
`--family wilds` / `--family amazon` for those benchmarks. `--algos` also takes
`grasp`, `dsdm`, `greedy`, `random`, `all_sources`, `single_source`.

Results are written to `--out` (default `results/repeated_splits`) as a
`_runs.csv` with one row per split and a `_summary.csv` with the aggregates.

---

## Key parameters

| Flag | Meaning | Paper setting |
|---|---|---|
| `--smax` | budget *B*, max active-set size | ACS 7, Amazon 10, Wilds 15 |
| `--method` | marginal-gain oracle | `normal` / `dp` / `gradmatch` |
| `--n-splits` | target splits (ACS) or model seeds (Wilds, Amazon) | 30 / 10 |
| `--val-frac` | fraction of target guiding selection | 0.30 |
| `--eval-seed` | model seed | 42 |

The profit function is `accuracy*100 + lambda * tpr_parity`, where `tpr_parity`
is the **signed** difference `TPR(protected) - TPR(privileged)`
(`core/metrics.py`). `lambda = 50` on ACSIncome and `0` elsewhere. Because the
term is signed, a classifier that reverses the disparity can score above 100.

---

