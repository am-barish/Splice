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
git clone <repo-url> && cd splice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All commands are run from the repository root.

Python 3.10+. Runtimes in the paper were measured on an Apple MacBook M3 Pro
with 36 GB RAM.

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
  warm_caches.py        pre-builds surrogate and gradient caches
  regen_meta_history.py rebuilds Splice-DP artifacts from the val split
harness/
  stabilize.py          oracle training-config patches and probes
  repeated_splits.py    repeated-split driver (to run experiments)
```

---

## Data

Expected under a directory passed as `--data-dir` (default `.`), with derived
artifacts written to `--cache-dir`:

Before any multi-seed run, build the shared caches once. Several artifacts are
written lazily on first use and shared across processes — the ACS parquet, the
target splits, `<DS>_surrogate.joblib` (Splice-DP) and `<DS>_gradients.joblib`
(Splice-Grad):

```bash
python -m data_prep.warm_caches --methods normal dp gradmatch
python -m experiments.create_acs_splits
```

`create_acs_splits` writes 30 splits per ACS dataset (seeds 12345 and 1000-1028)
to `target_splits/`, matching the replicates behind main experiment table. For Wilds and
Amazon, the Splice-DP artifacts are rebuilt from the validation split with:

```bash
python -m data_prep.regen_meta_history --family wilds  --iters 5
python -m data_prep.regen_meta_history --family amazon --iters 5
```

- **ACS** — Folktables downloads on first use. Build contexts with
  `python -m data_prep.acs`, then create canonical splits:
  ```bash
  python -m experiments.create_acs_splits
  ```
- **Amazon Reviews** — `reviews_raw.parquet` plus LSA features
  `reviews_features_lsa150.parquet`. Re-split with:
  ```bash
  python -m experiments.resplit_amazon --val-frac 0.30 --seed 42 --stratify
  ```
- **Wilds** — camera-trap images; features are extracted once with a frozen
  ImageNet ResNet-18 (512-dim) and cached.
- **SANTOS (IPO / YDNPA)** — monthly spending CSVs in a directory passed via
  `--data-dir`, one table held out as the target via `--target-file`.
  
- **SynPubCov** — 1,000 synthetic sources of 5,000 examples each, drawn from
  ACSPublicCoverage by stratified resampling within label strata with a
  per-source positive rate spread uniformly around the pool base rate
  (`repartition_stratified_skew` in `data_prep/acs.py`).

ACS uses a 30/70 validation/test split and base seed `12345`; Wilds and Amazon
use base seed `42`.

---

## Example: Running multiseed experiment for Splice:

```bash
python -m harness.repeated_splits \
    --dataset ACSPublicCoverage --family acs \
    --method normal --algos splice \
    --n-splits 30 --smax 7 --kmax 7 --split-seed-base 1000
```

Repeat with `--method dp` and `--method gradmatch` for the surrogate rows, and
with `--family wilds` / `--family amazon` for those benchmarks.

---

## Key parameters

| Flag | Meaning | Paper setting |
|---|---|---|
| `--smax` | budget *B*, max active-set size | ACS 7, Amazon 10, Wilds 15 |
| `--kmax` | max swap size *k*<sub>max</sub> | `= B` in Paper |
| `--method` | marginal-gain oracle | `normal`/`dp`/`gradmatch` |
| `--val-frac` | fraction of target guiding selection | 0.30 |
| `--n-jobs` | worker processes for the parallel path | 8 for reported speedups |

The profit function is `accuracy*100 + lambda * tpr_parity`, where `tpr_parity`
is the **signed** difference `TPR(protected) - TPR(privileged)`
(`core/metrics.py`). `lambda = 50` on ACSIncome and `0` elsewhere.

---
