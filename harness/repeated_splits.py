from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TARGET_STATE = "PR"


POOLS = {
    "ACSIncome":         [2, 9, 10, 11, 12, 15, 19, 23, 25, 29, 33, 35, 39, 40, 49],
    "ACSPublicCoverage": [4, 9, 8, 21, 22, 31, 34, 43, 42, 18, 11, 48, 32, 26, 19],
    "ACSTravelTime":     [3, 6, 2, 21, 26, 31, 34, 43, 42, 17, 12, 49, 33, 25, 10],
    "Wilds":             list(range(30)),
    "Amazon":            list(range(24)),


}


STABILIZE_DEFAULTS = {
    "amazon": dict(epochs=15, lr=5e-3, n_avg=1),
    "wilds":  dict(vary="both"),
}


RESAMPLES_TARGET = {"acs"}


_CACHE = {}


def get_ctx(dataset, cache_dir, data_dir, val_frac, split_mode, method,
            split_seed=None, family="acs", stabilize_cfg=None,
            n_sources=None, pool_seed=None, n_base=None):
    key = (family, dataset, cache_dir, method, split_seed,
           n_sources, pool_seed, n_base)
    if key not in _CACHE:
        from core import splice_ext
        splice_ext.install()

        cfg = STABILIZE_DEFAULTS if stabilize_cfg is None else stabilize_cfg
        if family == "amazon" and cfg.get("amazon") is not None:
            from harness import stabilize
            stabilize.patch_amazon(**cfg["amazon"])
        elif family == "wilds" and cfg.get("wilds") is not None:
            from harness import stabilize
            stabilize.patch_wilds(**cfg["wilds"])

        extra = {}
        if n_sources is not None:
            extra.update(n_sources=n_sources, pool_seed=pool_seed,
                         n_base=n_base)
        ctx = splice_ext.make_ctx(family=family, dataset=dataset,
                                  cache_dir=cache_dir, data_dir=data_dir,
                                  val_frac=val_frac, split_mode=split_mode,
                                  split_seed=split_seed, method=method,
                                  **extra)
        if family == "acs":
            splice_ext.assert_split_ready(ctx)
        _CACHE[key] = ctx
    return _CACHE[key]


def _sens_vector(masks, n):
    if not masks:
        return None
    v = np.zeros(n, dtype=np.float32)
    for g, m in masks.items():
        v[np.asarray(m)] = g
    return v


def target_full(ctx, split_dir="target_splits"):
    d = ctx.dataset
    path = Path(split_dir) / f"{d}_target_{TARGET_STATE}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the context once with the default "
            f"split_seed first (it writes this file), or pass --rebuild.")
    blob = json.loads(path.read_text())
    vi = np.asarray(blob["val_idx"], dtype=np.int64)
    ti = np.asarray(blob["test_idx"], dtype=np.int64)
    n = int(blob["n"])

    Xv, Xt = ctx.val_X_scaled[d], ctx.test_X_scaled[d]
    yv, yt = ctx.val_y[d], ctx.test_y[d]
    if len(vi) != len(yv) or len(ti) != len(yt) or len(vi) + len(ti) != n:
        raise RuntimeError(
            f"{path} does not match the loaded context "
            f"(json {len(vi)}/{len(ti)} vs ctx {len(yv)}/{len(yt)}). The "
            f"cached split is stale -- delete it and rebuild.")

    X = np.empty((n, Xv.shape[1]), dtype=Xv.dtype)
    X[vi], X[ti] = Xv, Xt
    y = np.empty(n, dtype=yv.dtype)
    y[vi], y[ti] = yv, yt

    sv = _sens_vector(ctx.val_sens_masks.get(d, {}), len(yv))
    st = _sens_vector(ctx.test_sens_masks.get(d, {}), len(yt))
    if sv is None or st is None:
        sens = None
    else:
        sens = np.empty(n, dtype=np.float32)
        sens[vi], sens[ti] = sv, st
    return X, y, sens


def replicate_split(dataset, y, sens, seed, val_frac):
    from data_prep.splits import get_split


    sens_for_split = None if dataset == "ACSPublicCoverage" else sens
    return get_split(f"{dataset}_target_{TARGET_STATE}", y,
                     sens=sens_for_split,
                     is_regression=(dataset == "ACSTravelTime"),
                     val_frac=val_frac, seed=int(seed), verbose=False)


def apply_split(ctx, X, y, sens, val_idx, test_idx, eval_seed=42):
    from core import splice_ext
    d = ctx.dataset

    def masks(idx):
        if sens is None:
            return {}
        v = sens[idx]
        return {int(g): (v == g) for g in np.unique(sens)}

    ctx.val_X_scaled[d], ctx.val_y[d] = X[val_idx], y[val_idx]
    ctx.val_sens_masks[d] = masks(val_idx)
    ctx.test_X_scaled[d], ctx.test_y[d] = X[test_idx], y[test_idx]
    ctx.test_sens_masks[d] = masks(test_idx)
    ctx.val_idx = None
    splice_ext.set_eval_config(ctx, seed=eval_seed, split="val")
    return len(val_idx), len(test_idx)


def run_one(rep, split_seed, algo, pool, smax, kmax, eval_seed, cfg, val_frac,
            rebuild=False, parallel=False, inner_jobs=-1, profits_out=""):
    import time
    from core import splice_ext
    from core.experiment import run_experiment

    family = cfg.get("family", "acs")


    if family != "acs" or cfg.get("dataset") == "Scaled_Pubcov":

        ctx = get_ctx(**cfg)
        eval_seed = int(split_seed)
        splice_ext.set_eval_config(ctx, seed=eval_seed, split="val")
        d = ctx.dataset
        n_val = len(ctx.val_y[d]) if d in getattr(ctx, "val_y", {}) else -1
        n_test = len(ctx.test_y[d]) if d in getattr(ctx, "test_y", {}) else -1
    elif rebuild:
        ctx = get_ctx(**cfg, split_seed=split_seed)
        n_val = len(ctx.val_y[ctx.dataset])
        n_test = len(ctx.test_y[ctx.dataset])
        splice_ext.set_eval_config(ctx, seed=eval_seed, split="val")
    else:
        ctx = get_ctx(**cfg)
        X, y, sens = _full_arrays(ctx)
        vi, ti = replicate_split(ctx.dataset, y, sens, split_seed, val_frac)
        n_val, n_test = apply_split(ctx, X, y, sens, vi, ti, eval_seed)

    ctx.source_list = list(pool)
    ctx.random_subsets = None
    ctx.k_max = kmax

    t0 = time.time()
    res = run_experiment(ctx, algo, smax=smax, seed=eval_seed,
                     parallel=parallel, n_jobs=inner_jobs)
    if profits_out:
        import json as _json, pathlib
        pathlib.Path(profits_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(profits_out).write_text(_json.dumps(res.get("profits", [])))

    elapsed = time.time() - t0
    subset = sorted(res.get("subset") or [])


    def _test_profit(sub):
        if not len(sub):
            return float("nan")
        if family == "acs":
            return float(splice_ext.final_eval(ctx, sub, seed=eval_seed)["profit"])
        from core.metrics import get_profit
        with splice_ext.using_split(ctx, "test"):
            return float(get_profit(ctx, list(sub)))


    draws = getattr(ctx, "random_subsets", None)
    if (algo == "random" or not subset) and draws:
        test = float(np.mean([_test_profit(s) for s in draws]))
    elif subset:
        test = _test_profit(subset)
    else:
        test = float("nan")


    explored = res.get("models_explored")
    if explored is None:
        explored = {"all_sources": 1, "single_source": len(pool),
                    "random": len(draws) if draws else 0}.get(algo)

    return dict(family=family, dataset=ctx.dataset, method=cfg.get("method"),
                rep=rep, split_seed=split_seed, algo=algo,
                subset=json.dumps(subset), n_selected=len(subset),
                val_profit=float(res["profit"]), test_profit=test,
                overfit_gap=float(res["profit"]) - test,
                n_val=n_val, n_test=n_test,
                models_explored=explored,
                time_elapsed=res.get("time_elapsed", elapsed))


_FULL = {}


def _full_arrays(ctx):
    if ctx.dataset not in _FULL:
        _FULL[ctx.dataset] = target_full(ctx)
    return _FULL[ctx.dataset]


def nadeau_bengio(d, n_val, n_test, alpha=0.05):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    k = len(d)
    out = dict(k=k, mean=float(d.mean()) if k else np.nan,
               sd=float(d.std(ddof=1)) if k > 1 else np.nan)
    if k < 2:
        return {**out, "factor": np.nan, "t": np.nan, "p": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "naive_t": np.nan,
                "naive_p": np.nan}

    s2 = d.var(ddof=1)
    factor = 1.0 / k + n_test / n_val
    se = np.sqrt(factor * s2)
    tcrit = stats.t.ppf(1 - alpha / 2, k - 1)

    if se == 0:
        return {**out, "factor": float(factor), "t": np.nan, "p": np.nan,
                "ci_lo": out["mean"], "ci_hi": out["mean"],
                "naive_t": np.nan, "naive_p": np.nan}

    t = out["mean"] / se
    naive_se = np.sqrt(s2 / k)
    naive_t = out["mean"] / naive_se if naive_se > 0 else np.nan
    return {**out, "factor": float(factor), "t": float(t),
            "p": float(2 * stats.t.sf(abs(t), k - 1)),
            "ci_lo": float(out["mean"] - tcrit * se),
            "ci_hi": float(out["mean"] + tcrit * se),
            "naive_t": float(naive_t),
            "naive_p": float(2 * stats.t.sf(abs(naive_t), k - 1))}


def subset_stability(subsets):
    sets = [frozenset(json.loads(s)) for s in subsets]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return dict(n=len(sets), exact_match=np.nan, mean_jaccard=np.nan,
                    n_distinct=len(set(sets)))
    pairs = list(combinations(sets, 2))
    exact = np.mean([a == b for a, b in pairs])
    jac = np.mean([len(a & b) / len(a | b) for a, b in pairs])
    return dict(n=len(sets), exact_match=float(exact),
                mean_jaccard=float(jac), n_distinct=len(set(sets)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ACSPublicCoverage")
    ap.add_argument("--family", default=None,
                    choices=["acs", "wilds", "amazon"],
                    )
    ap.add_argument("--cache-dir", default="acs_cache")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--split-mode", default="stratified",
                    choices=["stratified", "legacy"])
    ap.add_argument("--method", default="normal",
                    choices=["normal", "dp", "gradmatch", "hybrid"])
    ap.add_argument("--pool", type=int, nargs="+", default=None,
                    )
    ap.add_argument("--n-sources", type=int, default=1000,
                    )
    ap.add_argument("--pool-seed", type=int, default=42,
                    )
    ap.add_argument("--n-base", type=int, default=1000,
                    )
    ap.add_argument("--spread-check", action="store_true",
                    )
    ap.add_argument("--algos", nargs="+",
                    default=["splice", "greedy", "grasp"])
    ap.add_argument("--reference", default="splice")
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--split-seed-base", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--smax", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=15)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--rebuild", action="store_true",
                    )
    ap.add_argument("--parallel", action="store_true",
                    )
    ap.add_argument("--profits-out", default="")
    ap.add_argument("--inner-jobs", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/repeated_splits")
    a = ap.parse_args()

    from data_prep.splits import VAL_FRAC, SPLIT_SEED
    val_frac = a.val_frac if a.val_frac is not None else VAL_FRAC
    family = a.family or ("acs" if a.dataset.startswith("ACS")
                          or a.dataset == "Scaled_Pubcov" else
                          {"Wilds": "wilds", "Amazon": "amazon"}.get(a.dataset))
    if family is None:
        raise SystemExit(f"cannot infer --family for dataset {a.dataset!r}")
    if a.pool is None:
        if a.dataset == "Scaled_Pubcov":
            a.pool = list(range(a.n_sources))
        elif a.dataset not in POOLS:
            raise SystemExit(f"no default pool for {a.dataset!r}; pass --pool")
        else:
            a.pool = POOLS[a.dataset]
    cfg = dict(family=family, dataset=a.dataset, cache_dir=a.cache_dir,
               data_dir=a.data_dir, val_frac=val_frac,
               split_mode=a.split_mode, method=a.method)
    if a.dataset == "Scaled_Pubcov":
        cfg.update(n_sources=a.n_sources, pool_seed=a.pool_seed,
                   n_base=a.n_base)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)


    resamples = family == "acs" and a.dataset != "Scaled_Pubcov"
    if resamples:
        seeds = [SPLIT_SEED] + [a.split_seed_base + i
                                for i in range(a.n_splits - 1)]
    else:
        seeds = [a.eval_seed + i for i in range(a.n_splits)]

    ctx = get_ctx(**cfg)
    d = ctx.dataset

    if a.spread_check:
        from data_prep.acs import spread_check as _sc
        _sc(ctx)
        return

    if resamples and not a.rebuild:
        X, y, sens = _full_arrays(ctx)
        vi0, ti0 = replicate_split(d, y, sens, SPLIT_SEED, val_frac)
        same = (len(vi0) == len(ctx.val_y[d]) and
                np.array_equal(y[vi0], np.asarray(ctx.val_y[d])))
        if not same:
            raise SystemExit(
                "Reconstruction check FAILED -- do not trust the replicates. "
                "Rerun with --rebuild.")
        sizes = [len(replicate_split(d, y, sens, s, val_frac)[0]) for s in seeds]

    n_val = len(ctx.val_y[d]) if d in getattr(ctx, "val_y", {}) else -1
    n_test = len(ctx.test_y[d]) if d in getattr(ctx, "test_y", {}) else -1
    corrected = resamples and n_val > 0
    if corrected:
        factor = 1 / len(seeds) + n_test / n_val
    else:
        pass

    (out.with_name(out.name + "_config.json")).write_text(json.dumps(dict(
        family=family, dataset=a.dataset, method=a.method,
        pool=a.pool, algos=a.algos, split_seeds=seeds, kmax=a.kmax,
        parallel=a.parallel, stabilize=STABILIZE_DEFAULTS,
        val_frac=val_frac, split_mode=a.split_mode, smax=a.smax,
        eval_seed=a.eval_seed, n_val=n_val, n_test=n_test), indent=2))

    if a.dry_run:
        return

    tasks = [(r, s, alg) for r, s in enumerate(seeds) for alg in a.algos]

    if a.n_jobs == 1:
        from tqdm import tqdm
        rows = [run_one(r, s, alg, a.pool, a.smax, a.kmax, a.eval_seed, cfg,
                        val_frac, a.rebuild, a.parallel, a.inner_jobs, a.profits_out)
                for r, s, alg in tqdm(tasks, desc="repeated splits")]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=a.n_jobs, prefer="processes", verbose=10)(
            delayed(run_one)(r, s, alg, a.pool, a.smax, a.kmax, a.eval_seed, cfg,
                             val_frac, a.rebuild, a.parallel, a.inner_jobs, a.profits_out)
            for r, s, alg in tasks)

    runs = pd.DataFrame(rows)
    rc = out.with_name(out.name + "_runs.csv")
    runs.to_csv(rc, index=False)

    nv = float(runs.n_val.mean())
    nt = float(runs.n_test.mean())

    for alg, g in runs.groupby("algo"):
        st = subset_stability(g.subset.tolist())

    piv = runs.pivot_table(index="rep", columns="algo", values="test_profit")
    ref = a.reference
    if ref not in piv.columns:
        raise SystemExit(f"--reference {ref!r} not among {list(piv.columns)}")

    summary = []
    for alg in [c for c in piv.columns if c != ref]:
        d_vec = (piv[ref] - piv[alg]).to_numpy()


        r = nadeau_bengio(d_vec, nv, nt if corrected else 0.0)
        w = int((piv[ref] > piv[alg]).sum())
        l = int((piv[ref] < piv[alg]).sum())
        summary.append(dict(baseline=alg, wins=w, losses=l,
                            ties=len(piv) - w - l, **r))
        if not np.isnan(r["ci_lo"]) and r["ci_lo"] <= 0 <= r["ci_hi"]:
            pass

    pd.DataFrame(summary).to_csv(
        out.with_name(out.name + "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
