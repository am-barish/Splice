from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


from data_prep.splits import VAL_FRAC

YEARS = ["2014", "2015", "2016", "2017", "2018"]

STATE_LIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

TARGET_COL = {"ACSIncome": "PINCP", "ACSPublicCoverage": "PUBCOV",
              "ACSTravelTime": "JWMNP", "Scaled_Pubcov": "PUBCOV"}
SENS_COL = {"ACSIncome": "SEX", "ACSPublicCoverage": "RAC1P",
            "ACSTravelTime": None, "Scaled_Pubcov": "RAC1P"}
GAIN_LAMBDA = {"ACSIncome": 50.0, "ACSPublicCoverage": 0.0,
               "ACSTravelTime": 0.0, "Scaled_Pubcov": 0.0}


SYNTHETIC_BASE = {"Scaled_Pubcov": "ACSPublicCoverage"}


def source_id_map():
    return {i: s for i, s in enumerate(STATE_LIST[:-1])}


def _check_acs_dataset(dataset):
    if dataset not in TARGET_COL:
        raise ValueError(
            f"{dataset!r} is not an ACS dataset. This builder handles "
            f"{sorted(TARGET_COL)}. For Wilds/Amazon use --family wilds "
            f"/ --family amazon, which go through setup_wilds_context / "
            f"setup_amazon_context instead.")


def download_acs(dataset, years=YEARS, states=None, root="acs_raw",
                 download=True):
    _check_acs_dataset(dataset)
    from data_prep.loaders import acs_dict, race_encode
    from folktables import ACSDataSource
    from tqdm import tqdm

    states = states or STATE_LIST
    frames = []

    for year in tqdm(years, desc=f"{dataset}: years"):
        src = ACSDataSource(survey_year=year, horizon="1-Year",
                            survey="person", root_dir=root)
        for state in tqdm(states, desc=f"  {year}", leave=False):
            raw = src.get_data(states=[state], download=download)
            if dataset == "ACSTravelTime":
                part = _travel_time_frame(raw)
            else:
                feats, labels, _ = acs_dict[dataset].df_to_pandas(raw)
                part = pd.concat([feats.reset_index(drop=True),
                                  labels.reset_index(drop=True)], axis=1)
            part.insert(0, "State", state)
            part.insert(0, "Year", year)
            frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    if dataset == "ACSPublicCoverage":
        df["RAC1P"] = df["RAC1P"].apply(race_encode)
    if dataset == "ACSTravelTime":
        df = df.dropna()
    return df.reset_index(drop=True)


def _travel_time_frame(raw):
    df = raw.copy()
    df = df[(df["AGEP"] > 16) & (df["PWGTP"] >= 1) & (df["ESR"] == 1)]
    cols = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'MIG', 'RELP',
            'RAC1P', 'PUMA', 'ST', 'CIT', 'OCCP', 'JWTR', 'POWPUMA', 'POVPIP']
    X = pd.DataFrame(np.nan_to_num(df[cols].to_numpy(), -1), columns=cols)
    y = df["JWMNP"].reset_index(drop=True)
    return pd.concat([X, y], axis=1)


def split_ood(df, dataset, target_state="PR", val_frac=VAL_FRAC,
              split_mode="stratified", split_seed=None):
    from data_prep.splits import get_split

    target_col = TARGET_COL[dataset]
    sens_col = SENS_COL[dataset]

    sources = []
    for sid, state in enumerate(STATE_LIST[:-1]):
        part = df[df["State"] == state].reset_index(drop=True)
        part["Source ID"] = sid
        sources.append(part)

    tgt = df[df["State"] == target_state].reset_index(drop=True)
    if tgt.empty:
        raise ValueError(f"target state {target_state} has no rows")

    if split_mode == "legacy":
        from sklearn.model_selection import train_test_split

        test_df, val_df = train_test_split(
            tgt, test_size=val_frac,
            random_state=42 if split_seed is None else split_seed)
        return sources, val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    y = tgt[target_col].to_numpy()
    sens = tgt[sens_col].to_numpy() if sens_col else None
    kw = {} if split_seed is None else {"seed": int(split_seed)}
    sens_for_split = None if dataset == "ACSPublicCoverage" else sens
    val_idx, test_idx = get_split(
        f"{dataset}_target_{target_state}", y, sens=sens_for_split,
        is_regression=(dataset == "ACSTravelTime"), val_frac=val_frac,
        verbose=(split_seed is None), **kw)


    return (sources,
            tgt.iloc[val_idx].reset_index(drop=True),
            tgt.iloc[test_idx].reset_index(drop=True))


def fit_scaler(sources, dataset):
    target_col = TARGET_COL[dataset]
    drop = ["State", "Year", target_col, "Source ID"]
    X = pd.concat([s.drop(columns=[c for c in drop if c in s.columns])
                   for s in sources], ignore_index=True)
    return StandardScaler().fit(X.values), list(X.columns)


def _xy(frame, dataset, scaler, feat_cols):
    target_col = TARGET_COL[dataset]
    X = scaler.transform(frame[feat_cols].values).astype(np.float32)
    y = frame[target_col].to_numpy()
    y = y.astype(np.float32) if dataset == "ACSTravelTime" else y.astype(np.int8)
    return X, y


def _masks(frame, dataset):
    col = SENS_COL[dataset]
    if not col or col not in frame.columns:
        return {}
    v = frame[col].to_numpy()
    return {int(g): (v == g) for g in np.unique(v)}


def build_acs_context(dataset, *, years=YEARS, states=None, raw_root="acs_raw",
                      cache_dir=None, refresh=False, source_list=None,
                      gain_lambda=None, cost_type="zero", limit=100_000,
                      k_max=15, val_frac=VAL_FRAC, split_mode="stratified",
                      split_seed=None, download=True, method="normal",
                      with_surrogate=False, with_gradients=False,
                      keep_comb=False, surrogate_jobs=3, gradient_jobs=-1):
    from core.context import ExperimentContext

    _check_acs_dataset(dataset)
    df = _load_or_download(dataset, years, states, raw_root, cache_dir,
                           refresh, download)
    sources, val_df, test_df = split_ood(df, dataset, val_frac=val_frac,
                                         split_mode=split_mode,
                                         split_seed=split_seed)
    scaler, feat_cols = fit_scaler(sources, dataset)

    comb = pd.concat(sources, ignore_index=True)


    if method in ("dp", "hybrid"):
        with_surrogate = True
    if method in ("gradmatch", "hybrid"):
        with_gradients = True

    ctx = ExperimentContext(
        dataset=dataset,
        method=method,
        gain_lambda=(GAIN_LAMBDA[dataset] if gain_lambda is None else gain_lambda),
        cost_type=cost_type, limit=limit, k_max=k_max,
        source_list=list(source_list) if source_list else
                    list(range(len(sources))),
    )


    ctx.comb[dataset] = comb if keep_comb else None
    ctx.scaler[dataset] = scaler
    ctx.test_source[dataset] = test_df

    ctx.source_X[dataset], ctx.source_y[dataset] = {}, {}
    ctx.source_sens[dataset] = {}
    for sid, frame in enumerate(sources):
        X, y = _xy(frame, dataset, scaler, feat_cols)
        ctx.source_X[dataset][sid] = X
        ctx.source_y[dataset][sid] = y
        if SENS_COL[dataset]:
            ctx.source_sens[dataset][sid] =                frame[SENS_COL[dataset]].to_numpy().astype(np.float32)

    Xv, yv = _xy(val_df, dataset, scaler, feat_cols)
    Xt, yt = _xy(test_df, dataset, scaler, feat_cols)
    ctx.val_X_scaled[dataset], ctx.val_y[dataset] = Xv, yv
    ctx.val_sens_masks[dataset] = _masks(val_df, dataset)
    ctx.test_X_scaled[dataset], ctx.test_y[dataset] = Xt, yt
    ctx.test_sens_masks[dataset] = _masks(test_df, dataset)
    ctx.eval_split = "val"


    if with_surrogate:
        _build_surrogate(ctx, sources, val_df, dataset, cache_dir, refresh,
                         n_jobs=surrogate_jobs)
    if with_gradients:
        _build_gradients(ctx, sources, val_df, test_df, dataset, feat_cols,
                         cache_dir, refresh, n_jobs=gradient_jobs)

    _assert_method_ready(ctx, dataset, method)

    return ctx


def _load_or_download(dataset, years, states, raw_root, cache_dir, refresh,
                      download):
    if cache_dir:
        cache = Path(cache_dir) / f"{dataset}.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)
    df = download_acs(dataset, years, states, raw_root, download)
    if cache_dir:
        cache = Path(cache_dir) / f"{dataset}.parquet"
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def _assert_method_ready(ctx, dataset, method):
    need = []
    if method in ("dp", "hybrid") and dataset not in getattr(ctx, "surr_model", {}):
        need.append("with_surrogate=True  (CLI: --with-surrogate)")
    if method in ("gradmatch", "hybrid") and dataset not in getattr(ctx, "g_val", {}):
        need.append("with_gradients=True  (CLI: --with-gradients)")
    if need:
        raise RuntimeError(
            f"method={method!r} needs artifacts that were not built for "
            f"{dataset}: " + "; ".join(need) +
            ". They are cached after the first build, so this is a one-time cost.")


def _artifact(cache_dir, dataset, kind):
    return None if not cache_dir else Path(cache_dir) / f"{dataset}_{kind}.joblib"


def _build_surrogate(ctx, sources, val_df, dataset, cache_dir=None,
                     refresh=False, n_jobs=3, artifact_tag=None):
    import joblib


    path = _artifact(cache_dir, artifact_tag or dataset, "surrogate")
    if path and path.exists() and not refresh:
        blob = joblib.load(path)
        ctx.surr_model[dataset] = blob["model"]
        ctx.surr_scaler[dataset] = blob["scaler"]
        ctx.dp_dict[dataset] = {}
        return

    from surrogates.dataprofiles import train_dp_surr_model
    prev, ctx.eval_split = getattr(ctx, "eval_split", "val"), "val"
    try:
        res = train_dp_surr_model(ctx, {dataset: sources}, {dataset: val_df},
                                  [dataset], "g", n_jobs=n_jobs)
    finally:
        ctx.eval_split = prev
        ctx.reset_for_run()
    ctx.surr_model[dataset], ctx.surr_scaler[dataset] = res[dataset][0], res[dataset][1]
    ctx.dp_dict[dataset] = {}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(model=res[dataset][0], scaler=res[dataset][1],
                         split="val"), path)


def _build_gradients(ctx, sources, val_df, test_df, dataset, feat_cols,
                     cache_dir=None, refresh=False, n_jobs=-1,
                     artifact_tag=None):
    import joblib

    path = _artifact(cache_dir, artifact_tag or dataset, "gradients")
    if path and path.exists() and not refresh:
        blob = joblib.load(path)
        ctx.norm_g_sources[dataset] = blob["norm_g_sources"]
        ctx.g_val[dataset] = blob["g_val"]
        return


    from surrogates.gradientmatch import calculate_gradients, create_balanced_validation
    target_col, sens_col = TARGET_COL[dataset], SENS_COL[dataset]

    val_fair = {dataset: val_df}
    if sens_col:
        y = val_df[target_col]
        s = val_df[sens_col]
        X = val_df.drop(columns=[target_col])
        n = min(15000, len(val_df))
        idx = create_balanced_validation(X, y, s, n_samples=n,
                                         target_balance=0.5)
        fair = X.iloc[idx].copy()
        fair[target_col] = y.iloc[idx]
        val_fair = {dataset: fair.reset_index(drop=True)}

    g_sources, g_val = calculate_gradients(
        ctx, {dataset: sources}, {dataset: val_df}, val_fair,
        {dataset: test_df}, ctx.scaler, [dataset], "g", n_jobs=n_jobs)
    ctx.norm_g_sources[dataset], ctx.g_val[dataset] =        g_sources[dataset], g_val[dataset]
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(norm_g_sources=g_sources[dataset],
                         g_val=g_val[dataset], split="val"), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["ACSIncome", "ACSPublicCoverage", "ACSTravelTime"])
    ap.add_argument("--years", nargs="+", default=YEARS)
    ap.add_argument("--states", nargs="+", default=None,
                    )
    ap.add_argument("--raw-root", default="acs_raw")
    ap.add_argument("--out-dir", default="acs_cache")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC,
                    )
    ap.add_argument("--split-mode", default="stratified",
                    choices=["stratified", "legacy"],
                    )
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--keep-comb", action="store_true",
                    )
    ap.add_argument("--no-download", action="store_true",
                    )
    ap.add_argument("--method", default="normal",
                    choices=["normal", "dp", "gradmatch", "hybrid"],
                    )
    ap.add_argument("--surrogate-jobs", type=int, default=3,
                    )
    ap.add_argument("--with-surrogate", action="store_true")
    ap.add_argument("--with-gradients", action="store_true")
    ap.add_argument("--list-sources", action="store_true")
    args = ap.parse_args()

    if args.list_sources:
        for sid, st in source_id_map().items():
            pass
        return

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        ctx = build_acs_context(
            ds, years=args.years, states=args.states, raw_root=args.raw_root,
            cache_dir=args.out_dir, refresh=args.refresh,
            val_frac=args.val_frac, split_mode=args.split_mode,
            download=not args.no_download, method=args.method,
            with_surrogate=args.with_surrogate,
            with_gradients=args.with_gradients, keep_comb=args.keep_comb,
            surrogate_jobs=args.surrogate_jobs)
        meta = dict(dataset=ds, years=args.years,
                    split_mode=args.split_mode, val_frac=args.val_frac,
                    n_sources=len(ctx.source_X[ds]),
                    n_val=int(len(ctx.val_y[ds])),
                    n_test=int(len(ctx.test_y[ds])),
                    target_state=STATE_LIST[-1],
                    source_id_map=source_id_map())
        (out / f"{ds}_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


def repartition_stratified_skew(sources, n_sources, dataset="Scaled_Pubcov",
                                seed=42, n_per_source=5000, skew=0.30,
                                n_base=1000):
    if n_sources > n_base:
        raise ValueError(f"n_sources={n_sources} exceeds n_base={n_base}")
    target_col = TARGET_COL[dataset]
    pooled = pd.concat(sources, ignore_index=True)
    if "Source ID" in pooled.columns:
        pooled = pooled.drop(columns=["Source ID"])

    y = pooled[target_col].to_numpy()
    strata = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    if len(strata) != 2:
        raise ValueError(f"expected a binary target, found {len(strata)} classes")
    pos_label = max(strata)
    pos_idx, neg_idx = strata[pos_label], strata[min(strata)]

    rng = np.random.RandomState(seed)
    base_rate = len(pos_idx) / len(pooled)
    rates = np.clip(rng.uniform(base_rate - skew, base_rate + skew, n_sources),
                    0.02, 0.98)

    out = []
    for sid in range(n_sources):
        n_pos = int(round(rates[sid] * n_per_source))
        n_neg = n_per_source - n_pos
        take = np.concatenate([
            rng.choice(pos_idx, n_pos, replace=n_pos > len(pos_idx)),
            rng.choice(neg_idx, n_neg, replace=n_neg > len(neg_idx)),
        ])
        part = pooled.iloc[rng.permutation(take)].reset_index(drop=True)
        part["Source ID"] = sid
        out.append(part)
    return out


def build_scaled_pubcov_context(n_sources=1000, pool_seed=42,
                                n_base=1000, *, years=YEARS,
                                states=None, raw_root="acs_raw", cache_dir=None,
                                refresh=False, gain_lambda=None,
                                cost_type="zero", limit=10_000_000, k_max=15,
                                val_frac=VAL_FRAC, split_mode="stratified",
                                split_seed=None, download=True,
                                method="normal", with_surrogate=False,
                                with_gradients=False, keep_comb=False,
                                surrogate_jobs=3, gradient_jobs=-1):
    from core.context import ExperimentContext

    dataset = "Scaled_Pubcov"
    base = SYNTHETIC_BASE[dataset]
    df = _load_or_download(base, years, states, raw_root, cache_dir,
                           refresh, download)
    state_sources, val_df, test_df = split_ood(df, base, val_frac=val_frac,
                                               split_mode=split_mode,
                                               split_seed=split_seed)
    sources = repartition_stratified_skew(state_sources, n_sources,
                                          dataset=dataset, seed=pool_seed,
                                          n_base=n_base)
    scaler, feat_cols = fit_scaler(sources, dataset)
    comb = pd.concat(sources, ignore_index=True)

    if method in ("dp", "hybrid"):
        with_surrogate = True
    if method in ("gradmatch", "hybrid"):
        with_gradients = True

    ctx = ExperimentContext(
        dataset=dataset, method=method,
        gain_lambda=(GAIN_LAMBDA[dataset] if gain_lambda is None else gain_lambda),
        cost_type=cost_type, limit=limit, k_max=k_max,
        source_list=list(range(n_sources)),
    )
    ctx.comb[dataset] = comb if keep_comb else None
    ctx.scaler[dataset] = scaler
    ctx.test_source[dataset] = test_df

    ctx.source_X[dataset], ctx.source_y[dataset] = {}, {}
    ctx.source_sens[dataset] = {}
    for sid, frame in enumerate(sources):
        X, y = _xy(frame, dataset, scaler, feat_cols)
        ctx.source_X[dataset][sid] = X
        ctx.source_y[dataset][sid] = y
        if SENS_COL[dataset]:
            ctx.source_sens[dataset][sid] =                frame[SENS_COL[dataset]].to_numpy().astype(np.float32)

    Xv, yv = _xy(val_df, dataset, scaler, feat_cols)
    Xt, yt = _xy(test_df, dataset, scaler, feat_cols)
    ctx.val_X_scaled[dataset], ctx.val_y[dataset] = Xv, yv
    ctx.val_sens_masks[dataset] = _masks(val_df, dataset)
    ctx.test_X_scaled[dataset], ctx.test_y[dataset] = Xt, yt
    ctx.test_sens_masks[dataset] = _masks(test_df, dataset)
    ctx.eval_split = "val"

    sizes = [len(s) for s in sources]

    tag = f"{dataset}_n{n_sources}_b{n_base}_s{pool_seed}"
    if with_surrogate:
        _build_surrogate(ctx, sources, val_df, dataset, cache_dir, refresh,
                         n_jobs=surrogate_jobs, artifact_tag=tag)
    if with_gradients:
        _build_gradients(ctx, sources, val_df, test_df, dataset, feat_cols,
                         cache_dir, refresh, n_jobs=gradient_jobs,
                         artifact_tag=tag)
    return ctx


def spread_check(ctx, n_probe=20, seed=0):
    from core.metrics import get_profit
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(ctx.source_y[ctx.dataset]),
                     size=min(n_probe, len(ctx.source_y[ctx.dataset])),
                     replace=False)
    vals = [float(get_profit(ctx, [int(i)])) for i in ids]
    return dict(min=min(vals), max=max(vals), range=max(vals) - min(vals),
                values=vals)
