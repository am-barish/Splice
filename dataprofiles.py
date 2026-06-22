from __future__ import annotations

import gc
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from context import ExperimentContext
from metrics import estimate_bayes_error_fast


def get_base_rate(s, dataset):
    if len(s) == 0:
        return 1.0
    target = "PINCP" if dataset == "ACSIncome" else "PUBCOV"
    sens = "SEX" if dataset == "ACSIncome" else "RAC1P"
    priv = s[s[sens] == 1]
    prot = s[s[sens] == 2]
    if len(priv) == 0 or len(prot) == 0:
        return 0.0
    return float((priv[target] == 1).mean() - (prot[target] == 1).mean())


def get_or_fit_pca(ctx, df):
    key = ctx.dataset
    if key not in ctx.pca_cache:
        pca = PCA(n_components=0.95)
        pca.fit(df)
        ctx.pca_cache[key] = pca
    return ctx.pca_cache[key]


def calculate_data_profiles(ctx, df, target: pd.Series):
    dp = {}
    dp["num_rows"] = float(len(df))

    for col in df.select_dtypes(include=["object", "category"]).columns:
        probs = df[col].value_counts(normalize=True)
        dp[f"entropy_{col}"] = float(entropy(probs, base=2))

    pca = get_or_fit_pca(ctx, df)
    df_transformed = pca.transform(df)

    corr_arr = df.corrwith(target)
    kurt = df.kurtosis()
    skw = df.skew()

    dp["num_pca"] = pca.n_components_ / len(df.columns)
    dp["first_pc_skewness"] = float(stats.skew(df_transformed[:, 0]))
    dp["first_pc_kurtosis"] = float(stats.kurtosis(df_transformed[:, 0]))

    for col in df.columns:
        val = corr_arr.get(col, np.nan)
        dp[f"corr_{col}"] = 0.0 if np.isnan(val) else float(val)

    dp["corr_min"] = float(np.nanmin(corr_arr))
    dp["corr_max"] = float(np.nanmax(corr_arr))
    dp["corr_mean"] = float(np.nanmean(corr_arr))
    dp["corr_std"] = float(np.nanstd(corr_arr))
    dp["kurtosis_min"] = float(np.min(kurt))
    dp["kurtosis_max"] = float(np.max(kurt))
    dp["kurtosis_mean"] = float(np.mean(kurt))
    dp["kurtosis_std"] = float(np.std(kurt))
    dp["skewness_min"] = float(np.min(skw))
    dp["skewness_max"] = float(np.max(skw))
    dp["skewness_mean"] = float(np.mean(skw))
    dp["skewness_std"] = float(np.std(skw))

    if ctx.dataset != "ACSTravelTime":
        pos = (target == 1).sum()
        neg = (target == 0).sum()
        dp["class_imbalance"] = float(pos / neg) if neg > 0 else 0.0
    else:
        dp["target_mean"] = float(np.mean(target))
        dp["target_std"] = float(np.std(target))

    return dp


def _build_profile_for_subset(ctx, subset, target_col, sens_col):
    if not subset:
        return None

    key = frozenset(subset)
    cached = ctx.get_dp(key)
    if cached is not None:
        return cached

    if ctx.source_X and ctx.dataset in ctx.source_X:
        X_all = np.vstack([ctx.source_X[ctx.dataset][s] for s in subset])
        y_all = np.concatenate([ctx.source_y[ctx.dataset][s] for s in subset])
        sample_n = max(1, int(0.01 * len(X_all)))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_all), sample_n, replace=False)
        n_cols = X_all.shape[1]
        feat_cols = [f"f{i}" for i in range(n_cols)]
        x_df = pd.DataFrame(X_all[idx], columns=feat_cols)
        y_s  = pd.Series(y_all[idx].astype(float))
    else:
        comb = ctx.comb[ctx.dataset]
        d = comb[comb["Source ID"].isin(subset)]
        d = d.sample(max(1, int(0.01 * len(d))), random_state=42)
        x_cols = [c for c in d.columns if c not in ["State", "Year", target_col, "Source ID"]]
        x_df = pd.DataFrame(
            ctx.scaler[ctx.dataset].transform(d[x_cols]),
            columns=x_cols, index=d[x_cols].index,
        )
        y_s = d[target_col]

    profile = list(calculate_data_profiles(ctx, x_df, y_s).values())

    if ctx.is_fairness_dataset and sens_col:
        if ctx.source_X and ctx.dataset in ctx.source_X:
            sens_all = np.concatenate([ctx.source_sens[ctx.dataset][s] for s in subset])
            mask1, mask2 = sens_all == 1, sens_all == 2
            br = float((y_all[mask1] == 1).mean() - (y_all[mask2] == 1).mean()) \
                 if mask1.any() and mask2.any() else 0.0
            ber = estimate_bayes_error_fast(y_all[idx], sens_all[idx])
        else:
            br  = get_base_rate(d, ctx.dataset)
            ber = estimate_bayes_error_fast(y_s.values, d[sens_col].values)
        ber_diff = (
            ber.get(1, {}).get("Bayes_error_estimate", 0.0)
            - ber.get(2, {}).get("Bayes_error_estimate", 0.0)
        )
        profile.append(br)
        profile.append(ber_diff)

    with ctx._cache_lock:
        existing = ctx.dp_dict[ctx.dataset].get(key)
        if existing is None:
            ctx.dp_dict[ctx.dataset][key] = profile
        else:
            profile = existing

    return profile


def get_data_profile_difference(ctx, s1, s2):
    target_col = ctx.target_col
    sens_col = ctx.sensitive_col

    l1 = _build_profile_for_subset(ctx, s1, target_col, sens_col)
    l2 = _build_profile_for_subset(ctx, s2, target_col, sens_col)

    if l1 is None and l2 is None:
        return []
    if l1 is None:
        l1 = [0.0] * len(l2)
    if l2 is None:
        l2 = [0.0] * len(l1)

    return [b - a for a, b in zip(l1, l2)]


def get_actual_marginal(ctx, s1, s2):
    from metrics import get_profit
    return get_profit(ctx, s2) - get_profit(ctx, s1)


def _dp_worker(ctx, s1, s2):
    dp_diff = get_data_profile_difference(ctx, s1, s2)
    a_mg = get_actual_marginal(ctx, s1, s2)
    return dp_diff, a_mg


def train_dp_surr_model(ctx, train_sources, test_source, datasets, metric, n_jobs= -1):
    result = {}
    np.random.seed(42)
    random.seed(42)

    for x in datasets:
        ctx.dataset = x
        N = len(train_sources[x]) if x != "Scaled_Pubcov" else 50
        problem = "regression" if x == "ACSTravelTime" else "classification"

        subset_pairs= []
        for i in range(N):
            others = list(set(range(N)) - {i})
            for _ in range(5):
                s1 = random.sample(others, np.random.randint(1, N - 1))
                s2 = s1 + [i]
                subset_pairs.append((s1, s2))
            for _ in range(5):
                s1 = random.sample(others, np.random.randint(1, N - 1))
                s2 = s1 + [i]
                subset_pairs.append((s2, s1))

        if n_jobs == 1:
            results_raw = [
                _dp_worker(ctx, sp[0], sp[1])
                for sp in tqdm(subset_pairs, desc=f"Surrogate [{x}]")
            ]
        else:
            results_raw = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_dp_worker)(ctx, sp[0], sp[1])
                for sp in tqdm(subset_pairs, desc=f"Surrogate [{x}] (parallel)")
            )

        dp_diffs = [r[0] for r in results_raw]
        a_mg = [r[1] for r in results_raw]

        target_col = ctx.target_col
        test_cols = test_source[x].drop(columns=["State", "Year", target_col]).columns
        corr_arr = [f"corr_{col}" for col in test_cols]
        if problem == "classification":
            prob_arr = ["class_imbalance", "base_rate", "bayes_error_rate"]
        else:
            prob_arr = ["target_mean", "target_std"]

        col_names = (
            ["num_rows", "num_pca", "first_pc_skewness", "first_pc_kurtosis"]
            + corr_arr
            + [
                "corr_min", "corr_max", "corr_mean", "corr_std",
                "kurtosis_min", "kurtosis_max", "kurtosis_mean", "kurtosis_std",
                "skewness_min", "skewness_max", "skewness_mean", "skewness_std",
            ]
            + prob_arr
        )

        surr_df = pd.DataFrame(dp_diffs, columns=col_names)
        surr_df["Actual marginal"] = a_mg

        m = int(3 / 5 * len(surr_df))
        X_tr = surr_df.iloc[:m].drop(columns=["Actual marginal"])
        X_te = surr_df.iloc[m:].drop(columns=["Actual marginal"])
        y_tr = surr_df["Actual marginal"].iloc[:m]
        y_te = surr_df["Actual marginal"].iloc[m:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        lr = LinearRegression()
        lr.fit(X_tr_sc, y_tr)

        train_mse = mean_squared_error(y_tr, lr.predict(X_tr_sc))
        test_mse = mean_squared_error(y_te, lr.predict(X_te_sc))

        result[x] = (
            lr,
            sc,
            {
                "train": {"mse": train_mse, "rmse": np.sqrt(train_mse)},
                "test": {"mse": test_mse, "rmse": np.sqrt(test_mse)},
            },
            surr_df,
        )

    return result
