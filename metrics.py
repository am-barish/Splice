from __future__ import annotations

import copy
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from joblib import Memory
from functools import cache
from context import ExperimentContext

def train_eval(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(
        C=0.1, max_iter=500, random_state=42
    )
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_te, clf.predict(X_te), zero_division=0))

def santos_tables_gain(ctx, indices):
    X_selected  = np.vstack([ctx.xs[i] for i in indices])
    y_selected  = np.concatenate([ctx.ys[i] for i in indices])
    f1_selected = train_eval(X_selected, y_selected, ctx.X_t, ctx.y_t)
    return f1_selected

def compute_accuracy(y_true, y_pred):
    return float(np.sum((y_pred > 0.5) == y_true) / len(y_pred))


def compute_fairness(y_pred, X_test, y_test, metric, dataset):

    if dataset in ("ACSIncome", "ACSIncome56"):
        protected_idx = X_test[X_test["SEX"] == 2].index
        privileged_idx = X_test[X_test["SEX"] == 1].index
    elif dataset in ("ACSPublicCoverage", "Scaled_Pubcov"):
        protected_idx = X_test[X_test["RAC1P"] == 2].index
        privileged_idx = X_test[X_test["RAC1P"] == 1].index
    else:
        return 0, 0, 0

    p_protected = y_pred[protected_idx].sum() / len(protected_idx)
    p_privileged = y_pred[privileged_idx].sum() / len(privileged_idx)
    statistical_parity = p_protected - p_privileged

    y_test_prot = y_test[protected_idx]
    y_pred_prot = y_pred[protected_idx]
    tp_prot = sum(y_pred_prot[y_test_prot == 1])
    ap_prot = len(y_test_prot[y_test_prot == 1])
    tpr_protected = tp_prot / ap_prot if ap_prot > 0 else 0

    y_test_priv = y_test[privileged_idx]
    y_pred_priv = y_pred[privileged_idx]
    tp_priv = sum(y_pred_priv[y_test_priv == 1])
    ap_priv = len(y_test_priv[y_test_priv == 1])
    tpr_privileged = tp_priv / ap_priv if ap_priv > 0 else 0

    tpr_parity = tpr_protected - tpr_privileged
    predictive_parity = 0.0

    if metric == 3:
        return statistical_parity, tpr_parity, predictive_parity
    return [statistical_parity, tpr_parity, predictive_parity][metric]


def _compute_fairness_from_masks(y_pred, y_test, sens_masks):
    if not sens_masks:
        return 0.0, 0.0, 0.0

    mask_priv = sens_masks.get(1, np.zeros(len(y_pred), dtype=bool))
    mask_prot = sens_masks.get(2, np.zeros(len(y_pred), dtype=bool))

    def _rate(pred, mask):
        return float(pred[mask].mean()) if mask.sum() > 0 else 0.0

    def _tpr(pred, true, mask):
        pos = true[mask] == 1
        return float(pred[mask][pos].sum() / pos.sum()) if pos.sum() > 0 else 0.0

    sp  = _rate(y_pred, mask_prot) - _rate(y_pred, mask_priv)
    tpr = _tpr(y_pred, y_test, mask_prot) - _tpr(y_pred, y_test, mask_priv)
    return sp, tpr, 0.0


def get_scores(ctx, source_subset, problem):
    dataset = ctx.dataset

    if ctx.source_X and dataset in ctx.source_X:
        X_train = np.vstack([ctx.source_X[dataset][s] for s in source_subset])
        y_train = np.concatenate([ctx.source_y[dataset][s] for s in source_subset])
        X_test  = ctx.test_X_scaled[dataset]
        y_test  = ctx.test_y[dataset]

        if problem == "classification":
            clf = LogisticRegression(solver="lbfgs", max_iter=100, tol=1e-4, random_state=42, n_jobs=1)
            clf.fit(X_train, y_train)
            y_pred   = clf.predict(X_test)
            accuracy = compute_accuracy(y_test, y_pred)
            sp, tpr, pp = _compute_fairness_from_masks(
                y_pred, y_test, ctx.test_sens_masks.get(dataset, {})
            )
            return sp, tpr, pp, accuracy
        else:
            reg    = LinearRegression().fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

    target       = ctx.target_col
    cols_to_drop = ["State", "Year", target] if dataset != "Flights" else ["OP_UNIQUE_CARRIER", target]
    comb         = ctx.comb[dataset]
    train_data   = comb[comb["Source ID"].isin(source_subset)].drop(columns=["Source ID"]).reset_index(drop=True)
    test_data    = ctx.test_source[dataset]

    X_train = train_data.drop(columns=cols_to_drop).reset_index(drop=True)
    y_train = train_data[target].reset_index(drop=True).astype("int")
    X_test  = test_data.drop(columns=cols_to_drop).reset_index(drop=True)
    y_test  = test_data[target].reset_index(drop=True).astype("int")

    normalized_X_train = ctx.scaler[dataset].transform(X_train)
    normalized_X_test  = ctx.scaler[dataset].transform(X_test)

    if problem == "classification":
        clf    = LogisticRegression(random_state=42)
        clf.fit(normalized_X_train, y_train)
        y_pred   = clf.predict(normalized_X_test)
        accuracy = compute_accuracy(y_test, y_pred)
        if dataset != "Flights":
            sp, tpr, pp = compute_fairness(y_pred, X_test, y_test, 3, dataset)
            return sp, tpr, pp, accuracy
        return accuracy, f1_score(y_test, y_pred)
    else:
        reg    = LinearRegression().fit(normalized_X_train, y_train)
        y_pred = reg.predict(normalized_X_test)
        return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


def get_gain(ctx, source_subset):
    from wilds_new import get_subset_acc
    from amazon import get_subset_score

    sub = frozenset(source_subset)
    x = ctx.dataset
    lam = ctx.gain_lambda

    if not source_subset:
        return -10_000_000 if ctx.is_regression else 0

    ctx.mark_covered(sub)

    if ctx.has_metric(sub):
        cached = ctx.get_metric(sub)
    else:
        cached = None

    if cached is None:
        if x in ("SANTOS", "SANTOS_IPO"):
            return santos_tables_gain(ctx, source_subset)
        elif x not in ("Wilds", "Amazon"):
            result = get_scores(ctx, source_subset,
                                "regression" if ctx.is_regression else "classification")
        elif x == "Wilds":
            result = get_subset_acc(source_subset, ctx)

        else:
            result = get_subset_score(
                source_subset,
                ctx,
            )
        ctx.set_metric(sub, result)
        cached = result

    if x == "ACSTravelTime":
        return -cached[0]
    if x == "Flights":
        return cached[1]
    if x in ("Wilds", "Amazon"):
        return cached
    return cached[3] * 100 + lam * cached[1]


def get_cost(ctx, source_subset):
    c = 0.0
    cost_type = ctx.cost_type
    x = ctx.dataset

    if cost_type == "zero":
        return 0.0
    if cost_type == "constant":
        return 0.02 * len(source_subset)

    thresholds = {
        "ACSIncome": 70, "ACSPublicCoverage": 70,
        "Flights": 90,
    }
    base = thresholds.get(x, 0)

    for i in source_subset:
        g = get_gain(ctx, [i]) - base
        if cost_type == "linear":
            c += g / 100
        elif cost_type == "quadratic":
            c += (g ** 2) / 100
        elif cost_type == "step":
            c += 0 if g < 2 else 0.2 if g < 4 else 0.4

    return max(c, 0.0)


def get_profit(ctx, source_subset):
    return get_gain(ctx, source_subset) - get_cost(ctx, source_subset)


def get_grad_profit(ctx, s1, s2):
    return subset_score(ctx, s2) - subset_score(ctx, s1)


def subset_score(ctx, indices):
    from amazon import estimated_gain_text
    from wilds_new import subset_grad

    x = ctx.dataset
    if not indices:
        return -np.inf

    if x == "Wilds":
        return subset_grad(ctx.g_val[x], ctx.norm_g_sources[x], indices)
    if x == "Amazon":
        return estimated_gain_text(ctx.g_val[x], ctx.norm_g_sources[x], indices)

    g_A_acc = sum(ctx.norm_g_sources[x][i][0] for i in indices) / len(indices)
    g_A_fair = sum(ctx.norm_g_sources[x][i][1] for i in indices) / len(indices)
    score_acc = cos_sim(g_A_acc, ctx.g_val[x][0])

    if ctx.is_fairness_dataset:
        score_fair = cos_sim(g_A_fair, ctx.g_val[x][1])
        return (1 - ctx.gain_lambda / 100) * score_acc + (ctx.gain_lambda / 100) * score_fair

    return score_acc


def get_marginal_profit_dp(ctx, s1, s2):
    from dataprofiles import get_data_profile_difference
    from dataprofiles_unstructured import get_marginal_profit_dp_unst

    if ctx.dataset not in ("Wilds", "Amazon"):
        dpdf = get_data_profile_difference(ctx, s1, s2)
        return ctx.surr_model[ctx.dataset].predict(
            ctx.surr_scaler[ctx.dataset].transform([dpdf])
        )
    return get_marginal_profit_dp_unst(ctx, s1, s2,
                                       ctx.source_pool[ctx.dataset],
                                       ctx.test_dict[ctx.dataset])


def get_subset_percentile(ctx, ss):
    return stats.percentileofscore(
        [get_profit(ctx, x) for x in ctx.subset_array],
        get_profit(ctx, ss),
        kind="weak",
    )


def get_subset_rank(ctx, ss):
    if not ss:
        return 100
    d = {str(sorted(set(k))): get_profit(ctx, k) for k in ctx.subset_array}
    ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    rank_map = {x[0]: i for i, x in enumerate(ranked)}
    return rank_map[str(sorted(set(ss)))] + 1


def init_subset_array(ctx, num):
    local_subsets = [[]]
    subset_array = []
    for el in ctx.source_list[:num]:
        for i in range(len(local_subsets)):
            local_subsets.append(local_subsets[i] + [el])
            subset_array.append(local_subsets[-1])
    ctx.subset_array = subset_array


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def estimate_bayes_error_fast(y, sensitive_attr):
    results = {}
    for group in np.unique(sensitive_attr):
        mask = sensitive_attr == group
        yg = np.asarray(y)[mask]
        if len(yg) == 0:
            results[group] = {"Bayes_error_estimate": 0.0}
            continue
        p = (yg == 1).mean()
        results[group] = {"Bayes_error_estimate": min(p, 1 - p)}
    return results


def calculate_kl_divergence_for_features(df1, df2, features, bins = 10):
    kl_divergences = {}
    for feature in features:
        if feature not in df1.columns or feature not in df2.columns:
            continue
        if df1[feature].dtype in ("object", "category"):
            p = df1[feature].value_counts(normalize=True)
            q = df2[feature].value_counts(normalize=True)
            cats = p.index.union(q.index)
            p_dist = p.reindex(cats, fill_value=1e-10).values
            q_dist = q.reindex(cats, fill_value=1e-10).values
            kl_divergences[feature] = entropy(p_dist, q_dist)
        else:
            lo = min(df1[feature].min(), df2[feature].min())
            hi = max(df1[feature].max(), df2[feature].max())
            edges = np.linspace(lo, hi, bins + 1)
            ph, _ = np.histogram(df1[feature], bins=edges, density=True)
            qh, _ = np.histogram(df2[feature], bins=edges, density=True)
            ph[ph == 0] = 1e-10
            qh[qh == 0] = 1e-10
            kl_divergences[feature] = entropy(ph, qh)
    return kl_divergences
