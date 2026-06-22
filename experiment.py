from __future__ import annotations

import os

from dataprofiles_unstructured import generate_meta_history

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from context import ExperimentContext


def set_reproducibility(seed= 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def build_source_arrays(ctx, datasets):
    import numpy as np

    _target = {
        "ACSIncome": "PINCP",
        "ACSPublicCoverage": "PUBCOV",
        "Scaled_Pubcov": "PUBCOV",
        "ACSTravelTime": "JWMNP",
    }
    _sens = {
        "ACSIncome": "SEX",
        "ACSPublicCoverage": "RAC1P",
        "Scaled_Pubcov": "RAC1P",
    }
    _drop = ["Year", "State", "Source ID"]

    for dataset in datasets:
        target = _target[dataset]
        sens_col = _sens.get(dataset)
        scaler = ctx.scaler[dataset]
        comb = ctx.comb[dataset]

        feat_cols = [c for c in comb.columns
                     if c not in _drop + [target]]

        src_ids = comb["Source ID"].values
        X_all = scaler.transform(comb[feat_cols].values).astype(np.float32)
        y_all = comb[target].values.astype(np.int8)

        ctx.source_X[dataset] = {}
        ctx.source_y[dataset] = {}
        ctx.source_sens[dataset] = {}

        for sid in np.unique(src_ids):
            mask = src_ids == sid
            ctx.source_X[dataset][sid] = X_all[mask]
            ctx.source_y[dataset][sid] = y_all[mask]
            if sens_col:
                ctx.source_sens[dataset][sid] = comb[sens_col].values[mask].astype(np.float32)

        test_df = ctx.test_source[dataset]
        test_feat_cols = [c for c in test_df.columns
                          if c not in ["Year", "State", target]]

        model_feat_cols = [c for c in test_feat_cols if c != "Source ID"]

        ctx.test_X_scaled[dataset] = scaler.transform(
            test_df[model_feat_cols].values
        ).astype(np.float32)
        ctx.test_y[dataset] = test_df[target].values.astype(np.int8)

        ctx.test_sens_masks[dataset] = {}
        if sens_col and sens_col in test_df.columns:
            sens_vals = test_df[sens_col].values
            for group in np.unique(sens_vals):
                ctx.test_sens_masks[dataset][int(group)] = (sens_vals == group)

        print(f"  [{dataset}] {len(np.unique(src_ids))} sources, "
              f"{X_all.shape} train, {ctx.test_X_scaled[dataset].shape} test")


def setup_acs_context(dataset, *, data_dir = ".", gain_lambda = 10.0, method = "normal", cost_type = "zero", limit = 100_000, k_max = 15, source_list = None, seed= 42):
    set_reproducibility(seed)
    p = Path(data_dir)

    def _load(fname):
        with open(p / fname, "rb") as f:
            return pickle.load(f)

    # Loading saved dicts for repeated experiments
    sorted_train = _load("SORTED_TRAIN_SOURCES_FINAL.pkl")
    test_src = _load("TEST_SOURCE_FINAL.pkl")
    comb = _load("COMB_DATA_FINAL.pkl")
    metrics_dict = _load("METRICS_DICT.pkl")
    scaler = _load("SCALER_FINAL.pkl")
    surr_model = _load("SURR_MODEL_FINAL.pkl")
    surr_scaler = _load("SURR_SCALER_FINAL.pkl")
    dp_dict = _load("DP_DICT_FINAL.pkl")
    norm_g = _load("NORM_G_SOURCES_FINAL.pkl")
    g_val = _load("G_VAL_FINAL.pkl")

    ctx = ExperimentContext(
        dataset=dataset,
        sorted_train_sources=sorted_train,
        test_source=test_src,
        comb=comb,
        scaler=scaler,
        surr_model=surr_model,
        surr_scaler=surr_scaler,
        dp_dict=dp_dict,
        metrics_dict=metrics_dict,
        norm_g_sources=norm_g,
        g_val=g_val,
        gain_lambda=gain_lambda,
        method=method,
        cost_type=cost_type,
        limit=limit,
        k_max=k_max,
        source_list=source_list or [],
    )
    print("Pre-extracting source arrays...")
    build_source_arrays(ctx, [dataset])
    return ctx


def setup_wilds_context(*, data_dir = ".", gain_lambda = 0.0, method = "gradmatch", limit= 100_000, source_list = None, seed = 42):
    import wilds_new as _wn
    from wilds_new import get_grads
    from dataprofiles_unstructured import (
        MetaSurrogatePipeline,
        prepare_wilds_pool,
    )

    set_reproducibility(seed)
    p = Path(data_dir)

    def _load(fname):
        with open(p / fname, "rb") as f:
            return pickle.load(f)

    ctx = ExperimentContext(
        dataset="Wilds",
        gain_lambda=gain_lambda,
        method=method,
        limit=limit,
        source_list=source_list or [],
        num_classes=_wn.num_classes,
    )
    ctx.test_source["Wilds"] = []

    ctx.norm_g_sources["Wilds"], ctx.g_val["Wilds"] = get_grads(ctx)

    s_dict = torch.load(p / "source.pt", weights_only=False, map_location="cpu")
    t_dict = torch.load(p / "target_test.pt", weights_only=False, map_location="cpu")
    ctx.source_pool["Wilds"] = prepare_wilds_pool(s_dict)
    ctx.test_dict["Wilds"] = t_dict

    ctx.kurt = _load("Wilds_target_profile.pkl")
    history = _load("history_list_wilds_latestt.pkl")
    surrogate = MetaSurrogatePipeline()
    surrogate.train_surrogate(ctx, history)
    ctx.surr_model["Wilds"] = surrogate
    ctx.dp_dict["Wilds"] = {}

    return ctx


def setup_amazon_context(*, data_dir = ".", gain_lambda = 0.0, method = "gradmatch", limit = 100_000, source_list = None, seed = 42):
    from amazon import get_amazon_info, amazon_grad_match, get_target_split
    from dataprofiles_unstructured import (
        MetaSurrogatePipeline,
        create_amazon_source_pool,
    )

    set_reproducibility(seed)
    p = Path(data_dir)

    def _load(fname):
        with open(p / fname, "rb") as f:
            return pickle.load(f)

    ctx = ExperimentContext(
        dataset="Amazon",
        gain_lambda=gain_lambda,
        method=method,
        limit=limit,
        source_list=source_list or [],
    )
    ctx.test_source["Amazon"] = []

    (
        ctx.amazon_source_cats,
        ctx.amazon_target_cats,
        ctx.amazon_source_sources,
        ctx.amazon_target_sources,
    ) = get_amazon_info()

    ctx.norm_g_sources["Amazon"], ctx.g_val["Amazon"] = amazon_grad_match(
        ctx.amazon_source_sources, ctx.amazon_target_sources, ctx.amazon_source_cats
    )

    target_cats = ctx.amazon_target_cats
    t_feats_list, t_labels_list = [], []
    for cat in target_cats:
        feats, labels, splits = ctx.amazon_target_sources[cat]
        mask = splits == "train"
        t_feats_list.append(feats[mask])
        t_labels_list.append(labels[mask])

    t_feats_np = np.vstack(t_feats_list)
    t_labels_np = np.hstack(t_labels_list)
    pos_mask = t_labels_np == 1
    neg_mask = t_labels_np == 0

    ctx.amazon_target_stats = {
        "pos_rate": float(pos_mask.mean()),
        "mu_global": t_feats_np.mean(axis=0),
        "mu_pos": t_feats_np[pos_mask].mean(axis=0) if pos_mask.any() else t_feats_np.mean(axis=0),
        "mu_neg": t_feats_np[neg_mask].mean(axis=0) if neg_mask.any() else t_feats_np.mean(axis=0),
        "sep": float(np.linalg.norm(
            t_feats_np[pos_mask].mean(axis=0) - t_feats_np[neg_mask].mean(axis=0)
        )) if pos_mask.any() and neg_mask.any() else 0.0,
    }

    source_pool_amazon = create_amazon_source_pool(ctx.amazon_source_sources)
    target_pool_amazon = create_amazon_source_pool(ctx.amazon_target_sources)

    merged_feats = torch.cat([target_pool_amazon[c]["feats"] for c in target_cats])
    merged_labels = torch.cat([target_pool_amazon[c]["labels"] for c in target_cats])
    ctx.source_pool["Amazon"] = source_pool_amazon
    ctx.test_dict["Amazon"] = {"feats": merged_feats, "labels": merged_labels}

    ctx.prev = _load("Amazon_target_profile.pkl")
    history = _load("history_list_amazon_latest.pkl")
    surrogate = MetaSurrogatePipeline()
    surrogate.train_surrogate(ctx, history)
    ctx.surr_model["Amazon"] = surrogate
    ctx.dp_dict["Amazon"] = {}

    return ctx

def setup_santos_context(*, method = "normal", limit = 100_000, source_list = None):
    ctx = ExperimentContext(
        dataset="SANTOS",
        gain_lambda=0,
        method=method,
        limit=limit,
        source_list=source_list or [],
    )
    ctx.dataset = "SANTOS"
    ctx.gain_lambda = 0

    return ctx


def run_experiment(ctx, algorithm, smax = None, parallel = False, n_jobs = -1, seed = 42):
    set_reproducibility(seed)
    ctx.reset_for_run()

    source_list = ctx.source_list
    if not source_list:
        raise ValueError("ctx.source_list is empty")

    _smax = smax if smax is not None else max(1, len(source_list) // 2)

    if algorithm == "splice":
        from splice import get_best_subset
        profit, subset, elapsed, explored, profits = get_best_subset(
            ctx, source_list, _smax, parallel=parallel, n_jobs=n_jobs
        )
        return dict(profit=profit, subset=subset, time_elapsed=elapsed,
                    models_explored=explored, profits=profits)

    if algorithm == "grasp":
        from grasp import grasp
        profit, subset, _, _, elapsed, explored, profits = grasp(ctx, source_list, 20, 5)
        return dict(profit=profit, subset=subset, time_elapsed=elapsed,
                    models_explored=explored, profits=profits)

    if algorithm == "dsdm":
        from dsdm import get_dsdm_result
        profit, subset, _, _, elapsed, explored = get_dsdm_result(ctx, source_list, 500)
        return dict(profit=profit, subset=subset, time_elapsed=elapsed,
                    models_explored=explored)

    if algorithm == "forward_greedy":
        from baselines import forward_greedy
        profit, subset, _, _, elapsed, explored = forward_greedy(ctx, source_list)
        return dict(profit=profit, subset=subset, time_elapsed=elapsed,
                    models_explored=explored)

    if algorithm == "brute_force":
        from baselines import brute_force
        profit, subset, pct, rank, elapsed, explored = brute_force(ctx, source_list)
        return dict(profit=profit, subset=subset, time_elapsed=elapsed,
                    models_explored=explored, percentile=pct, rank=rank)

    if algorithm == "random":
        from baselines import random_subset
        profit, subset, _, _, elapsed, explored = random_subset(ctx, source_list)
        return dict(profit=profit, subset=subset, time_elapsed=elapsed,
                    models_explored=explored)

    if algorithm == "all_sources":
        from baselines import all_sources_gain
        profit, _, _ = all_sources_gain(ctx, source_list)
        return dict(profit=profit)

    if algorithm == "single_source":
        from baselines import single_source_gain
        profit, _, _ = single_source_gain(ctx, source_list)
        return dict(profit=profit)

    if algorithm in ("k_center_coreset", "gradmatch_point"):
        budget = 906312
        if budget is None:
            raise ValueError(
                f"algorithm='{algorithm}' requires budget= to be passed as a kwarg. "
            )
        from coreset_baselines import k_center_coreset, gradmatch_point
        if algorithm == "k_center_coreset":
            return k_center_coreset(ctx, source_list, budget=budget, seed=seed)
        return gradmatch_point(
            ctx, source_list, budget=budget,
            batch_size=256, seed=seed,
        )

    raise ValueError(f"Unknown algorithm: {algorithm!r}. ")