from __future__ import annotations

import contextlib
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score

from core import metrics as _metrics
from core.metrics import compute_accuracy, _compute_fairness_from_masks

try:
    from data_prep.splits import VAL_FRAC
except ImportError:
    VAL_FRAC = 0.30

_INSTALLED = False


def _make_model(name, problem, seed):
    if name == "logreg":
        if problem == "classification":
            return LogisticRegression(solver="lbfgs", max_iter=100,
                                      tol=1e-4, random_state=seed, n_jobs=1)
        return LinearRegression()

    if name == "hist_gbdt":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        cls = (HistGradientBoostingClassifier if problem == "classification"
               else HistGradientBoostingRegressor)
        return cls(max_iter=200, max_depth=6, learning_rate=0.1,
                   random_state=seed)

    if name == "rf":
        from sklearn.ensemble import (RandomForestClassifier,
                                      RandomForestRegressor)
        cls = (RandomForestClassifier if problem == "classification"
               else RandomForestRegressor)
        return cls(n_estimators=200, n_jobs=1, random_state=seed)

    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        common = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                      random_state=seed, n_jobs=1, verbosity=0)
        return (XGBClassifier(eval_metric="logloss", **common)
                if problem == "classification" else XGBRegressor(**common))

    if name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        common = dict(n_estimators=200, max_depth=-1, num_leaves=63,
                      learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                      random_state=seed, n_jobs=1, verbose=-1)
        cls = LGBMClassifier if problem == "classification" else LGBMRegressor
        return cls(**common)

    raise ValueError(f"Unknown downstream model {name!r}")


def _arrays(ctx):
    if hasattr(ctx, "eval_arrays"):
        return ctx.eval_arrays()
    d = ctx.dataset
    split = getattr(ctx, "eval_split", "val")
    if split == "val":
        if not getattr(ctx, "val_X_scaled", None) or d not in ctx.val_X_scaled:
            raise RuntimeError(
                f"No validation split for {d}. Apply leak_fixes.md §1-2 "
                f"(context.py + build_source_arrays) before running these "
                f"experiments — selection must not read the test split.")
        return (ctx.val_X_scaled[d], ctx.val_y[d],
                ctx.val_sens_masks.get(d, {}))
    return (ctx.test_X_scaled[d], ctx.test_y[d],
            ctx.test_sens_masks.get(d, {}))


@contextlib.contextmanager
def using_split(ctx, split):
    prev = getattr(ctx, "eval_split", "val")
    prev_cfg = getattr(ctx, "_active_eval_config", None)
    ctx.eval_split = split
    ctx._active_eval_config = _config_tag(ctx)
    try:
        yield ctx
    finally:
        ctx.eval_split = prev
        ctx._active_eval_config = prev_cfg


def _config_tag(ctx):
    vi = getattr(ctx, "val_idx", None)
    vi_tag = None if vi is None else (len(vi), int(np.sum(vi)) % (2**31))
    return (getattr(ctx, "eval_split", "val"),
            getattr(ctx, "eval_seed", 42),
            getattr(ctx, "downstream_model", "logreg"),
            vi_tag)


_KEEP = object()


def set_eval_config(ctx, *, seed=None, model=None, val_idx=_KEEP,
                    split=None):
    if seed is not None:
        ctx.eval_seed = int(seed)
    if model is not None:
        ctx.downstream_model = str(model)
    if split is not None:
        if split not in ("val", "test"):
            raise ValueError("split must be 'val' or 'test'")
        ctx.eval_split = split
    if not hasattr(ctx, "eval_split"):
        ctx.eval_split = "val"
    if val_idx is not _KEEP:
        ctx.val_idx = val_idx
    ctx.reset_for_run()
    ctx._active_eval_config = _config_tag(ctx)
    return ctx


def _check_config(ctx):
    active = getattr(ctx, "_active_eval_config", None)
    if active is not None and active != _config_tag(ctx):
        raise RuntimeError(
            "Evaluation config changed without set_eval_config(); the profit "
            "cache is stale. Use set_eval_config(...) or using_split(...).")


_orig_get_scores = _metrics.get_scores


def get_scores_ext(ctx, source_subset, problem):
    dataset = ctx.dataset
    if not (ctx.source_X and dataset in ctx.source_X):
        return _orig_get_scores(ctx, source_subset, problem)

    _check_config(ctx)
    seed = int(getattr(ctx, "eval_seed", 42))
    model_name = getattr(ctx, "downstream_model", "logreg")

    X_train = np.vstack([ctx.source_X[dataset][s] for s in source_subset])
    y_train = np.concatenate([ctx.source_y[dataset][s] for s in source_subset])
    X_ev, y_ev, masks = _arrays(ctx)


    val_idx = getattr(ctx, "val_idx", None)
    if val_idx is not None and getattr(ctx, "eval_split", "val") == "val":
        X_ev = X_ev[val_idx]
        y_ev = y_ev[val_idx]
        masks = {g: m[val_idx] for g, m in masks.items()}

    est = _make_model(model_name, problem, seed)
    est.fit(X_train, y_train)
    y_pred = np.asarray(est.predict(X_ev))

    if problem == "classification":
        acc = compute_accuracy(y_ev, y_pred)
        sp, tpr, pp = _compute_fairness_from_masks(y_pred, y_ev, masks)
        return sp, tpr, pp, acc
    return mean_squared_error(y_ev, y_pred), r2_score(y_ev, y_pred)


def final_eval(ctx, subset, model=None, seed=None, return_preds=False):
    dataset = ctx.dataset
    problem = "regression" if ctx.is_regression else "classification"
    seed = int(seed if seed is not None else getattr(ctx, "eval_seed", 42))
    model = model or getattr(ctx, "downstream_model", "logreg")

    X_train = np.vstack([ctx.source_X[dataset][s] for s in subset])
    y_train = np.concatenate([ctx.source_y[dataset][s] for s in subset])

    with using_split(ctx, "test"):
        X_te, y_te, masks = _arrays(ctx)

    est = _make_model(model, problem, seed)
    est.fit(X_train, y_train)
    y_pred = np.asarray(est.predict(X_te))

    if problem == "classification":
        acc = compute_accuracy(y_te, y_pred)
        sp, tpr, _ = _compute_fairness_from_masks(y_pred, y_te, masks)
        out = dict(profit=float(acc * 100 + ctx.gain_lambda * tpr),
                   accuracy=float(acc), tpr_parity=float(tpr),
                   stat_parity=float(sp), model=model, seed=seed, split="test")
    else:
        mse = mean_squared_error(y_te, y_pred)
        out = dict(profit=float(-mse), mse=float(mse),
                   r2=float(r2_score(y_te, y_pred)),
                   model=model, seed=seed, split="test")
    if return_preds:
        out["y_pred"], out["y_test"] = y_pred, np.asarray(y_te)
    return out


evaluate_final_tabular = final_eval


def _patch_wilds():
    try:
        from data_prep import wilds as wn
    except Exception as e:
        return
    import torch

    def get_subset_acc_ext(subset, ctx):
        _check_config(ctx)
        seed = int(getattr(ctx, "eval_seed", 42))
        torch.manual_seed(seed); np.random.seed(seed)
        head = wn.train_on_subset(subset, wn.s_feats, wn.s_labels,
                                  wn.s_locs, wn.num_classes, wn.device)
        if getattr(ctx, "eval_split", "val") == "test":
            feats, labels = wn.t_feats, wn.t_labels
        else:
            feats, labels = wn.v_feats, wn.v_labels
            vi = getattr(ctx, "val_idx", None)
            if vi is not None:
                feats, labels = feats[vi], labels[vi]
        score = wn.evaluate_on_target(head, feats, labels, wn.device)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return score

    def final_eval_wilds(ctx, subset, seed=None):
        s = int(seed if seed is not None else getattr(ctx, "eval_seed", 42))
        torch.manual_seed(s); np.random.seed(s)
        head = wn.train_on_subset(subset, wn.s_feats, wn.s_labels,
                                  wn.s_locs, wn.num_classes, wn.device)
        return dict(profit=float(wn.evaluate_on_target(
            head, wn.t_feats, wn.t_labels, wn.device)), seed=s, split="test")

    wn.get_subset_acc = get_subset_acc_ext
    globals()["final_eval_wilds"] = final_eval_wilds


def _patch_amazon():
    try:
        from data_prep import amazon as az
    except Exception as e:
        return
    import torch

    def _train_head(ctx, subset):
        cats = [ctx.amazon_source_cats[i] for i in subset]
        feats = [ctx.source_pool["Amazon"][c]["feats"].numpy() for c in cats]
        labels = [ctx.source_pool["Amazon"][c]["labels"].numpy() for c in cats]
        head = az.build_head_binary(device="cpu")
        az.train_binary(head, feats, labels, epochs=15)
        return head

    def _score(head, ctx, split, idx=None):
        head.eval()
        f1s = []
        with torch.no_grad():
            for cat, (feats, labs, splits) in ctx.amazon_target_sources.items():
                m = splits == split
                if m.sum() == 0:
                    continue
                Xv, yv = feats[m], np.asarray(labs[m])
                if idx is not None:
                    keep = idx[idx < len(yv)]
                    if len(keep) == 0:
                        continue
                    Xv, yv = Xv[keep], yv[keep]
                prob = torch.sigmoid(
                    head(torch.tensor(Xv).float())).numpy().reshape(-1)
                f1s.append(f1_score(yv, (prob > 0.5).astype(int),
                                    average="binary", zero_division=0))
        return float(np.mean(f1s)) if f1s else 0.0

    def get_subset_score_ext(subset, ctx):
        _check_config(ctx)
        seed = int(getattr(ctx, "eval_seed", 42))
        torch.manual_seed(seed); np.random.seed(seed)
        head = _train_head(ctx, subset)


        if getattr(ctx, "eval_split", "val") == "test":
            return _score(head, ctx, "train")
        return _score(head, ctx, "val", getattr(ctx, "val_idx", None))

    def final_eval_amazon(ctx, subset, seed=None):
        s = int(seed if seed is not None else getattr(ctx, "eval_seed", 42))
        torch.manual_seed(s); np.random.seed(s)
        return dict(profit=_score(_train_head(ctx, subset), ctx, "train"),
                    seed=s, split="test")

    az.get_subset_score = get_subset_score_ext
    globals()["final_eval_amazon"] = final_eval_amazon


def _patch_baselines():
    import random as _random
    import time
    try:
        from baselines import classical as bl
    except Exception:
        return
    from core.metrics import get_profit

    def random_subset_ext(ctx, source_list):
        start = time.time()
        rng = _random.Random(int(getattr(ctx, "eval_seed", 42)))
        smx = 15 if ctx.dataset == "Wilds" else 10 if ctx.dataset == "Amazon" else 7
        hi = min(smx, len(source_list))
        subsets = [rng.sample(source_list, rng.randint(1, hi)) for _ in range(10)]
        results = [get_profit(ctx, s) for s in subsets]

        ctx.random_subsets = subsets
        ctx.algo_over = True
        return (float(np.mean(results)), [], 0, 0,
                time.time() - start, ctx.covered_count())

    bl.random_subset = random_subset_ext


_FAMILY_BY_DATASET = {
    "ACSIncome": "acs", "ACSPublicCoverage": "acs", "ACSTravelTime": "acs",
    "Scaled_Pubcov": "acs", "Wilds": "wilds", "Amazon": "amazon",
}


def _resolve_family(family, dataset):
    inferred = _FAMILY_BY_DATASET.get(dataset)
    if inferred is None:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Expected one of "
            f"{sorted(_FAMILY_BY_DATASET)}.")
    if inferred != family:
        pass
    return inferred


def make_ctx(family="acs", dataset=None, *, cache_dir="acs_cache",
             data_dir=".", source_list=None, val_frac=VAL_FRAC,
             split_mode="stratified", split_seed=None, legacy_pickles=False,
             method="normal", with_surrogate=False, with_gradients=False,
             gain_lambda=None, n_sources=1000, pool_seed=42,
             n_base=1000):
    family = _resolve_family(family, dataset)
    if family != "acs" and cache_dir not in (None, "acs_cache"):
        pass

    if dataset == "Scaled_Pubcov":


        from data_prep.acs import build_scaled_pubcov_context
        ctx = build_scaled_pubcov_context(
            n_sources=n_sources, pool_seed=pool_seed,
            n_base=n_base, cache_dir=cache_dir,
            val_frac=val_frac, split_mode=split_mode, split_seed=split_seed,
            method=method, gain_lambda=gain_lambda,
            with_surrogate=with_surrogate, with_gradients=with_gradients)
        if source_list:
            ctx.source_list = list(source_list)
        ctx.method = method
        return ctx

    if family == "acs":
        if legacy_pickles:
            from core.experiment import setup_acs_context
            lam = ({"ACSIncome": 50.0}.get(dataset, 0.0)
                   if gain_lambda is None else gain_lambda)
            ctx = setup_acs_context(dataset, data_dir=data_dir, gain_lambda=lam)
        else:
            try:
                from data_prep.acs import build_acs_context
            except ImportError as e:
                raise ImportError(
                    "Could not import data_prep.acs. "
                                        "pass --legacy-pickles to use the old *_FINAL.pkl path."
                ) from e
            ctx = build_acs_context(
                dataset, cache_dir=cache_dir, source_list=source_list,
                val_frac=val_frac, split_mode=split_mode,
                split_seed=split_seed, gain_lambda=gain_lambda, method=method,
                with_surrogate=with_surrogate, with_gradients=with_gradients)
    elif family == "wilds":
        from core.experiment import setup_wilds_context
        ctx = setup_wilds_context(data_dir=data_dir, method=method)
    elif family == "amazon":
        from core.experiment import setup_amazon_context
        ctx = setup_amazon_context(data_dir=data_dir, method=method)
    else:
        raise ValueError(f"unknown family {family!r}")

    if source_list:
        ctx.source_list = list(source_list)
    if not hasattr(ctx, "eval_split"):
        ctx.eval_split = "val"
    ctx.method = method
    return ctx


def assert_split_ready(ctx):
    d = ctx.dataset
    if d in ("Wilds", "Amazon"):
        return
    if not getattr(ctx, "val_X_scaled", None) or d not in ctx.val_X_scaled:
        raise RuntimeError(
            f"{d} has no validation split. Apply leak_fixes.md §1-2 first; "
            f"otherwise selection reads the test set and every number is "
            f"optimistically biased.")
    nv = len(ctx.val_y[d]); nt = len(ctx.test_y[d])


def install(patch_wilds=True, patch_amazon=True, patch_baselines=True):
    global _INSTALLED
    if _INSTALLED:
        return
    _metrics.get_scores = get_scores_ext
    if patch_baselines:
        _patch_baselines()
    if patch_wilds:
        _patch_wilds()
    if patch_amazon:
        _patch_amazon()
    _INSTALLED = True
