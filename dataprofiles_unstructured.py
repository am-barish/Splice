from __future__ import annotations

from copyreg import pickle
from typing import Any, Dict, List, Optional

import gc
import numpy as np
import torch
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
import random

from context import ExperimentContext


def get_universal_profile(feats, labels, target_feats, num_classes):
    feats_np = feats.numpy() if torch.is_tensor(feats) else np.asarray(feats)
    labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    target_np = target_feats.numpy() if torch.is_tensor(target_feats) else np.asarray(target_feats)

    if feats_np.ndim == 1:
        feats_np = feats_np.reshape(1, -1)

    dp = np.zeros(7)

    counts = np.bincount(labels_np.astype(int), minlength=num_classes)
    dp[0] = (counts > 0).mean()
    probs = counts / (counts.sum() + 1e-8)
    dp[1] = -np.sum(probs * np.log(probs + 1e-8))

    n_comp = min(feats_np.shape[0], feats_np.shape[1], 10)
    if n_comp > 1:
        pca = PCA(n_components=n_comp)
        pca.fit(feats_np)
        dp[2] = np.sum(pca.explained_variance_ratio_)
        transformed = pca.transform(feats_np)
        dp[3] = stats.skew(transformed[:, 0])
        dp[4] = stats.kurtosis(transformed[:, 0])

    dp[5] = np.linalg.norm(np.mean(feats_np, axis=0) - np.mean(target_np, axis=0))

    unique_labels = np.unique(labels_np)
    if len(unique_labels) > 1 and feats_np.shape[0] > len(unique_labels):
        overall_mean = np.mean(feats_np, axis=0)
        between_var = sum(
            len(feats_np[labels_np == c]) *
            np.linalg.norm(np.mean(feats_np[labels_np == c], axis=0) - overall_mean) ** 2
            for c in unique_labels
        )
        within_var = sum(np.var(feats_np[labels_np == c], axis=0).sum() for c in unique_labels)
        dp[6] = between_var / (within_var + 1e-8)

    return dp


def get_wilds_profile(feats, labels, target_feats, target_labels, num_classes, target_counts, tail_classes, mu_T):
    feats_np = feats.numpy() if torch.is_tensor(feats) else np.asarray(feats)
    labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    target_np = target_feats.numpy() if torch.is_tensor(target_feats) else np.asarray(target_feats)
    tgt_labels_np = target_labels.numpy() if torch.is_tensor(target_labels) else np.asarray(target_labels)

    dp = []

    counts = np.bincount(labels_np.astype(int), minlength=num_classes)
    total = counts.sum() + 1e-8
    present = counts > 0

    dp.append(present.mean())

    tail_mask = np.zeros(num_classes, dtype=bool)
    tail_mask[tail_classes] = True
    dp.append((present & tail_mask).sum() / (tail_mask.sum() + 1e-8))

    present_counts = counts[present]
    if len(present_counts) > 0:
        dp.append(np.log1p(present_counts.min()))
        dp.append(np.log1p(present_counts.mean()))
        dp.append(present_counts.max() / (present_counts.min() + 1e-8))
    else:
        dp += [0.0, 0.0, 0.0]

    tgt_counts = np.bincount(tgt_labels_np.astype(int), minlength=num_classes).astype(float)
    p = counts / total
    q = tgt_counts / (tgt_counts.sum() + 1e-8)
    m = 0.5 * (p + q)
    js_div = 0.5 * (
        np.sum(p * np.log((p + 1e-8) / (m + 1e-8))) +
        np.sum(q * np.log((q + 1e-8) / (m + 1e-8)))
    )
    dp.append(js_div)

    cls_dists, tail_cls_dists = [], []
    for c in range(num_classes):
        if counts[c] == 0 or target_counts[c] == 0:
            continue
        mu_S = feats_np[labels_np == c].mean(axis=0)
        dist = np.linalg.norm(mu_S - mu_T[c])
        cls_dists.append(dist)
        if c in tail_classes:
            tail_cls_dists.append(dist)

    dp.append(np.mean(cls_dists) if cls_dists else 0.0)
    dp.append(np.mean(tail_cls_dists) if tail_cls_dists else 0.0)

    compact = [
        feats_np[labels_np == c].std(axis=0).mean()
        for c in range(num_classes) if counts[c] > 1
    ]
    dp.append(np.mean(compact) if compact else 0.0)

    mu_S_global = feats_np.mean(axis=0)
    mu_T_global = target_np.mean(axis=0)
    dp.append(np.linalg.norm(mu_S_global - mu_T_global))
    dp.append(feats_np.var(axis=0).mean() / (target_np.var(axis=0).mean() + 1e-8))

    return np.array(dp, dtype=np.float32)


def get_amazon_profile(feats, labels, target_feats, target_labels, target_stats):
    feats_np = feats.numpy() if torch.is_tensor(feats) else np.asarray(feats)
    labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)

    if feats_np.ndim == 1:
        feats_np = feats_np.reshape(1, -1)

    N = len(labels_np)
    if N == 0:
        return np.zeros(12, dtype=np.float32)

    dp = []
    pos_mask = labels_np == 1
    neg_mask = labels_np == 0
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
    pos_rate = n_pos / float(N)

    dp.append(np.log1p(N))
    dp.append(pos_rate)
    effective_n = 2 * n_pos * n_neg / (n_pos + n_neg + 1e-8)
    dp.append(np.log1p(effective_n))

    p = np.array([1 - pos_rate, pos_rate])
    q = np.array([1 - target_stats["pos_rate"], target_stats["pos_rate"]])
    m = 0.5 * (p + q)
    js = 0.5 * (
        np.sum(p * np.log((p + 1e-8) / (m + 1e-8))) +
        np.sum(q * np.log((q + 1e-8) / (m + 1e-8)))
    )
    dp.append(js)

    mu_S = feats_np.mean(axis=0)
    std_T = target_stats.get("std_global", np.ones_like(target_stats["mu_global"]))
    dp.append(np.linalg.norm((mu_S - target_stats["mu_global"]) / (std_T + 1e-8)))

    mu_S_pos = feats_np[pos_mask].mean(axis=0) if n_pos > 0 else mu_S
    mu_S_neg = feats_np[neg_mask].mean(axis=0) if n_neg > 0 else mu_S
    dp.append(np.linalg.norm(mu_S_pos - target_stats["mu_pos"]) if n_pos > 0 else 0.0)
    dp.append(np.linalg.norm(mu_S_neg - target_stats["mu_neg"]) if n_neg > 0 else 0.0)

    if n_pos > 0 and n_neg > 0:
        pooled_std = (
            feats_np[pos_mask].std(axis=0) * n_pos +
            feats_np[neg_mask].std(axis=0) * n_neg
        ) / (n_pos + n_neg + 1e-8)
        sep_S = np.linalg.norm(
            (mu_S_pos - mu_S_neg) / (pooled_std + 1e-8)
        )
    else:
        sep_S = 0.0

    dp.append(sep_S)
    dp.append(abs(sep_S - target_stats["sep"]))

    comp = []
    if n_pos > 1:
        comp.append(feats_np[pos_mask].std(axis=0).mean())
    if n_neg > 1:
        comp.append(feats_np[neg_mask].std(axis=0).mean())
    dp.append(np.mean(comp) if comp else 0.0)

    def _cos(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    dp.append(_cos(mu_S_pos, target_stats["mu_pos"]) if n_pos > 0 else 0.0)
    dp.append(_cos(mu_S_neg, target_stats["mu_neg"]) if n_neg > 0 else 0.0)


    if n_pos > 1 and n_neg > 1:
        within_scatter = (feats_np[pos_mask].var(axis=0).sum() * n_pos +
                        feats_np[neg_mask].var(axis=0).sum() * n_neg) / N
        between_scatter = pos_rate * (1 - pos_rate) * np.dot(mu_S_pos - mu_S_neg, mu_S_pos - mu_S_neg)
        fisher_ratio = between_scatter / (within_scatter + 1e-8)
    else:
        fisher_ratio = 0.0
    dp.append(fisher_ratio)

    cos_global = np.dot(mu_S, target_stats["mu_global"]) / (
        np.linalg.norm(mu_S) * np.linalg.norm(target_stats["mu_global"]) + 1e-8
    )
    dp.append(cos_global)

    var_S = feats_np.var(axis=0).mean()
    var_T = target_stats.get("var_global", feats_np.var(axis=0).mean())
    dp.append(var_S / (var_T + 1e-8))

    return np.array(dp, dtype=np.float32)


def get_marginal_profit_dp_unst(ctx, s1, s2, source_pool, target_data):
    unique_locs = sorted(list(source_pool.keys()))

    t_feats = target_data["feats"]
    t_labels = target_data["labels"]
    num_classes = ctx.num_classes
    t_labels_np = t_labels.numpy()
    target_counts = np.bincount(t_labels_np, minlength=num_classes)
    present = np.where(target_counts > 0)[0]
    freq_order = present[np.argsort(target_counts[present])]
    tail_classes = freq_order[:int(0.3 * num_classes)]
    mu_T = {
        c: t_feats[t_labels_np == c].numpy().mean(axis=0) if (t_labels_np == c).sum() > 0 else None
        for c in range(num_classes)
    }
    mu_T = {k: v for k, v in mu_T.items() if v is not None}

    def _profile(subset_indices):
        if not subset_indices:
            return np.zeros(7 if ctx.dataset != "Wilds" else 11)
        ids = [unique_locs[i] for i in subset_indices]
        f = torch.cat([source_pool[sid]["feats"] for sid in ids], dim=0)
        l = torch.cat([torch.atleast_1d(source_pool[sid]["labels"]) for sid in ids], dim=0)
        if ctx.dataset == "Wilds":
            return get_wilds_profile(f, l, t_feats, t_labels, num_classes,
                                     target_counts, tail_classes, mu_T)
        return get_universal_profile(f, l, t_feats, num_classes)

    def _profile_amazon(subset_indices):
        if not subset_indices:
            return np.zeros(15, dtype=np.float32)
        source_ids = sorted(list(source_pool.keys()))
        ids = [source_ids[i] for i in subset_indices]
        f = torch.cat([source_pool[sid]["feats"] for sid in ids], dim=0)
        l = torch.cat([torch.atleast_1d(source_pool[sid]["labels"]) for sid in ids], dim=0)
        return get_amazon_profile(f, l, t_feats, t_labels, ctx.amazon_target_stats)

    dd = ctx.dp_dict[ctx.dataset]
    k1, k2 = str(sorted(s1)), str(sorted(s2))

    if k1 not in dd:
        dd[k1] = _profile(s1) if ctx.dataset != "Amazon" else _profile_amazon(s1)
    if k2 not in dd:
        dd[k2] = _profile(s2) if ctx.dataset != "Amazon" else _profile_amazon(s2)

    return ctx.surr_model[ctx.dataset].estimate_gain_ctx(ctx, dd[k1], dd[k2])


class MetaSurrogatePipeline:
    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.is_trained = False

    def train_surrogate(self, ctx: ExperimentContext, history_list: List):
        p_T = ctx.kurt if ctx.dataset == "Wilds" else ctx.prev
        X_meta, y_meta = [], []
        for entry in history_list:
            p1, p2 = entry["p1"], entry["p2"]
            diff = p2 - p1
            row = np.concatenate([
                p1, p2, diff,
                np.abs(p1 - p_T),
                np.abs(p2 - p_T),
            ])
            X_meta.append(row)
            y_meta.append(entry["actual_f1_gain"])
        self.model.fit(np.array(X_meta), np.array(y_meta))
        self.is_trained = True
        y_pred_train = self.model.predict(np.array(X_meta))
        # from scipy.stats import spearmanr
        # corr, _ = spearmanr(y_pred_train, np.array(y_meta))
        # print(f"Surrogate train Spearman r={corr:.3f} on {len(X_meta)} samples")
        # print(f"Surrogate trained on {len(X_meta)} samples.")

    def estimate_gain_ctx(self, ctx, p1, p2):
        p_T = ctx.kurt if ctx.dataset == "Wilds" else ctx.prev
        diff = p2 - p1
        row = np.concatenate([p1, p2, diff, np.abs(p1 - p_T), np.abs(p2 - p_T)]).reshape(1, -1)
        return float(self.model.predict(row)[0])


def generate_meta_history(ctx, source_pool, target_data, iterations_per_source = 5):
    from wilds_new import get_subset_acc
    from amazon import get_subset_f1_amazon

    random.seed(42)
    history = []
    source_ids = list(source_pool.keys())
    ids2idx = {s: i for i, s in enumerate(source_ids)}
    N = len(source_ids)

    if ctx.dataset == "Amazon":
        subset_score_fn = get_subset_f1_amazon
    else:
        subset_score_fn = get_subset_acc

    t_feats = target_data["feats"]
    t_labels = target_data["labels"]
    num_classes = ctx.num_classes
    t_labels_np = t_labels.numpy()
    target_counts = np.bincount(t_labels_np, minlength=num_classes)
    present = np.where(target_counts > 0)[0]
    freq_order = present[np.argsort(target_counts[present])]
    tail_classes = freq_order[:int(0.3 * num_classes)]
    mu_T = {
        c: t_feats[t_labels_np == c].numpy().mean(axis=0)
        for c in range(num_classes)
        if (t_labels_np == c).sum() > 0
    }
    import pickle
    if ctx.dataset != "Amazon":
        p_T = get_wilds_profile(t_feats, t_labels, t_feats, t_labels,
                                num_classes, target_counts, tail_classes, mu_T)
        ctx.kurt = p_T
    else:
        p_T = get_amazon_profile(t_feats, t_labels, t_feats, t_labels, ctx.amazon_target_stats)
        with open("Amazon_target_profile.pkl", "wb") as f:
            pickle.dump(p_T, f)
        ctx.prev = p_T

    for src_id in tqdm(source_ids, desc="Generating meta-history"):
        other = [s for s in source_ids if s != src_id]
        for _ in range(iterations_per_source):
            if ctx.dataset != "Amazon":
                s1_ids = random.sample(other, k=random.randint(1, N - 1))
            else:
                max_k = N - 1
                target_size = random.choice([
                    random.randint(1, max(1, max_k // 4)),
                    random.randint(max_k // 4, max_k // 2),
                    random.randint(max_k // 2, max_k),
                ])
                s1_ids = random.sample(other, k=min(target_size, len(other)))
            s2_ids = s1_ids + [src_id]
            for start_ids, end_ids in [(s1_ids, s2_ids), (s2_ids, s1_ids)]:
                f_s = torch.cat([source_pool[s]["feats"] for s in start_ids])
                l_s = torch.cat([source_pool[s]["labels"] for s in start_ids])
                f_e = torch.cat([source_pool[s]["feats"] for s in end_ids])
                l_e = torch.cat([source_pool[s]["labels"] for s in end_ids])

                if ctx.dataset != "Wilds":
                    p1 = get_amazon_profile(f_s, l_s, t_feats, t_labels, ctx.amazon_target_stats)
                    p2 = get_amazon_profile(f_e, l_e, t_feats, t_labels, ctx.amazon_target_stats)
                else:
                    p1 = get_wilds_profile(f_s, l_s, t_feats, t_labels, num_classes,
                                           target_counts, tail_classes, mu_T)
                    p2 = get_wilds_profile(f_e, l_e, t_feats, t_labels, num_classes,
                                           target_counts, tail_classes, mu_T)

                gain = (
                    subset_score_fn([ids2idx[s] for s in end_ids], ctx) -
                    subset_score_fn([ids2idx[s] for s in start_ids], ctx)
                )
                history.append({"p1": p1, "p2": p2, "actual_f1_gain": gain})
                gc.collect()
                del f_s, l_s, f_e, l_e

    return history


def prepare_wilds_pool(wilds_dict):
    unique_locs = np.unique(wilds_dict["locs"])
    return {
        loc: {
            "feats": wilds_dict["feats"][wilds_dict["locs"] == loc],
            "labels": wilds_dict["labels"][wilds_dict["locs"] == loc],
        }
        for loc in unique_locs
    }


def create_amazon_source_pool(amazon_raw_dict, sample_size = 500):
    pool = {}
    for category, (feats, labels, splits) in amazon_raw_dict.items():
        train_mask = splits == "train"
        f_train, l_train = feats[train_mask], labels[train_mask]
        n = min(len(f_train), sample_size)
        idx = np.random.choice(len(f_train), n, replace=False)
        pool[category] = {
            "feats": torch.from_numpy(f_train[idx]).float(),
            "labels": torch.from_numpy(l_train[idx]).long(),
        }
    print(f"Amazon source pool: {len(pool)} categories")
    return pool
