from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm

from context import ExperimentContext

np.random.seed(42)


class TorchLinReg(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


class TorchLogReg(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def loss_fn(probs, y):
    return F.binary_cross_entropy(probs, y.view(-1, 1).float())


def loss_fn_reg(preds, y):
    return F.mse_loss(preds, y.view(-1, 1).float())


def differentiable_tpr_parity_loss(
    probs, y_true, sensitive_attr, group_0 = 0, group_1 = 1
):
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(sensitive_attr, torch.Tensor):
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)

    probs = probs.view(-1).float()
    y_true = y_true.view(-1).float()
    mask_0 = (sensitive_attr == group_0).float()
    mask_1 = (sensitive_attr == group_1).float()

    eps = 1e-8
    tpr_0 = (probs * y_true * mask_0).sum() / ((y_true * mask_0).sum() + eps)
    tpr_1 = (probs * y_true * mask_1).sum() / ((y_true * mask_1).sum() + eps)
    return torch.abs(tpr_0 - tpr_1)


def loss_fairness(probs, y, sensitive_attr):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(sensitive_attr, torch.Tensor):
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)
    return differentiable_tpr_parity_loss(probs, y, sensitive_attr)


def flatten_grads(model):
    grads = [
        p.grad.detach().flatten()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    return torch.cat(grads)


def compute_val_grad(ctx, model, X_val, y_val, X_val_fair, y_val_fair, sensitive_attr):
    model.zero_grad()
    probs = model(X_val)
    loss = loss_fn(probs, y_val) if not ctx.is_regression else loss_fn_reg(probs, y_val)
    loss.backward()
    g_val_acc = flatten_grads(model)
    model.zero_grad()

    if ctx.is_fairness_dataset:
        probs = model(X_val_fair)
        loss_fair = loss_fairness(probs, y_val_fair, sensitive_attr)
        loss_fair.backward()
        g_val_fair = flatten_grads(model)
        model.zero_grad()
    else:
        g_val_fair = []

    return g_val_acc, g_val_fair


def _compute_source_grad_single(model, X_s, y_s, sensitive_attr, is_regression, is_fairness):
    import copy as _copy
    m = _copy.deepcopy(model)
    X_t = torch.from_numpy(X_s).float()
    y_t = torch.from_numpy(y_s.to_numpy()).float()

    m.zero_grad()
    probs = m(X_t)
    loss = loss_fn(probs, y_t) if not is_regression else loss_fn_reg(probs, y_t)
    loss.backward()
    g_acc = flatten_grads(m)
    m.zero_grad()

    if is_fairness:
        probs = m(X_t)
        lf = loss_fairness(probs, y_t, sensitive_attr)
        lf.backward()
        g_fair = flatten_grads(m)
        m.zero_grad()
    else:
        g_fair = []

    return g_acc, g_fair


def create_balanced_validation(X_all_train, y_all_train, s_all_train, n_samples = 5000, target_balance = 0.5, seed = 42):
    np.random.seed(seed)
    n_per = n_samples // 2
    n_pos = int(n_per * target_balance)
    n_neg = n_per - n_pos

    def _pick(mask, n):
        idx = np.where(mask)[0]
        return np.random.choice(idx, size=min(n, len(idx)), replace=False)

    return np.concatenate([
        _pick((s_all_train == 1.0) & (y_all_train == False), n_neg),
        _pick((s_all_train == 1.0) & (y_all_train == True),  n_pos),
        _pick((s_all_train == 2.0) & (y_all_train == False), n_neg),
        _pick((s_all_train == 2.0) & (y_all_train == True),  n_pos),
    ])


def calculate_gradients(ctx, train_sources, val_source, val_source_fair, test_source, scaler, datasets, metric, n_jobs = -1):
    g_sources = {}
    g_val = {}

    for x in datasets:
        ctx.dataset = x
        N = len(train_sources[x])
        target = ctx.target_col
        sens = ctx.sensitive_col

        X_s, y_s, sens_s = [], [], []
        for i in range(N):
            y_s.append(train_sources[x][i][target])
            sens_s.append(train_sources[x][i][sens] if sens else [])
            t_x = train_sources[x][i].drop(
                columns=["Year", "State", target, "Source ID"]
            )
            X_s.append(scaler[x].transform(t_x))

        y_val = val_source[x][target]
        X_val = scaler[x].transform(
            val_source[x].drop(columns=["Year", "State", target])
        )
        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val.to_numpy()).float()

        if ctx.is_fairness_dataset and sens:
            y_val_fair = val_source_fair[x][target]
            s_fair = val_source_fair[x][sens]
            X_val_fair = scaler[x].transform(
                val_source_fair[x].drop(columns=["Year", "State", target])
            )
            X_val_t_fair = torch.from_numpy(X_val_fair).float()
            y_val_t_fair = torch.from_numpy(y_val_fair.to_numpy()).float()
            s_attr_fair = torch.from_numpy(s_fair.to_numpy()).float()
        else:
            X_val_t_fair = y_val_t_fair = s_attr_fair = []

        model = TorchLogReg(X_val_t.shape[1]) if not ctx.is_regression \
            else TorchLinReg(X_val_t.shape[1])

        g_val[x] = compute_val_grad(
            ctx, model, X_val_t, y_val_t, X_val_t_fair, y_val_t_fair, s_attr_fair
        )

        if n_jobs == 1:
            raw = [
                _compute_source_grad_single(
                    model, X_s[i], y_s[i], sens_s[i],
                    ctx.is_regression, ctx.is_fairness_dataset
                )
                for i in tqdm(range(len(X_s)), desc=f"Source grads [{x}]")
            ]
        else:
            raw = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_compute_source_grad_single)(
                    model, X_s[i], y_s[i], sens_s[i],
                    ctx.is_regression, ctx.is_fairness_dataset
                )
                for i in tqdm(range(len(X_s)), desc=f"Source grads [{x}] (parallel)")
            )

        g_sources[x] = []
        for i, (g_acc, g_fair) in enumerate(raw):
            n = len(X_s[i])
            if ctx.is_fairness_dataset and not ctx.is_regression:
                g_sources[x].append((g_acc / n, g_fair / n))
            else:
                g_sources[x].append((g_acc / n, g_acc / n))

    return g_sources, g_val
