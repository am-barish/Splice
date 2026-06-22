from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error
from tqdm import tqdm

from context import ExperimentContext


def _pool_acs(ctx, source_list):
    dataset = ctx.dataset
    if not ctx.source_X or dataset not in ctx.source_X:
        raise RuntimeError(
            f"ctx.source_X not populated for '{dataset}'."
        )
    X = np.vstack([ctx.source_X[dataset][s] for s in source_list]).astype(np.float32)
    y = np.concatenate([ctx.source_y[dataset][s] for s in source_list])
    return X, y


def _pool_wilds(ctx, source_list):
    pool = ctx.source_pool.get("Wilds", {})
    if not pool:
        raise RuntimeError(
            "ctx.source_pool['Wilds'] is empty."
        )
    import wilds_new as wn
    unique_locs = np.unique(wn.s_locs)
    X_parts, y_parts = [], []
    for idx in source_list:
        loc   = unique_locs[idx]
        entry = pool[loc]
        feats  = entry["feats"]
        labels = entry["labels"]
        X_parts.append(feats.numpy()  if isinstance(feats,  torch.Tensor) else feats)
        y_parts.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    return (
        np.vstack(X_parts).astype(np.float32),
        np.concatenate(y_parts),
    )


def _pool_amazon(ctx: ExperimentContext, source_list: List[int]):
    pool = ctx.source_pool.get("Amazon", {})
    if not pool:
        raise RuntimeError(
            "ctx.source_pool['Amazon'] is empty. "
        )
    cats = ctx.amazon_source_cats
    X_parts, y_parts = [], []
    for idx in source_list:
        cat    = cats[idx]
        feats  = pool[cat]["feats"]
        labels = pool[cat]["labels"]
        X_parts.append(feats.numpy()  if isinstance(feats,  torch.Tensor) else feats)
        y_parts.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    return (
        np.vstack(X_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int64),
    )


def pool_sources(ctx, source_list, dataset_type):
    if dataset_type == "acs":
        return _pool_acs(ctx, source_list)
    elif dataset_type == "wilds":
        return _pool_wilds(ctx, source_list)
    elif dataset_type == "amazon":
        return _pool_amazon(ctx, source_list)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def _eval_acs(ctx, X_train, y_train):
    from metrics import compute_accuracy

    dataset = ctx.dataset
    X_test  = ctx.test_X_scaled[dataset]
    y_test  = ctx.test_y[dataset]
    lam     = ctx.gain_lambda

    if ctx.is_regression:
        reg    = LinearRegression().fit(X_train, y_train.astype(np.float32))
        y_pred = reg.predict(X_test)
        return float(-mean_squared_error(y_test, y_pred))

    clf = LogisticRegression(solver="lbfgs", max_iter=100, tol=1e-4,
                             random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    y_pred   = clf.predict(X_test)
    accuracy = compute_accuracy(y_test, y_pred) * 100

    masks = ctx.test_sens_masks.get(dataset, {})

    if not masks and dataset in ctx.test_source:
        test_df  = ctx.test_source[dataset]
        sens_col = {"ACSIncome": "SEX", "ACSPublicCoverage": "RAC1P",
                    "Scaled_Pubcov": "RAC1P"}.get(dataset)
        if sens_col and sens_col in test_df.columns:
            sens_vals = test_df[sens_col].values
            masks = {int(g): (sens_vals == g) for g in np.unique(sens_vals)}

    if not masks:
        print(f" No fairness masks for {dataset} ")
    tpr_parity = 0.0
    if masks:
        mp = masks.get(1, np.zeros(len(y_pred), dtype=bool))
        mg = masks.get(2, np.zeros(len(y_pred), dtype=bool))

        def _tpr(pred, true, mask):
            pos = true[mask] == 1
            return float(pred[mask][pos].sum() / pos.sum()) if pos.sum() > 0 else 0.0

        tpr_parity = _tpr(y_pred, y_test, mg) - _tpr(y_pred, y_test, mp)
        if mp.sum() > 0 and mg.sum() > 0:
            priv_pred_pos   = float(y_pred[mp].mean())
            unpriv_pred_pos = float(y_pred[mg].mean())
            priv_true_pos   = float((y_test[mp] == 1).mean())
            unpriv_true_pos = float((y_test[mg] == 1).mean())
            print(f"Predicted positive rate: "
                f"privileged={priv_pred_pos:.3f} (true={priv_true_pos:.3f}), "
                f"unprivileged={unpriv_pred_pos:.3f} (true={unpriv_true_pos:.3f})")
    profit = accuracy + lam * tpr_parity
    print(f" acc={accuracy:.3f}  tpr_parity={tpr_parity:.4f}  "
          f"lam={lam}  profit={profit:.4f}")
    return profit


def _eval_wilds(ctx, X_train, y_train):
    import wilds_new as wn

    device    = wn.device
    n_classes = ctx.num_classes
    torch.manual_seed(42)

    n_per_class = 500
    X_parts, y_parts = [], []
    rng = np.random.default_rng(42)
    for cls in range(n_classes):
        mask = (y_train == cls)
        if mask.sum() == 0:
            continue
        idx    = np.where(mask)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        X_parts.append(X_train[chosen])
        y_parts.append(y_train[chosen])

    if not X_parts:
        return 0.0

    X_bal = torch.tensor(np.vstack(X_parts), dtype=torch.float32).to(device)
    y_bal = torch.tensor(np.concatenate(y_parts), dtype=torch.long).to(device)

    head = nn.Linear(X_bal.shape[1], n_classes).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3)
    head.train()
    for _ in range(15):
        perm = torch.randperm(len(X_bal))
        for i in range(0, len(X_bal), 64):
            idx  = perm[i:i+64]
            loss = F.cross_entropy(head(X_bal[idx]), y_bal[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    t_dict   = ctx.test_dict["Wilds"]
    t_feats  = t_dict["feats"]
    t_labels = t_dict["labels"]
    if isinstance(t_labels, torch.Tensor):
        t_labels = t_labels.numpy()
    t_feats  = (t_feats.to(device) if isinstance(t_feats, torch.Tensor)
                else torch.tensor(t_feats).to(device))

    head.eval()
    with torch.no_grad():
        preds = head(t_feats).argmax(1).cpu().numpy()

    return float(f1_score(t_labels, preds, average="macro", zero_division=0))


def _eval_amazon(ctx, X_train, y_train):
    from amazon import build_head_binary, train_binary, evaluate_binary
    head = build_head_binary(feat_dim=X_train.shape[1], device="cpu")
    train_binary(head, [X_train], [y_train.astype(np.float32)], epochs=15)
    return float(evaluate_binary(head, ctx.amazon_target_sources))


def eval_coreset(ctx, X_train, y_train, dataset_type):
    if dataset_type == "acs":
        return _eval_acs(ctx, X_train, y_train)
    elif dataset_type == "wilds":
        return _eval_wilds(ctx, X_train, y_train)
    elif dataset_type == "amazon":
        return _eval_amazon(ctx, X_train, y_train)
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")


def compute_budget(ctx, splice_subset, dataset_type,
                   budget_frac=0.05, budget_mode="fraction"):
    if dataset_type == "acs":
        dataset   = ctx.dataset
        all_rows  = {s: len(v) for s, v in ctx.source_X[dataset].items()}
        n_total   = sum(all_rows.values())
        n_sources = len(all_rows)
    elif dataset_type == "wilds":
        import wilds_new as wn
        pool      = ctx.source_pool.get("Wilds", {})
        n_total   = sum(len(v["feats"]) for v in pool.values())
        n_sources = len(pool)
    elif dataset_type == "amazon":
        pool      = ctx.source_pool.get("Amazon", {})
        n_total   = sum(len(v["feats"]) for v in pool.values())
        n_sources = len(pool)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    if budget_mode == "fraction":
        return max(100, int(budget_frac * n_total))
    elif budget_mode == "splice_sources":
        return max(100, int((len(splice_subset) / max(n_sources, 1)) * n_total))
    elif budget_mode == "splice_rows":
        if dataset_type == "acs":
            dataset = ctx.dataset
            return int(sum(len(ctx.source_X[dataset][s]) for s in splice_subset
                           if s in ctx.source_X.get(dataset, {})))
        elif dataset_type == "wilds":
            import wilds_new as wn
            unique_locs = np.unique(wn.s_locs)
            return int(sum(len(pool[unique_locs[idx]]["feats"])
                           for idx in splice_subset
                           if idx < len(unique_locs) and unique_locs[idx] in pool))
        elif dataset_type == "amazon":
            cats = ctx.amazon_source_cats
            return int(sum(len(pool[cats[i]]["feats"]) for i in splice_subset
                           if i < len(cats) and cats[i] in pool))
    raise ValueError(f"Unknown budget_mode: {budget_mode!r}")


def random_coreset(ctx, source_list, budget, dataset_type="acs",
                   n_runs=5, seed=42):
    t0 = time.time()
    X_pool, y_pool = pool_sources(ctx, source_list, dataset_type)
    budget = min(budget, len(X_pool))

    profits = []
    for run in tqdm(range(n_runs), desc="Random runs", unit="run", leave=False):
        rng = np.random.default_rng(seed + run)
        idx = rng.choice(len(X_pool), size=budget, replace=False)
        profits.append(eval_coreset(ctx, X_pool[idx], y_pool[idx], dataset_type))

    profit  = float(np.mean(profits))
    elapsed = time.time() - t0
    print(f"profit={profit:.4f}  n_runs={n_runs}  time={elapsed:.1f}s")
    return dict(profit=profit, subset_size=budget, time_elapsed=elapsed,
                method="random_coreset", pool_size=len(X_pool))


def _k_center_indices(X, budget, seed=42):
    np.random.seed(seed)
    n         = len(X)
    budget    = min(budget, n)
    X         = X.astype(np.float32)
    X_norm_sq = (X * X).sum(axis=1)
    selected  = [int(np.random.randint(n))]
    min_dists = np.full(n, np.inf, dtype=np.float32)

    for _ in tqdm(range(budget - 1), desc="k-Center", unit="pt", leave=False):
        c        = X[selected[-1]]
        c_sq     = float(np.dot(c, c))
        sq_dists = X_norm_sq - 2.0 * X.dot(c) + c_sq
        np.minimum(min_dists, sq_dists, out=min_dists)
        min_dists[selected] = 0.0
        selected.append(int(np.argmax(min_dists)))

    return np.array(selected, dtype=np.int64)


def k_center_coreset(ctx, source_list, budget, dataset_type="acs", seed=42):
    t0 = time.time()
    X_pool, y_pool = pool_sources(ctx, source_list, dataset_type)
    budget = min(budget, len(X_pool))
    print(f"pool={len(X_pool):,}  budget={budget:,}  dataset={ctx.dataset}")

    idx    = _k_center_indices(X_pool, budget, seed=seed)
    profit = eval_coreset(ctx, X_pool[idx], y_pool[idx], dataset_type)
    elapsed = time.time() - t0

    print(f"profit={profit:.4f}  time={elapsed:.1f}s")
    return dict(profit=profit, subset_size=budget, time_elapsed=elapsed,
                method="k_center_coreset", pool_size=len(X_pool))


def _flat_grad(model):
    return torch.cat([p.grad.detach().flatten()
                      for p in model.parameters()
                      if p.requires_grad and p.grad is not None])


class _ACSLogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class _ACSLinReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        return self.fc(x)

def _warmstart_acs_model(model, X_pool, y_pool, is_regression,
                          epochs=5, batch_size=256, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if is_regression:
        n_warm = min(len(X_pool), 20000)
        idx_warm = rng.choice(len(X_pool), size=n_warm, replace=False)
        X_w = torch.from_numpy(X_pool[idx_warm]).float()
        y_w = torch.from_numpy(y_pool[idx_warm].astype(np.float32))
    else:
        classes = np.unique(y_pool)
        X_parts, y_parts = [], []
        for cls in classes:
            mask = (y_pool == cls)
            idx = np.where(mask)[0]
            chosen = rng.choice(idx, size=min(5000, len(idx)), replace=False)
            X_parts.append(X_pool[chosen])
            y_parts.append(y_pool[chosen])
        X_w = torch.from_numpy(np.vstack(X_parts)).float()
        y_w = torch.from_numpy(np.concatenate(y_parts).astype(np.float32))

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    print(f"Warm-starting reference model "
          f"({len(X_w):,} pts, {epochs} epochs, "
          f"{'regression' if is_regression else 'classification'})")

    n = len(X_w)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            pred = model(X_w[idx])
            if is_regression:
                loss = F.mse_loss(pred.squeeze(-1), y_w[idx])
            else:
                loss = F.binary_cross_entropy(pred.squeeze(-1), y_w[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model

def _compute_grads_acs_vectorised(X_pool, y_pool, model, is_reg,
                                   batch_size=65536):
    n = len(X_pool)
    all_grads = []
    with torch.no_grad():
        w = model.fc.weight.data.squeeze(0)
        b = model.fc.bias.data.squeeze(0)
    for start in tqdm(range(0, n, batch_size), desc="  Point grads (vec)",
                      unit="batch", leave=False):
        end = min(start + batch_size, n)
        X   = torch.from_numpy(X_pool[start:end]).float()
        y   = torch.from_numpy(y_pool[start:end].astype(np.float32))
        with torch.no_grad():
            logits    = X @ w + b
            residuals = (2*(logits - y) if is_reg
                         else torch.sigmoid(logits) - y)
            g_w = residuals.unsqueeze(1) * X
            g_b = residuals.unsqueeze(1)
        g = F.normalize(torch.cat([g_w, g_b], dim=1), dim=1)
        all_grads.append(g)
    return torch.cat(all_grads, dim=0)


def _point_grad_acs(x, y, model, is_reg):
    model.zero_grad()
    Xt   = torch.from_numpy(x.reshape(1, -1)).float()
    yt   = torch.tensor([[y]], dtype=torch.float32)
    pred = model(Xt)
    loss = (F.mse_loss(pred, yt) if is_reg
            else F.binary_cross_entropy(pred, yt))
    loss.backward()
    g = F.normalize(_flat_grad(model).detach().clone(), dim=0)
    model.zero_grad()
    return g


def _get_val_grad_acs(ctx, model):
    dataset = ctx.dataset
    Xv = torch.from_numpy(ctx.test_X_scaled[dataset]).float()
    yv = torch.from_numpy(ctx.test_y[dataset].astype(np.float32))

    m = copy.deepcopy(model)
    m.zero_grad()
    pred = m(Xv)
    loss = (F.mse_loss(pred, yv.unsqueeze(1)) if ctx.is_regression
            else F.binary_cross_entropy(pred, yv.unsqueeze(1)))
    loss.backward()
    return F.normalize(_flat_grad(m), dim=0)

def _get_fair_grad_acs(ctx, model):
    dataset = ctx.dataset
    Xv = ctx.test_X_scaled[dataset]
    yv = ctx.test_y[dataset]

    masks = ctx.test_sens_masks.get(dataset, {})
    if not masks and dataset in ctx.test_source:
        sens_col = {"ACSIncome": "SEX", "ACSPublicCoverage": "RAC1P"}.get(dataset)
        if sens_col and sens_col in ctx.test_source[dataset].columns:
            sens_vals = ctx.test_source[dataset][sens_col].values
            masks = {int(g): (sens_vals == g) for g in np.unique(sens_vals)}

    mp = masks.get(1, np.zeros(len(yv), dtype=bool))
    mg = masks.get(2, np.zeros(len(yv), dtype=bool))

    pos_p = mp & (yv == 1)
    pos_g = mg & (yv == 1)

    if pos_p.sum() == 0 or pos_g.sum() == 0:
        return _get_val_grad_acs(ctx, model)

    m = copy.deepcopy(model)
    m.zero_grad()
    Xp = torch.from_numpy(Xv[pos_p]).float()
    Xg = torch.from_numpy(Xv[pos_g]).float()
    yp = torch.ones(int(pos_p.sum()), 1).float()
    yg = torch.ones(int(pos_g.sum()), 1).float()

    pred_p = m(Xp)
    pred_g = m(Xg)
    loss_p = F.binary_cross_entropy(pred_p, yp)
    loss_g = F.binary_cross_entropy(pred_g, yg)
    loss = (loss_g - loss_p)
    loss.backward()
    return F.normalize(_flat_grad(m), dim=0)


def _soft_macro_f1_loss(logits, labels, num_classes, eps=1e-7):
    probs = torch.softmax(logits, dim=1)
    y_oh  = F.one_hot(labels.long(), num_classes).float()
    tp    = (probs * y_oh).sum(0)
    fp    = (probs * (1 - y_oh)).sum(0)
    fn    = ((1 - probs) * y_oh).sum(0)
    f1    = (2 * tp) / (2 * tp + fp + fn + eps)
    return 1.0 - f1.mean()


def _point_grad_wilds(x, y, model, num_classes):
    model.zero_grad()
    Xt   = torch.from_numpy(x.reshape(1, -1)).float()
    yt   = torch.tensor([y], dtype=torch.long)
    loss = _soft_macro_f1_loss(model(Xt), yt, num_classes)
    loss.backward()
    g    = F.normalize(_flat_grad(model).detach().clone(), dim=0)
    model.zero_grad()
    return g


def _warmstart_wilds_model(model, X_pool, y_pool, num_classes,
                           device, epochs=5, batch_size=256):
    rng = np.random.default_rng(0)
    X_parts, y_parts = [], []
    for cls in range(num_classes):
        mask = (y_pool == cls)
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        chosen = rng.choice(idx, size=min(500, len(idx)), replace=False)
        X_parts.append(X_pool[chosen])
        y_parts.append(y_pool[chosen])

    if not X_parts:
        return model

    X_w = torch.tensor(np.vstack(X_parts), dtype=torch.float32).to(device)
    y_w = torch.tensor(np.concatenate(y_parts), dtype=torch.long).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    print(f"Warm-starting reference model "
          f"({len(X_w):,} pts, {epochs} epochs) ...")
    for ep in range(epochs):
        perm = torch.randperm(len(X_w))
        for i in range(0, len(X_w), batch_size):
            idx  = perm[i:i+batch_size]
            loss = F.cross_entropy(model(X_w[idx]), y_w[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def _compute_grads_wilds_lastlayer(X_pool, y_pool, model, num_classes,
                                    batch_size=4096):
    device = next(model.parameters()).device
    n      = len(X_pool)
    all_grads = []
    model.eval()
    for start in tqdm(range(0, n, batch_size),
                      desc="  Wilds grads (last-layer)", unit="batch",
                      leave=False):
        end = min(start + batch_size, n)
        Xb  = torch.from_numpy(X_pool[start:end]).float().to(device)
        yb  = torch.from_numpy(y_pool[start:end].astype(np.int64)).to(device)
        with torch.no_grad():
            logits = model(Xb)
            probs  = torch.softmax(logits, dim=1)
            y_oh   = F.one_hot(yb, num_classes).float().to(device)
            delta  = probs - y_oh
        all_grads.append(F.normalize(delta, dim=1).cpu())
    return torch.cat(all_grads, dim=0)


def _get_val_grad_wilds_lastlayer(ctx, model):
    n_cls  = ctx.num_classes
    g = ctx.g_val.get("Wilds")
    if g is not None:
        g_t = g.float() if isinstance(g, torch.Tensor) else torch.tensor(g).float()
        g_t = g_t.cpu()
        if g_t.shape[0] >= n_cls:
            return F.normalize(g_t[-n_cls:], dim=0)
        return F.normalize(g_t, dim=0)

    device   = next(model.parameters()).device
    t_dict   = ctx.test_dict["Wilds"]
    t_feats  = t_dict["feats"].float().to(device)
    t_labels = t_dict["labels"]
    if isinstance(t_labels, torch.Tensor):
        t_labels = t_labels.long().to(device)
    else:
        t_labels = torch.tensor(t_labels).long().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(t_feats)
        probs  = torch.softmax(logits, dim=1)
        y_oh   = F.one_hot(t_labels, n_cls).float().to(device)
        delta  = (probs - y_oh).mean(0)
    return F.normalize(delta, dim=0).cpu()


def _get_val_grad_wilds(ctx):
    g = ctx.g_val.get("Wilds")
    if g is None:
        raise RuntimeError(
            "ctx.g_val['Wilds'] is None. "
        )
    return F.normalize(g.float(), dim=0) if isinstance(g, torch.Tensor) else F.normalize(torch.tensor(g).float(), dim=0)


def _build_amazon_ref_model(d):
    return nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

def _warmstart_amazon_model(model, X_pool, y_pool,
                             epochs=5, batch_size=256, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    classes = np.unique(y_pool)
    X_parts, y_parts = [], []
    for cls in classes:
        mask = (y_pool == cls)
        idx = np.where(mask)[0]
        chosen = rng.choice(idx, size=min(5000, len(idx)), replace=False)
        X_parts.append(X_pool[chosen])
        y_parts.append(y_pool[chosen])

    X_w = torch.from_numpy(np.vstack(X_parts)).float()
    y_w = torch.from_numpy(np.concatenate(y_parts).astype(np.float32))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    print(f"Warm-starting reference model "
          f"({len(X_w):,} pts, {epochs} epochs) ...")

    n = len(X_w)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            logits = model(X_w[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_w[idx])
            loss.backward()
            opt.step()
        print(f"epoch {ep+1} loss: {loss.item():.4f}")
    model.eval()
    return model

def _point_grad_amazon(x, y, model):
    model.train()
    model.zero_grad()
    Xt   = torch.from_numpy(x.reshape(1, -1)).float()
    yt   = torch.tensor([[float(y)]]).float()
    pred = torch.sigmoid(model(Xt))
    F.binary_cross_entropy(pred, yt).backward()
    final_layer = None
    for layer in model:
        if isinstance(layer, nn.Linear):
            final_layer = layer
    g = final_layer.weight.grad.flatten().detach().clone()
    model.zero_grad()
    return F.normalize(g, dim=0)


def _compute_grads_amazon_lastlayer(X_pool, y_pool, model, batch_size=4096):
    n = len(X_pool)
    all_grads = []
    final_layer = None
    for layer in model:
        if isinstance(layer, nn.Linear):
            final_layer = layer

    for start in tqdm(range(0, n, batch_size),
                      desc="  Amazon grads (last-layer)", unit="batch",
                      leave=False):
        end = min(start + batch_size, n)
        Xb  = torch.from_numpy(X_pool[start:end]).float()
        yb  = torch.from_numpy(y_pool[start:end].astype(np.float32))
        with torch.no_grad():
            h = Xb
            for layer in model:
                if layer is final_layer:
                    break
                h = layer(h)
            logits = final_layer(h).squeeze(1)
            delta  = (torch.sigmoid(logits) - yb)
            g_w    = delta.unsqueeze(1) * h
        all_grads.append(F.normalize(g_w, dim=1).cpu())
    return torch.cat(all_grads, dim=0)


def _get_val_grad_amazon(ctx, model):
    from amazon import get_target_split, compute_target_gradient_averaged

    target_val_feats, target_val_labels, _, _ = \
        get_target_split(ctx.amazon_target_sources)
    print(f"val data: n={len(target_val_feats)}, "f"pos_rate={target_val_labels.float().mean().item():.3f}")
    g = compute_target_gradient_averaged(
        target_val_feats, target_val_labels, model,
        device=torch.device("cpu"),
        subset_size=512, n_seeds=5,
    )
    return F.normalize(g.float(), dim=0)


def _omp_select(X_pool, y_pool, g_val, budget,
               point_grad_fn=None, batch=256,
               min_per_class=2, G_precomputed=None):
    n = len(X_pool)
    budget = min(budget, n)

    classes, warm_selected = np.unique(y_pool), []
    rng = np.random.default_rng(42)
    for cls in classes:
        cls_idx = np.where(y_pool == cls)[0]
        warm_selected.extend(
            rng.choice(cls_idx, size=min(min_per_class, len(cls_idx)),
                       replace=False).tolist())
    warm_selected = list(dict.fromkeys(warm_selected))
    omp_budget    = max(0, budget - len(warm_selected))
    remaining     = [i for i in range(n) if i not in set(warm_selected)]

    if G_precomputed is not None:
        G = G_precomputed
    else:
        print(f"Pre-computing {n:,} point gradients ...")
        grads = []
        for i in tqdm(range(0, n, batch), desc="  Point grads",
                      unit="batch", leave=False):
            for j in range(i, min(i + batch, n)):
                grads.append(point_grad_fn(X_pool[j], y_pool[j]))
        G = torch.stack(grads)

    g_val  = g_val.cpu() if isinstance(g_val, torch.Tensor) else g_val
    if G_precomputed is not None:
        G_precomputed = G_precomputed.cpu()

    g_target = F.normalize(g_val, dim=0)
    g_curr   = (F.normalize(torch.stack([G[i] for i in warm_selected]).mean(0), dim=0)
                if warm_selected else torch.zeros_like(g_target))
    selected = list(warm_selected)

    warm_set   = set(warm_selected)
    active_idx = np.array([i for i in range(len(X_pool))
                           if i not in warm_set], dtype=np.int64)
    active_size = len(active_idx)
    G_active    = G[active_idx].clone()

    for step in tqdm(range(min(omp_budget, active_size)), desc="  OMP select",
                     unit="pt", leave=False):
        residual    = F.normalize(g_target - g_curr, dim=0)
        sims        = torch.mv(G_active[:active_size], residual)
        best_local  = int(torch.argmax(sims).item())
        best_glob   = int(active_idx[best_local])

        selected.append(best_glob)
        n_sel  = len(selected)
        g_curr = (g_curr * (n_sel - 1) + G[best_glob]) / n_sel

        active_size -= 1
        active_idx[best_local]    = active_idx[active_size]
        G_active[best_local]      = G_active[active_size]

    return np.array(selected, dtype=np.int64)


def gradmatch_coreset(ctx, source_list, budget, dataset_type="acs",
                      batch=256, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    X_pool, y_pool = pool_sources(ctx, source_list, dataset_type)
    budget = min(budget, len(X_pool))
    print(f"pool={len(X_pool):,}  budget={budget:,}  "
          f"dataset={ctx.dataset}")

    if dataset_type == "acs":
        d     = X_pool.shape[1]
        model = _ACSLinReg(d) if ctx.is_regression else _ACSLogReg(d)

        model = _warmstart_acs_model(
                    model, X_pool, y_pool, ctx.is_regression, epochs=5)

        g_acc = _get_val_grad_acs(ctx, model)

        print(" Vectorised analytical gradient computation ...")
        G_acc = _compute_grads_acs_vectorised(
                    X_pool, y_pool, model, ctx.is_regression)

        if ctx.is_fairness_dataset and ctx.gain_lambda > 0:
            g_fair = _get_fair_grad_acs(ctx, model)
            lam_w  = ctx.gain_lambda / 100.0
            g_val  = F.normalize(
                        (1 - lam_w) * g_acc + lam_w * g_fair, dim=0)
        else:
            g_val = g_acc

        G   = G_acc
        idx = _omp_select(X_pool, y_pool, g_val, budget,
                        G_precomputed=G, min_per_class=2)

    elif dataset_type == "wilds":
        import wilds_new as wn
        n_cls  = ctx.num_classes
        device = wn.device
        model  = nn.Linear(X_pool.shape[1], n_cls).to(device)
        model = _warmstart_wilds_model(
                    model, X_pool, y_pool, n_cls, device, epochs=5)
        g_val  = _get_val_grad_wilds_lastlayer(ctx, model)
        G      = _compute_grads_wilds_lastlayer(X_pool, y_pool, model, n_cls)
        idx    = _omp_select(X_pool, y_pool, g_val, budget,
                             G_precomputed=G, min_per_class=2)

    elif dataset_type == "amazon":
        d     = X_pool.shape[1]
        model = _build_amazon_ref_model(d)

        model = _warmstart_amazon_model(model, X_pool, y_pool, epochs=5)

        g_val = _get_val_grad_amazon(ctx, model)

        G     = _compute_grads_amazon_lastlayer(X_pool, y_pool, model)
        idx   = _omp_select(X_pool, y_pool, g_val, budget,
                            G_precomputed=G, min_per_class=2)

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
    print(f"coreset pos rate: {y_pool[idx].mean():.3f}, "
      f"pool pos rate: {y_pool.mean():.3f}")
    profit = eval_coreset(ctx, X_pool[idx], y_pool[idx], dataset_type)
    elapsed = time.time() - t0

    print(f"profit={profit:.4f}  time={elapsed:.1f}s")
    return dict(profit=profit, subset_size=budget, time_elapsed=elapsed,
                method="gradmatch_coreset", pool_size=len(X_pool))


def run_all_coresets(ctx, source_list, splice_profit, splice_subset,
                     dataset_type="acs", batch=256):
    budget = compute_budget(ctx, splice_subset, dataset_type,
                            budget_mode='fraction', budget_frac=0.05 if dataset_type == "wilds" else 0.01)
    if budget == 0:
        raise ValueError(
            "Budget is 0"
        )

    print(f"dataset={ctx.dataset}  type={dataset_type}")
    print(f"Splice selected     : {splice_subset}")
    print(f"Budget (rows)       : {budget:,}")
    print(f"Splice profit       : {splice_profit:.4f}")

    results = {}
    results["random_coreset"]   = random_coreset(ctx, source_list, budget, dataset_type)
    results["gradmatch_point"]  = gradmatch_coreset(ctx, source_list, budget, dataset_type, batch=batch)

    return results
