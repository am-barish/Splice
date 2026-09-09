from __future__ import annotations

import numpy as np


_ORIG = {}
_CFG = {"epochs": 300}


def patch_amazon(epochs=300, lr=1e-2, n_avg=1, dropout_eval=0.0,
                 cosine=True):
    import torch
    import torch.nn as nn
    from data_prep import amazon as az
    if "train_binary" not in _ORIG:
        _ORIG["train_binary"] = az.train_binary
        _ORIG["build_head_binary"] = az.build_head_binary

    def build_head_ext(feat_dim=None, device="cpu"):
        head = _ORIG["build_head_binary"](feat_dim, device)
        if dropout_eval is not None:
            for m in head.modules():
                if isinstance(m, nn.Dropout):
                    m.p = dropout_eval
        return head

    def train_binary_ext(head, feats_list, labels_list, epochs=None,
                         lr=lr, weight_balanced=True):


        epochs = _CFG["epochs"]
        device = next(head.parameters()).device
        X = torch.tensor(np.vstack(feats_list)).float().to(device)
        y = torch.tensor(np.hstack(labels_list)).float().unsqueeze(1).to(device)

        if weight_balanced:
            pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)]).to(device)
            crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            crit = nn.BCEWithLogitsLoss()

        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                 if cosine else None)

        head.train()
        for _ in range(epochs):
            loss = crit(head(X), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        return head

    _CFG["epochs"] = int(epochs)
    az.build_head_binary = build_head_ext
    az.train_binary = train_binary_ext
    _install_averaging(n_avg)


def _install_averaging(n_avg):
    from data_prep import amazon as az
    if n_avg <= 1:
        return
    if "get_subset_score" not in _ORIG:
        _ORIG["get_subset_score"] = az.get_subset_score
    base = _ORIG["get_subset_score"]

    def averaged(subset, ctx):


        from core import splice_ext
        base_seed = int(getattr(ctx, "eval_seed", 42))
        prev_tag = getattr(ctx, "_active_eval_config", None)
        vals = []
        try:
            for i in range(n_avg):
                ctx.eval_seed = base_seed * 1000 + i
                ctx._active_eval_config = splice_ext._config_tag(ctx)
                vals.append(base(subset, ctx))
        finally:
            ctx.eval_seed = base_seed
            ctx._active_eval_config = prev_tag
        return float(np.mean(vals))

    az.get_subset_score = averaged


def restore():
    from data_prep import amazon as az
    if not _ORIG:
        return
    for name, fn in _ORIG.items():
        setattr(az, name, fn)


_WORIG = {}


def patch_wilds(vary="both"):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from data_prep import wilds as wn
    if not _WORIG:
        _WORIG["get_balanced_subset"] = wn.get_balanced_subset
        _WORIG["train_on_subset"] = wn.train_on_subset

    if vary == "off":
        for k, v in _WORIG.items():
            setattr(wn, k, v)
        return

    def get_balanced_subset_ext(feats, labels, locs, target_locs, n_per_loc=500):
        if vary not in ("sampling", "both"):
            torch.manual_seed(42)
        idx = []
        for loc in sorted(target_locs):
            loc_idx = torch.where(torch.tensor(locs == loc))[0]
            perm = torch.randperm(len(loc_idx))
            idx.append(loc_idx[perm[:n_per_loc]])
        sel = torch.cat(idx)
        return feats[sel], labels[sel]

    def train_on_subset_ext(loc_indices, source_feats, source_labels,
                            source_locs, num_classes, device, epochs=15):
        if vary not in ("init", "both"):
            torch.manual_seed(42)
        unique_locs = sorted(np.unique(source_locs))
        current = [unique_locs[i] for i in sorted(loc_indices)]
        X_cpu, y_cpu = wn.get_balanced_subset(source_feats, source_labels,
                                              source_locs, current,
                                              n_per_loc=500)
        X, y = X_cpu.to(device), y_cpu.to(device)
        head = nn.Linear(X.shape[1], num_classes).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        head.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                b = perm[i:i + 64]
                loss = F.cross_entropy(head(X[b]), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        return head

    wn.get_balanced_subset = get_balanced_subset_ext
    wn.train_on_subset = train_on_subset_ext


