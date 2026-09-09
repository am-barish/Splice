from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


from core import splice_ext


def val_target_wilds(data_dir):
    d = torch.load(Path(data_dir) / "target_val.pt", weights_only=False,
                   map_location="cpu")
    return {"feats": d["feats"], "labels": d["labels"]}


def val_target_amazon(ctx):
    feats, labels = [], []
    for cat in ctx.amazon_target_cats:
        f, l, splits = ctx.amazon_target_sources[cat]
        m = splits == "val"
        if m.sum() == 0:
            continue
        feats.append(torch.as_tensor(f[m]))
        labels.append(torch.as_tensor(np.asarray(l)[m]))
    if not feats:
        raise RuntimeError(
            "No rows with split == 'val' in the Amazon target categories. "
            "Rebuild reviews_raw.parquet via amazon_review_setup().")
    return {"feats": torch.cat(feats), "labels": torch.cat(labels)}


def memoize_scorer(module, name):
    original = getattr(module, name)
    cache = {}

    def wrapped(subset, ctx):
        key = frozenset(subset)
        if key not in cache:
            cache[key] = original(subset, ctx)
        return cache[key]

    setattr(module, name, wrapped)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["wilds", "amazon"])
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--iters", type=int, default=5,
                    )
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--amazon-epochs", type=int, default=15,
                    )
    ap.add_argument("--amazon-navg", type=int, default=1,
                    )
    args = ap.parse_args()

    splice_ext.install()

    if args.family == "amazon" and args.amazon_epochs > 0:
        from harness import stabilize
        stabilize.patch_amazon(epochs=args.amazon_epochs, lr=5e-3, n_avg=args.amazon_navg)
    from surrogates.dataprofiles_unstructured import generate_meta_history

    out = Path(args.out_dir)
    t0 = time.time()

    if args.family == "wilds":
        from data_prep import wilds as wn
        from core.experiment import setup_wilds_context
        ctx = setup_wilds_context(data_dir=args.data_dir, method="normal")
        target_data = val_target_wilds(args.data_dir)
        cache = memoize_scorer(wn, "get_subset_acc")
        hist_name, prof_name = "history_list_wilds_val.pkl", "Wilds_target_profile_val.pkl"
    else:
        from data_prep import amazon as az
        from core.experiment import setup_amazon_context
        ctx = setup_amazon_context(data_dir=args.data_dir, method="normal")
        target_data = val_target_amazon(ctx)
        cache = memoize_scorer(az, "get_subset_score")
        hist_name, prof_name = "history_list_amazon_val.pkl", "Amazon_target_profile_val.pkl"

    ctx.eval_split = "val"
    if args.family == "amazon":
        pass

    history = generate_meta_history(
        ctx, ctx.source_pool[ctx.dataset], target_data,
        iterations_per_source=args.iters)

    with open(out / hist_name, "wb") as f:
        pickle.dump(history, f)

    profile = ctx.kurt if ctx.dataset == "Wilds" else ctx.prev
    with open(out / prof_name, "wb") as f:
        pickle.dump(profile, f)


if __name__ == "__main__":
    main()
