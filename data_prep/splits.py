from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_SEED = 12345
VAL_FRAC = 0.30
CACHE_DIR = Path("target_splits")
MIN_GROUP_WARN = 20


def _fingerprint(y, sens=None) -> str:
    h = hashlib.sha1(np.asarray(y).tobytes())
    if sens is not None:
        h.update(np.asarray(sens).tobytes())
    return h.hexdigest()[:16]


def build_strata(y, sens=None, is_regression=False, n_bins=4):
    y = np.asarray(y)
    if is_regression:
        edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
        lab = np.digitize(y, edges)
    else:
        lab = y.astype(np.int64)
    if sens is None:
        return lab
    return lab * 1000 + np.asarray(sens).astype(np.int64)


def make_val_test_idx(strata, val_frac=VAL_FRAC, seed=SPLIT_SEED):
    strata = np.asarray(strata)
    n = len(strata)
    rng = np.random.default_rng(seed)
    val_parts = []
    for st in np.unique(strata):
        idx = np.where(strata == st)[0]
        rng.shuffle(idx)
        if len(idx) < 2:
            continue
        k = int(round(val_frac * len(idx)))
        k = min(max(k, 1), len(idx) - 1)
        val_parts.append(idx[:k])
    val_idx = (np.sort(np.concatenate(val_parts)) if val_parts
               else np.array([], dtype=np.int64))
    mask = np.zeros(n, dtype=bool)
    mask[val_idx] = True
    return val_idx, np.where(~mask)[0]


def describe(name, y, val_idx, test_idx, sens=None, is_regression=False):
    y = np.asarray(y)
    if is_regression:
        return
    if sens is not None:
        sens = np.asarray(sens)
        for g in np.unique(sens):
            nv = int((sens[val_idx] == g).sum())
            npos = int(((sens[val_idx] == g) & (y[val_idx] == 1)).sum())
            flag = "  <-- SMALL, TPR-parity will be noisy" \
                if npos < MIN_GROUP_WARN else ""


def get_split(name, y, sens=None, is_regression=False,
              val_frac=VAL_FRAC, seed=SPLIT_SEED, cache_dir=CACHE_DIR,
              verbose=True):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    tag = "" if (seed == SPLIT_SEED and val_frac == VAL_FRAC) \
        else f"__seed{seed}_val{val_frac:g}"
    path = cache_dir / f"{name}{tag}.json"
    fp = _fingerprint(y, sens)

    if path.exists():
        blob = json.loads(path.read_text())
        if blob.get("fingerprint") == fp and blob.get("val_frac") == val_frac \
            and blob.get("seed") == seed:
            val = np.array(blob["val_idx"], dtype=np.int64)
            test = np.array(blob["test_idx"], dtype=np.int64)
            if verbose:
                pass
            return val, test

    strata = build_strata(y, sens, is_regression)
    val, test = make_val_test_idx(strata, val_frac, seed)
    path.write_text(json.dumps(
        dict(name=name, fingerprint=fp, val_frac=val_frac, seed=seed,
             n=len(np.asarray(y)),
             val_idx=val.tolist(), test_idx=test.tolist()), indent=1))
    if verbose:
        describe(name, y, val, test, sens, is_regression)
    return val, test
