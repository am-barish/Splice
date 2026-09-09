from __future__ import annotations

import numpy as np

POOL_SEED = 20240


def strata_key(y, sens=None, is_regression=False, n_bins=5):
    y = np.asarray(y)
    if is_regression:
        edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
        key = np.digitize(y, edges).astype(np.int64)
    else:
        _, key = np.unique(y, return_inverse=True)
        key = key.astype(np.int64)
    if sens is not None:
        s = np.asarray(sens)
        _, s = np.unique(s, return_inverse=True)
        key = key * (s.max() + 1) + s.astype(np.int64)
    return key


def fixed_test_split(strata, test_frac=0.30, seed=POOL_SEED):
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    pool, test = [], []
    n_rare_rows = n_rare_strata = 0

    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) < 2:
            test.append(idx)
            n_rare_rows += len(idx)
            n_rare_strata += 1
            continue
        idx = rng.permutation(idx)
        n_te = int(round(test_frac * len(idx)))
        n_te = min(max(n_te, 1), len(idx) - 1)
        test.append(idx[:n_te])
        pool.append(idx[n_te:])

    pool_idx = np.sort(np.concatenate(pool)) if pool else np.array([], int)
    test_idx = np.sort(np.concatenate(test)) if test else np.array([], int)
    assert not (set(pool_idx.tolist()) & set(test_idx.tolist()))
    assert len(pool_idx) + len(test_idx) == len(strata)

    info = dict(n_target=len(strata), n_pool=len(pool_idx), n_test=len(test_idx),
                test_frac_realised=len(test_idx) / len(strata),
                n_rare_strata_to_test=n_rare_strata,
                n_rare_rows_to_test=n_rare_rows,
                n_strata=int(len(np.unique(strata))))
    return pool_idx, test_idx, info


def nested_val_draws(strata, pool_idx, fracs, draw_seed=0):
    strata = np.asarray(strata)
    n_total = len(strata)
    rng = np.random.default_rng(1_000_000 + int(draw_seed))

    order, sizes = {}, {}
    for s in np.unique(strata[pool_idx]):
        members = pool_idx[strata[pool_idx] == s]
        order[s] = rng.permutation(members)
        sizes[s] = len(members)
    n_pool = sum(sizes.values())

    out = {}
    for f in sorted(fracs):
        chunks = []
        for s, members in order.items():
            k = int(round(f * n_total * sizes[s] / n_pool))
            k = min(k, sizes[s])
            if k > 0:
                chunks.append(members[:k])
        out[float(f)] = (np.sort(np.concatenate(chunks)) if chunks
                         else np.array([], dtype=int))

    fs = sorted(out)
    for a, b in zip(fs, fs[1:]):
        assert set(out[a].tolist()) <= set(out[b].tolist()),            f"nesting broken between f={a} and f={b}"
    return out


def draw_report(strata, val_idx, test_idx, n_total, label=""):
    import zlib
    strata = np.asarray(strata)
    sv, st = set(np.unique(strata[val_idx]).tolist()), set(np.unique(strata[test_idx]).tolist())
    return dict(label=label,
                val_hash=format(zlib.crc32(
                    np.ascontiguousarray(np.asarray(val_idx, dtype=np.int64))),
                    "08x"),
                n_val=int(len(val_idx)),
                frac_realised=float(len(val_idx) / n_total),
                strata_in_val=len(sv),
                strata_in_test=len(st),
                strata_in_test_missing_from_val=len(st - sv),
                min_stratum_count_in_val=(int(np.min(np.bincount(
                    np.searchsorted(np.unique(strata[val_idx]), strata[val_idx]))))
                    if len(val_idx) else 0))


def group_positive_counts(y, sens, idx):
    if sens is None or len(idx) == 0:
        return {}
    y, sens = np.asarray(y)[idx], np.asarray(sens)[idx]
    return {int(g): int(((sens == g) & (y == 1)).sum()) for g in np.unique(sens)}
