from __future__ import annotations

import random
import time
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from context import ExperimentContext
from metrics import get_profit, get_gain, get_cost, get_subset_percentile, get_subset_rank


def brute_force(ctx, source_list):
    start = time.time()
    maxyet = -10_000_000
    best_subset = []
    local_subsets= [[]]

    for el in source_list:
        new = []
        for s in local_subsets:
            candidate = s + [el]
            new.append(candidate)
            p = get_profit(ctx, candidate)
            if p > maxyet:
                maxyet = p
                best_subset = candidate
        local_subsets.extend(new)

    ctx.algo_over = True
    elapsed = time.time() - start
    return (
        maxyet, best_subset,
        get_subset_percentile(ctx, best_subset),
        get_subset_rank(ctx, best_subset),
        elapsed, ctx.covered_count()
    )

def random_subset(ctx, source_list):
    start = time.time()
    np.random.seed(42)
    random.seed(42)
    results = [
        get_profit(ctx, random.sample(source_list, random.randint(1, 15)))
        for _ in range(10)
    ]
    ctx.algo_over = True
    elapsed = time.time() - start
    return float(np.mean(results)), [], 0, 0, elapsed, ctx.covered_count()


def forward_greedy(ctx, source_list):
    start = time.time()
    best_subset = []
    best_val = get_profit(ctx, [])
    remaining = list(source_list)

    while remaining:
        best_gain = -float("inf")
        best_source = None
        best_val_candidate = best_val

        for s in remaining:
            val = get_profit(ctx, best_subset + [s])
            gain = val - best_val
            if gain > best_gain:
                best_gain = gain
                best_source = s
                best_val_candidate = val

        if best_gain <= 0:
            break

        best_subset.append(best_source)
        print(f"Added {best_source} | gain={best_gain:.4f} | total={best_val_candidate:.4f}")
        best_val = best_val_candidate
        remaining.remove(best_source)

    ctx.algo_over = True
    elapsed = time.time() - start
    return best_val, best_subset, 0, 0, elapsed, ctx.covered_count()


def all_sources_gain(ctx, source_list):
    g = get_gain(ctx, source_list)
    return g, g, g


def single_source_gain(ctx, source_list):
    gains = [get_gain(ctx, [s]) for s in source_list]
    g = float(np.max(gains))
    return g, g, g
