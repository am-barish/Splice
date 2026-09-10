from __future__ import annotations

import random
import time
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from core.context import ExperimentContext
from core.metrics import get_profit, get_gain, get_cost, get_subset_percentile, get_subset_rank


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


def greedy(
    ctx: ExperimentContext, source_list: List, budget: int
) -> Tuple[float, List, int, int, float, int]:
    start = time.time()
    individual_profits = [get_profit(ctx, [s])
                          for s in tqdm(source_list, desc="Greedy",
                                        leave=False)]
    sorted_idx = sorted(range(len(individual_profits)),
                        key=individual_profits.__getitem__, reverse=True)
    subset: List = []
    maxyet = -100_000
    best_subset: List = []

    for i in tqdm(sorted_idx, desc="Greedy", leave=False):
        subset = subset + [source_list[i]]
        p = get_profit(ctx, subset)
        if p > maxyet and len(subset) <= budget:
            maxyet = p
            best_subset = subset.copy()

    ctx.algo_over = True
    elapsed = time.time() - start
    return maxyet, best_subset, 0, 0, elapsed, ctx.covered_count()

def all_sources_gain(ctx, source_list):
    g = get_gain(ctx, source_list)
    return g, source_list, g


def single_source_gain(ctx, source_list):
    gains = [get_gain(ctx, [s])
             for s in tqdm(source_list, desc="Single", leave=False)]
    g, s = float(np.max(gains)), [source_list[np.argmax(gains)]]
    return g, s, g
