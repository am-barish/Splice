from __future__ import annotations

import time
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

from context import ExperimentContext
from metrics import get_profit, get_grad_profit, get_marginal_profit_dp


def _marginal(ctx, s1, s2):
    if ctx.method == "dp":
        return get_marginal_profit_dp(ctx, s1, s2)
    if ctx.method == "gradmatch":
        return get_grad_profit(ctx, s1, s2)
    if ctx.method == "hybrid":
        return 0.5 * get_marginal_profit_dp(ctx, s1, s2) + 0.5 * get_grad_profit(ctx, s1, s2)
    return get_profit(ctx, s2) - get_profit(ctx, s1)


def splicing(ctx, initial_set, source_list, kmax):
    maxprofit = get_profit(ctx, initial_set)
    if ctx.covered_count() > ctx.limit:
        return initial_set, maxprofit
    best_subset = initial_set.copy()
    inactive_set = list(set(source_list) - set(initial_set))

    for k_local in range(1, kmax + 1):
        rm_val = {}
        for src in initial_set:
            rm_set = [s for s in initial_set if s != src]
            rm_val[src] = -_marginal(ctx, initial_set, rm_set)

        rm_sorted = sorted(rm_val, key=rm_val.__getitem__)
        active_set = initial_set.copy()
        for i in range(k_local):
            if i < len(inactive_set) and i < len(initial_set):
                active_set.remove(rm_sorted[i])

        add_val = {}
        for src in inactive_set:
            add_set = active_set + [src]
            add_val[src] = _marginal(ctx, active_set, add_set)

        add_sorted = sorted(add_val, key=add_val.__getitem__, reverse=True)

        local_inactive = inactive_set.copy()
        for i in range(k_local):
            if i < len(inactive_set) and i < len(initial_set):
                local_inactive.remove(add_sorted[i])
                active_set = active_set + [add_sorted[i]]
                local_inactive = local_inactive + [rm_sorted[i]]
        curr_profit = get_profit(ctx, active_set)
        if curr_profit > maxprofit:
            best_subset = active_set.copy()
            maxprofit = curr_profit
        if curr_profit > ctx.profits[-1][0]:
            ctx.profits.append([curr_profit, active_set, ctx.covered_count()])

    return best_subset, maxprofit


def fixed_support(ctx, active_set, source_list):
    previous_set = []
    current_set = active_set.copy()

    while set(current_set) != set(previous_set):
        previous_set = current_set.copy()
        kmax = ctx.k_max if len(previous_set) > ctx.k_max else len(previous_set)
        current_set, _ = splicing(ctx, previous_set, source_list, kmax)
    profit = get_profit(ctx, current_set)
    if profit > ctx.profits[-1][0]:
        ctx.profits.append([profit, current_set, ctx.covered_count()])
    return current_set, profit


def get_profits(ctx, source_list, i):
    start = time.time()
    if not ctx.random_gene:
        active_set = sorted(
            source_list, reverse=True,
            key=lambda x: (get_profit(ctx, [x]), x)
        )[:i]
    else:
        active_set = source_list[:i]

    subset, profit = fixed_support(ctx, active_set, source_list)
    print(f"Time {i}: {time.time() - start:.2f}s")
    return subset, profit


def get_best_subset(ctx, source_list, smax, parallel = False, n_jobs = -1):
    if not ctx.random_gene:
        ctx.covered = {}

    start_time = time.time()
    maxprofit = -100_000_000
    best_subset = []
    if parallel:
        import copy
        import threading
        from joblib import Parallel, delayed

        def _worker(i):
            worker_ctx = copy.copy(ctx)

            worker_ctx.profits = [[0, 0, 0]]
            worker_ctx.covered = {}
            return get_profits(worker_ctx, source_list, i)

        with Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator_unordered") as parallel_pool:
            results_gen = parallel_pool(
                delayed(_worker)(i)
                for i in range(1, smax + 1)
            )

            for subset, profit in tqdm(results_gen, total=smax, desc="Splice (Parallel)"):
                if profit > maxprofit:
                    maxprofit = profit
                    best_subset = subset.copy()
    else:
        results = [
            get_profits(ctx, source_list, i)
            for i in tqdm(range(1, smax + 1), desc="Splice")
        ]
        for subset, profit in results:
            if profit > maxprofit:
                maxprofit = profit
                best_subset = subset.copy()

    ctx.algo_over = True
    time_elapsed = time.time() - start_time
    models_explored = ctx.covered_count()

    return get_profit(ctx, best_subset), best_subset, time_elapsed, models_explored, ctx.profits
