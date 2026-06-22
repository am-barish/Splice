
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from context import ExperimentContext
from metrics import get_profit, get_gain, get_cost


def _remove_smallest(profit_list, selection):
    idx = profit_list.index(min(profit_list))
    del profit_list[idx]
    del selection[idx]
    return profit_list, selection


def _rank(profit, profit_list):
    return 1 + sum(1 for x in profit_list if profit < x)


def construction(ctx, source_list, selected, max_profit, k):
    optimal = selected.copy()
    remaining = [x for x in source_list if x not in selected]

    for _ in range(len(remaining)):
        best_sel = []
        profit_list = []
        candidates = [x for x in source_list if x not in selected]

        for src in candidates:
            gain = (get_gain(ctx, selected + [src]) - get_gain(ctx, selected)) - get_cost(ctx, [src])
            if ctx.covered_count() > ctx.limit:
                break
            rank = _rank(gain, profit_list)
            if rank <= k:
                best_sel.append(src)
                profit_list.append(gain)
                if len(profit_list) > k:
                    profit_list, best_sel = _remove_smallest(profit_list, best_sel)

        if not profit_list:
            break

        chosen = best_sel[np.random.randint(len(profit_list))]
        selected.append(chosen)

        curr = get_profit(ctx, selected)
        if curr > ctx.profits[-1][0]:
            ctx.profits.append([curr, selected, ctx.covered_count()])
        if curr > max_profit:
            max_profit = curr
            optimal = selected.copy()

        if ctx.covered_count() > ctx.limit:
            return optimal, get_profit(ctx, optimal)

    return optimal, get_profit(ctx, optimal)


def local_search(ctx, source_list, selected, max_profit, k):
    changed = True
    while changed:
        changed = False
        for src in selected:
            without = [s for s in selected if s != src]
            sl_without = [s for s in source_list if s != src]
            new_sel, new_profit = construction(ctx, sl_without, without,
                                               get_profit(ctx, without), k)
            if new_profit > max_profit:
                selected = new_sel.copy()
                max_profit = new_profit
                changed = True
                break
    return selected, max_profit


def grasp(ctx, source_list, num_repetitions, k):
    ctx.covered = {}
    start = time.time()
    optimal = []
    max_profit = -1_000_000

    for _ in tqdm(range(num_repetitions), desc="GRASP"):
        sel, profit = construction(ctx, source_list, [], -100_000, k)
        sel, profit = local_search(ctx, source_list, sel, profit, k)
        if profit > max_profit:
            optimal = sel
            max_profit = profit

    ctx.algo_over = True
    elapsed = time.time() - start

    if get_profit(ctx, optimal) > ctx.profits[-1][0]:
        ctx.profits.append([get_profit(ctx, optimal), optimal, ctx.covered_count()])

    return get_profit(ctx, optimal), optimal, 0, 0, elapsed, ctx.covered_count(), ctx.profits
