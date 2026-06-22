from __future__ import annotations

import time
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from context import ExperimentContext
from metrics import get_profit


def create_subset_membership_dataframe(source_list, size):
    unique_rows: dict = {}
    cols = [str(x) for x in source_list]
    df = pd.DataFrame(columns=cols)

    for _ in tqdm(range(size), desc="DSDM: generating subsets"):
        row = np.random.randint(2, size=len(source_list))
        key = str(row)
        while key in unique_rows or not row.any():
            row = np.random.randint(2, size=len(source_list))
            key = str(row)
        unique_rows[key] = 1
        df.loc[len(df)] = row

    return df


def get_dsdm_result(ctx, source_list, trainsize):
    ctx.covered = {}
    start = time.time()

    subsets_df = create_subset_membership_dataframe(source_list, trainsize)
    y_col = []

    for i in tqdm(range(len(subsets_df)), desc="DSDM: evaluating subsets"):
        x = np.array(list(subsets_df.iloc[i]))
        subset = [source_list[p] for p in np.where(x == 1)[0]]
        y_col.append(get_profit(ctx, subset))

    subsets_df["profit"] = y_col
    X = subsets_df.drop(columns=["profit"])
    y = subsets_df["profit"]

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform([[v] for v in y])
    y_scaled = [v[0] for v in y_scaled]

    reg = LinearRegression().fit(X, y_scaled)
    ranked = [source_list[f] for f in np.argsort(reg.coef_)]
    ranked.reverse()

    best_subset: List = []
    subset: List = []
    maxprofit = -10_000_000

    for src in ranked:
        subset = subset + [src]
        p = get_profit(ctx, subset)
        if p > maxprofit:
            maxprofit = p
            best_subset = subset.copy()

    ctx.algo_over = True
    elapsed = time.time() - start
    return get_profit(ctx, best_subset), best_subset, 0, 0, elapsed, ctx.covered_count()
