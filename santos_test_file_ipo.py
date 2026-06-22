import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_DIR     = "ipo/"
TARGET_FILE  = "1004ipopayments.csv"
LR_C         = 0.1
MAX_ITER     = 500
CV_FOLDS     = 5
RANDOM_SEED  = 42
GREEDY_STEPS = 8


def load_ipo(path):
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["expense type", "expense area", "amount"])
    df = df[df["amount"] > 0]
    return (
        df[["expense type", "expense area", "amount"]]
        .rename(columns={
            "expense type": "Expense Type",
            "expense area": "Expense Area",
            "amount":       "Amount",
        })
        .copy()
    )


def load_all_tables(data_dir):
    tables = {}
    patterns = [
        os.path.join(data_dir, "*ipopayments.csv"),
        os.path.join(data_dir, "*IPOpayments.csv"),
        os.path.join(data_dir, "*IPOPayments.csv"),
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            base = os.path.basename(path)
            name = base.lower().replace("ipopayments.csv", "")
            tables[name] = load_ipo(path)
    return tables


def build_encoders(tables):
    ohe_type = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        pd.concat([d["Expense Type"] for d in tables.values()])
        .astype(str).values.reshape(-1, 1)
    )
    ohe_area = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        pd.concat([d["Expense Area"] for d in tables.values()])
        .astype(str).values.reshape(-1, 1)
    )
    return ohe_type, ohe_area


def make_features(df, ohe_type, ohe_area):
    return np.hstack([
        ohe_type.transform(df["Expense Type"].astype(str).values.reshape(-1, 1)),
        ohe_area.transform(df["Expense Area"].astype(str).values.reshape(-1, 1)),
    ])


def make_label(df, threshold):
    return (df["Amount"].values > threshold).astype(int)


def train_eval(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(
        C=LR_C, max_iter=MAX_ITER, random_state=RANDOM_SEED
    )
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_te, clf.predict(X_te), zero_division=0))


def self_train_upper_bound(X_t, y_t):
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    return float(np.mean([
        train_eval(X_t[tr], y_t[tr], X_t[te], y_t[te])
        for tr, te in skf.split(X_t, y_t)
    ]))


def greedy_forward_selection(source_names, src_feats, src_labels, X_t, y_t):
    remaining = list(source_names)
    selected  = []
    trail     = []

    for _ in range(min(GREEDY_STEPS, len(remaining))):
        best_cand, best_f1 = None, 0.0
        for cand in remaining:
            trial = selected + [cand]
            X_tr  = np.vstack([src_feats[n]  for n in trial])
            y_tr  = np.concatenate([src_labels[n] for n in trial])
            f1    = train_eval(X_tr, y_tr, X_t, y_t)
            if f1 > best_f1:
                best_f1, best_cand = f1, cand
        if best_cand is None:
            break
        selected.append(best_cand)
        remaining.remove(best_cand)
        trail.append((best_cand, best_f1, list(selected)))

    return trail

def santos_tables_gain(ctx, indices):
    X_selected  = np.vstack([ctx.xs[i] for i in indices])
    y_selected  = np.concatenate([ctx.ys[i] for i in indices])
    f1_selected = train_eval(X_selected, y_selected, ctx.X_t, ctx.y_t)
    return f1_selected


all_tables  = load_all_tables(DATA_DIR)
target_name = TARGET_FILE.lower().replace("ipopayments.csv", "")
print(target_name)
target_df    = all_tables[target_name]
source_names = [k for k in all_tables if k != target_name]
threshold    = target_df["Amount"].median()
ohe_type, ohe_area = build_encoders(all_tables)
X_t = make_features(target_df, ohe_type, ohe_area)
y_t = make_label(target_df, threshold)

src_feats  = {n: make_features(all_tables[n], ohe_type, ohe_area)
                for n in source_names}
src_labels = {n: make_label(all_tables[n], threshold)
                for n in source_names}

ub = self_train_upper_bound(X_t, y_t)
single_f1 = {
    n: train_eval(src_feats[n], src_labels[n], X_t, y_t)
    for n in source_names
}
X_all  = np.vstack(list(src_feats.values()))
y_all  = np.concatenate(list(src_labels.values()))
f1_all = train_eval(X_all, y_all, X_t, y_t)


from experiment import *
import gc
gc.collect()
ctx = ExperimentContext(dataset="SANTOS_IPO")
ctx.reset_for_run()
ctx.source_list = [i for i in range(19)]
ctx.xs = list(src_feats.values())
ctx.ys = list(src_labels.values())
ctx.X_t = X_t
ctx.y_t = y_t
run_experiment(ctx, algorithm="splice", parallel=False, smax=7, n_jobs=6)
