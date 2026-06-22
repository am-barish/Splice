import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


DATA_DIR      = "ydnpa/"
TARGET_FILE   = "ydnpaspendingdata-october2018.csv"
LR_C          = 0.1
MAX_ITER      = 500
CV_FOLDS      = 5
RANDOM_SEED   = 42
GREEDY_STEPS  = 8


def load_ydnpa(path):
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Expense Type", "Expense Area", "Amount"])
    df = df[df["Amount"] > 0]
    return df[["Expense Type", "Expense Area", "Amount"]].copy()


def load_all_tables(data_dir):
    tables = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "ydnpaspendingdata-*.csv"))):
        name = os.path.basename(path).replace("ydnpaspendingdata-", "").replace(".csv", "")
        tables[name] = load_ydnpa(path)
    return tables


def build_encoders(tables):
    ohe_type = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        pd.concat([d["Expense Type"] for d in tables.values()]).astype(str).values.reshape(-1, 1)
    )
    ohe_area = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        pd.concat([d["Expense Area"] for d in tables.values()]).astype(str).values.reshape(-1, 1)
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
    clf = LogisticRegression(C=LR_C, max_iter=MAX_ITER, random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te), zero_division=0))


def self_train_upper_bound(X_t, y_t):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
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


def main():
    all_tables  = load_all_tables(DATA_DIR)
    target_name = TARGET_FILE.replace("ydnpaspendingdata-", "").replace(".csv", "")
    if target_name not in all_tables:
        raise FileNotFoundError(f"{TARGET_FILE} not found in {DATA_DIR}")

    target_df    = all_tables[target_name]
    source_names = [k for k in all_tables if k != target_name]
    print(f"\n  Loaded {len(all_tables)} tables, {len(source_names)} source tables")

    ohe_type, ohe_area = build_encoders(all_tables)
    threshold = target_df["Amount"].median()

    X_t = make_features(target_df, ohe_type, ohe_area)
    y_t = make_label(target_df, threshold)

    src_feats  = {n: make_features(all_tables[n], ohe_type, ohe_area) for n in source_names}
    src_labels = {n: make_label(all_tables[n], threshold)              for n in source_names}

    ub = self_train_upper_bound(X_t, y_t)
    print(f"\n  Upper bound (self-train {CV_FOLDS}-fold CV) : F1 = {ub:.4f}")

    single_f1 = {
        n: train_eval(src_feats[n], src_labels[n], X_t, y_t)
        for n in source_names
    }

    print(f"  Single-source F1  (all {len(source_names)} SANTOS tables, sorted)")

    for name, f1 in sorted(single_f1.items(), key=lambda x: -x[1]):
        pos  = src_labels[name].mean()
        diff = f1 - ub
        print(f"  {name}, {f1},  {diff},  {pos}")

    X_all  = np.vstack(list(src_feats.values()))
    y_all  = np.concatenate(list(src_labels.values()))
    f1_all = train_eval(X_all, y_all, X_t, y_t)
    print(f"\n  ALL {len(source_names)} tables pooled (full SANTOS output) : "
          f"F1 = {f1_all:.4f}  (n_train = {len(y_all)})")

    print(f"  Greedy forward source selection")
    trail = greedy_forward_selection(
        source_names, src_feats, src_labels, X_t, y_t
    )

    best_single_name  = max(single_f1, key=single_f1.get)
    best_single_f1    = single_f1[best_single_name]
    worst_single_name = min(single_f1, key=single_f1.get)
    worst_single_f1   = single_f1[worst_single_name]

    best_step      = max(trail, key=lambda x: x[1])
    best_greedy_f1 = best_step[1]
    best_greedy_k  = len(best_step[2])
    best_greedy_set = best_step[2]

    print(best_single_name, best_single_f1, worst_single_name, worst_single_f1, best_step, best_greedy_f1, best_greedy_k, best_greedy_set)


if __name__ == "__main__":
    main()
