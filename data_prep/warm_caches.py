from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="acs_cache")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--datasets", nargs="+",
                    default=["ACSIncome", "ACSPublicCoverage", "ACSTravelTime"])
    ap.add_argument("--methods", nargs="+", default=["normal"],
                    choices=["normal", "dp", "gradmatch", "hybrid"])
    ap.add_argument("--with-wilds", action="store_true")
    ap.add_argument("--with-amazon", action="store_true")
    args = ap.parse_args()

    from data_prep.acs import build_acs_context

    for ds in args.datasets:
        for m in args.methods:
            t0 = time.time()
            build_acs_context(ds, cache_dir=args.cache_dir, method=m)

    if args.with_wilds:
        t0 = time.time()
        from data_prep import wilds as wn
        wn.init_wilds()

    if args.with_amazon:
        t0 = time.time()
        from data_prep.amazon import get_amazon_info
        src_cats, *_ = get_amazon_info()


if __name__ == "__main__":
    main()
