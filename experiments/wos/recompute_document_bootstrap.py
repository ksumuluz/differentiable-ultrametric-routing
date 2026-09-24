#!/usr/bin/env python3
"""Recompute the paired document bootstrap from stored per-document metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["leaf_accuracy", "nll", "wrong_parent_rate", "wrong_branch_mass_mean", "outside_gold50"]
CONTRASTS = [
    ("map_finite_minus_hard_map", "map_finite", "hard_map"),
    ("posterior_minus_map_finite", "posterior_finite", "map_finite"),
    ("posterior_minus_vanilla", "posterior_finite", "vanilla"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="wos_per_document_key_methods.csv.gz")
    ap.add_argument("--outdir", type=Path, default=Path("bootstrap_check"))
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--replicates", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    wide = df.pivot(index=["seed", "doc_index"], columns="method", values=METRICS)
    rng = np.random.default_rng(args.seed)
    rows, doc_rows = [], []

    for cname, a, b in CONTRASTS:
        for metric in METRICS:
            dsd = (wide[(metric, a)] - wide[(metric, b)]).rename("diff").reset_index()
            ds = dsd.groupby("doc_index", sort=True)["diff"].mean()
            d = ds.to_numpy(float)
            n = len(d)
            boot = np.empty(args.replicates, float)
            for start in range(0, args.replicates, args.chunk):
                m = min(args.chunk, args.replicates - start)
                idx = rng.integers(0, n, size=(m, n))
                boot[start:start + m] = d[idx].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            p2 = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
            rows.append({
                "contrast": cname, "method_a": a, "method_b": b, "metric": metric,
                "delta": float(d.mean()), "boot_ci_low": float(lo), "boot_ci_high": float(hi),
                "boot_p_two_sided": p2, "n_documents": n,
                "n_seeds": int(df.seed.nunique()), "bootstrap_replicates": args.replicates,
            })
            tmp = ds.reset_index(name="mean_seed_diff")
            tmp["contrast"] = cname
            tmp["metric"] = metric
            doc_rows.append(tmp)

    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / "wos_document_bootstrap.csv", index=False)
    (args.outdir / "wos_document_bootstrap.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    pd.concat(doc_rows, ignore_index=True).to_csv(
        args.outdir / "wos_document_level_differences.csv.gz",
        index=False, compression="gzip",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
