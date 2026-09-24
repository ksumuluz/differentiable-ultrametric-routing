#!/usr/bin/env python3
"""Portable matched finite-bias ablation for the WOS experiment pipeline.

This preserves the historical scientific protocol while replacing machine-local
paths with command-line arguments. For each requested head seed it compares:
  1) posterior + finite ultrametric bias,
  2) MAP parent + the same finite ultrametric bias,
  3) exact-support hard MAP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_experiment as r
from experiment_core import (
    branch_kernel_ultrametric,
    corrupt_q,
    leaf_metrics,
    one_hot,
    route_hard_map,
    route_soft_kernel,
)


def parse_seeds(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(r.SEEDS)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Root containing WOS46985/. If omitted, use WOS download fallbacks.")
    ap.add_argument("--embedding-cache", type=Path, default=None,
                    help="Optional ModernBERT mean-pool .npy cache. Regenerated if omitted.")
    ap.add_argument("--workdir", type=Path, default=Path("matched_work"),
                    help="Work directory for downloaded data/cache when needed.")
    ap.add_argument("--outdir", type=Path, default=Path("matched_results"))
    ap.add_argument("--seeds", default="all", help="Comma-separated seeds or 'all'.")
    ap.add_argument("--device", default="cuda" if r.torch.cuda.is_available() else "cpu")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--head-epochs", type=int, default=80)
    ap.add_argument("--head-batch", type=int, default=1024)
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    root = args.data_root if args.data_root else r.download_canonical(args.workdir / "data")
    wos = r.find_wos_dir(root)
    texts, y, yp_raw, y2, label_audit = r.load_canonical(wos)
    tr, va, te = r.make_split(y)
    parent_of_leaf, yp, taxonomy_audit = r.derive_train_taxonomy(y, yp_raw, tr)

    cache = args.embedding_cache or (args.workdir / "modernbert_base_meanpool_fp16.npy")
    if cache.exists():
        X = np.load(cache, mmap_mode="r")
    else:
        X = r.extract_embeddings(
            texts, cache, args.embed_batch_size, args.max_length, args.device,
            r.MODEL_ID, r.MODEL_REVISION,
        )

    all_rows: list[dict] = []
    tuning_rows: list[dict] = []
    for seed in parse_seeds(args.seeds):
        model, Xn, _, _ = r.train_heads(
            X, yp, y, tr, va, seed, args.device,
            epochs=args.head_epochs, batch_size=args.head_batch,
        )
        pv, lv = r.logits_all(model, Xn[va], args.device)
        pt, lt = r.logits_all(model, Xn[te], args.device)
        tp = r.fit_temperature(pv, yp[va])
        tl = r.fit_temperature(lv, y[va])
        qv = r.apply_temperature(pv, tp)
        sv = r.apply_temperature(lv, tl)
        qt = r.apply_temperature(pt, tp)
        st = r.apply_temperature(lt, tl)

        prm = r.tune_method("ultrametric", sv, qv, y[va], yp[va], parent_of_leaf, None)
        kernel = branch_kernel_ultrametric(parent_of_leaf, prm["rho"], prm["tau"])

        regimes = [
            ("exact", 0.0, one_hot(yp[te], 7), np.ones(len(te), dtype=bool)),
            ("predicted", 0.0, qt, np.ones(len(te), dtype=bool)),
        ]
        corruption_seed = seed * 1000 + 17
        for eps in r.EPS_GRID:
            regimes.append((
                "corrupted", eps,
                corrupt_q(qt, yp[te], eps, corruption_seed),
                np.ones(len(te), dtype=bool),
            ))

        for regime, eps, q_eval, active in regimes:
            methods = [
                ("posterior_finite", q_eval, False),
                ("map_finite_matched", one_hot(q_eval.argmax(axis=1), 7), False),
                ("hard_map", q_eval, True),
            ]
            for name, q_for_method, hard in methods:
                probs = (
                    route_hard_map(st, q_eval, parent_of_leaf, active)
                    if hard
                    else route_soft_kernel(st, q_for_method, kernel, active)
                )
                all_rows.append({
                    "seed": seed, "method": name, "regime": regime, "epsilon": eps,
                    **leaf_metrics(probs, y[te], yp[te], parent_of_leaf),
                })

        tuning_rows.append({
            "seed": seed,
            "parent_temperature": tp,
            "leaf_temperature": tl,
            "ultrametric_rho": prm["rho"],
            "ultrametric_tau": prm["tau"],
            "parent_head_accuracy": float(np.mean(qt.argmax(axis=1) == yp[te])),
            "semantic_leaf_accuracy": float(np.mean(st.argmax(axis=1) == y[te])),
        })

    df = pd.DataFrame(all_rows)
    df.to_csv(args.outdir / "map_finite_matched_per_seed.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(args.outdir / "head_reproduction.csv", index=False)

    metrics = [
        "leaf_accuracy", "macro_f1", "path_accuracy", "wrong_parent_rate",
        "nll", "ece", "wrong_branch_mass_mean", "catastrophic_wrong_branch",
    ]
    agg = df.groupby(["method", "regime", "epsilon"], dropna=False)[metrics].agg(["mean", "std"])
    agg.to_csv(args.outdir / "map_finite_matched_aggregate.csv")

    audit = {
        "label_audit": label_audit,
        "taxonomy_audit": taxonomy_audit,
        "n_seeds": int(df.seed.nunique()),
        "seeds": sorted(int(x) for x in df.seed.unique()),
        "test_documents": int(len(te)),
    }
    (args.outdir / "matched_ablation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"Wrote matched ablation outputs to {args.outdir}")


if __name__ == "__main__":
    main()
