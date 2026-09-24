#!/usr/bin/env python3
"""
Independent numerical verification for Theorem 1 in:

"Differentiable Ultrametric Routing for Hierarchical Evidence
under Uncertain Provenance"

IMPORTANT
---------
This script implements an independent numerical check from the equations printed in the
current main manuscript and supplementary material. It is not claimed
to be the historical verification script used during manuscript development.

The publication states that an executable verification used 3,888
(rho, s, D) grid points, but the exact historical grid values are not printed.
The independent check therefore uses a transparent 9 x 9 x 48 grid:

    rho in {0.1, ..., 0.9}
    s   in {0.1, ..., 0.9}
    D   in {2, ..., 49}

giving exactly 3,888 points.

The script verifies:
1. Strict decrease of G_tel(m) with depth m.
2. Strict decrease of G_exp(m) with depth m.
3. The identity

       kappa_tel(D) / kappa_exp(D)
       = rho^{-(D-1)} C(rho, s, D).

4. The supplementary-material bounds c1 <= C <= c2.
5. The stated consecutive-difference identities.

Outputs:
- independent_verification_grid.csv
- independent_verification_summary.json

Only Python stdlib + NumPy + pandas are required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def v_D(rho: float, s: float, D: int) -> float:
    del rho
    return s * (1.0 - s**D) / (1.0 - s)


def a_D(rho: float, s: float, D: int) -> float:
    return (-math.log(rho)) * rho ** v_D(rho, s, D)


def g_tel_profile(rho: float, s: float, D: int) -> np.ndarray:
    """Equation (9), m=0,...,D-1."""
    out = np.empty(D, dtype=np.float64)
    rs = rho * s
    for m in range(D):
        if m <= D - 2:
            geom = sum(rs**d for d in range(m, D - 1))
        else:
            geom = 0.0
        out[m] = (1.0 - rho) * geom + rs ** (D - 1)
    return out


def g_exp_profile(rho: float, s: float, D: int) -> np.ndarray:
    """Equation (10), m=0,...,D-1."""
    aa = a_D(rho, s, D)
    out = np.empty(D, dtype=np.float64)
    for m in range(D):
        out[m] = aa * sum(s**d for d in range(m, D)) + rho**D * s ** (D - 1)
    return out


def c_bounds(rho: float, s: float):
    """Explicit c1,c2 bounds from Supplementary Material Sec. 2.2."""
    L = -math.log(rho)
    a_minus = L * rho * s
    a_plus = L * rho * s / (1.0 - s)

    t_minus = 1.0 - rho
    t_plus = (1.0 - rho) / (1.0 - rho * s) + rho * s

    e_minus = a_minus
    e_plus = a_plus / (1.0 - s) + rho**2 * s

    c1 = t_minus * a_minus / e_plus
    c2 = t_plus * (a_plus + rho**2) / e_minus
    return c1, c2


def verify_point(rho: float, s: float, D: int) -> dict:
    gt = g_tel_profile(rho, s, D)
    ge = g_exp_profile(rho, s, D)

    tel_diffs = gt[:-1] - gt[1:]
    exp_diffs = ge[:-1] - ge[1:]

    # Published consecutive-difference identities, Supplement Eqs. (6)-(7).
    m = np.arange(D - 1, dtype=np.float64)
    tel_expected = (1.0 - rho) * (rho * s) ** m
    exp_expected = a_D(rho, s, D) * s**m

    tel_diff_abs_err = float(np.max(np.abs(tel_diffs - tel_expected)))
    exp_diff_abs_err = float(np.max(np.abs(exp_diffs - exp_expected)))

    T_D = float(gt[0])
    E_D = float(ge[0])
    aa = a_D(rho, s, D)

    C = T_D * (aa + rho**D) / E_D
    c1, c2 = c_bounds(rho, s)

    # Strict decrease means extrema are at m=0 and m=D-1.
    kappa_tel = float(gt[0] / gt[-1])
    kappa_exp = float(ge[0] / ge[-1])

    lhs = kappa_tel / kappa_exp
    rhs = rho ** (-(D - 1)) * C
    rel_identity_error = abs(lhs - rhs) / max(abs(rhs), np.finfo(float).tiny)

    return {
        "rho": rho,
        "s": s,
        "D": D,
        "tel_strictly_decreasing": bool(np.all(tel_diffs > 0.0)),
        "exp_strictly_decreasing": bool(np.all(exp_diffs > 0.0)),
        "C": C,
        "c1": c1,
        "c2": c2,
        "C_lower_bound_ok": bool(C >= c1),
        "C_upper_bound_ok": bool(C <= c2),
        "kappa_tel": kappa_tel,
        "kappa_exp": kappa_exp,
        "identity_lhs": lhs,
        "identity_rhs": rhs,
        "relative_identity_error": float(rel_identity_error),
        "tel_consecutive_diff_max_abs_error": tel_diff_abs_err,
        "exp_consecutive_diff_max_abs_error": exp_diff_abs_err,
    }


def parse_float_grid(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_grid(text: str):
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rho",
        default=",".join(f"{x/10:.1f}" for x in range(1, 10)),
        help="Comma-separated rho grid. Default: 0.1,...,0.9",
    )
    ap.add_argument(
        "--s",
        default=",".join(f"{x/10:.1f}" for x in range(1, 10)),
        help="Comma-separated s grid. Default: 0.1,...,0.9",
    )
    ap.add_argument(
        "--D",
        default=",".join(str(x) for x in range(2, 50)),
        help="Comma-separated depth grid. Default: integers 2,...,49",
    )
    ap.add_argument("--outdir", type=Path, default=Path("results/theory"))
    args = ap.parse_args()

    rhos = parse_float_grid(args.rho)
    ss = parse_float_grid(args.s)
    Ds = parse_int_grid(args.D)

    if any(not (0.0 < x < 1.0) for x in rhos):
        raise ValueError("All rho values must lie strictly inside (0,1).")
    if any(not (0.0 < x < 1.0) for x in ss):
        raise ValueError("All s values must lie strictly inside (0,1).")
    if any(D < 2 for D in Ds):
        raise ValueError("All D values must satisfy D>=2.")

    rows = [verify_point(rho, s, D) for rho in rhos for s in ss for D in Ds]
    df = pd.DataFrame(rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    grid_path = args.outdir / "independent_verification_grid.csv"
    summary_path = args.outdir / "independent_verification_summary.json"
    df.to_csv(grid_path, index=False)

    worst_idx = int(df["relative_identity_error"].idxmax())
    worst = df.loc[worst_idx]

    monotonicity_violations = int(
        (~df["tel_strictly_decreasing"]).sum()
        + (~df["exp_strictly_decreasing"]).sum()
    )
    bound_violations = int(
        (~df["C_lower_bound_ok"]).sum()
        + (~df["C_upper_bound_ok"]).sum()
    )

    summary = {
        "status": "PASS" if monotonicity_violations == 0 and bound_violations == 0 else "FAIL",
        "provenance_note": (
            "Independent executable check based on the manuscript/supplement equations. "
            "The exact historical 3,888-point coordinates were not printed in the paper; "
            "the default grid is stated explicitly in this script."
        ),
        "grid": {
            "rho": rhos,
            "s": ss,
            "D": Ds,
            "n_points": int(len(df)),
        },
        "checks": {
            "monotonicity_violations": monotonicity_violations,
            "bound_violations": bound_violations,
            "max_relative_identity_error": float(df["relative_identity_error"].max()),
            "max_tel_consecutive_diff_abs_error": float(
                df["tel_consecutive_diff_max_abs_error"].max()
            ),
            "max_exp_consecutive_diff_abs_error": float(
                df["exp_consecutive_diff_max_abs_error"].max()
            ),
        },
        "worst_identity_point": {
            "rho": float(worst["rho"]),
            "s": float(worst["s"]),
            "D": int(worst["D"]),
            "lhs": float(worst["identity_lhs"]),
            "rhs": float(worst["identity_rhs"]),
            "relative_error": float(worst["relative_identity_error"]),
        },
        "paper_statement_for_comparison": {
            "reported_grid_points": 3888,
            "reported_worst_relative_identity_error": 1.1e-14,
            "note": (
                "The independent default grid has the same number of points. "
                "Its floating-point worst-case error need not equal the historical "
                "reported value because the historical grid values are not given."
            ),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
