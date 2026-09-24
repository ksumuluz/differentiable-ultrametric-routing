from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, f1_score

EPS = 1e-12


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def one_hot(y: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(y), n), dtype=np.float64)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax_np(logits / max(float(temperature), 1e-4))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y.astype(int)], EPS, 1.0))))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(value)


def leaf_metrics(probs: np.ndarray, y_leaf: np.ndarray, y_parent: np.ndarray,
                 parent_of_leaf: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    pred_leaf = probs.argmax(axis=1)
    pred_parent = parent_of_leaf[pred_leaf]
    wrong_parent = pred_parent != y_parent
    wrong_branch_mass = np.ones(len(y_leaf), dtype=np.float64)
    for p in np.unique(y_parent):
        rows = np.where(y_parent == p)[0]
        cols = np.where(parent_of_leaf == p)[0]
        wrong_branch_mass[rows] = 1.0 - probs[np.ix_(rows, cols)].sum(axis=1)
    return {
        "leaf_accuracy": float(accuracy_score(y_leaf, pred_leaf)),
        "macro_f1": float(f1_score(y_leaf, pred_leaf, average="macro", zero_division=0)),
        # In a two-level single-label tree, a correct leaf implies a correct full path.
        "path_accuracy": float(accuracy_score(y_leaf, pred_leaf)),
        "wrong_parent_rate": float(np.mean(wrong_parent)),
        "nll": nll(probs, y_leaf),
        "ece": ece(probs, y_leaf, n_bins=n_bins),
        "wrong_branch_mass_mean": float(np.mean(wrong_branch_mass)),
        "catastrophic_wrong_branch": float(np.mean(wrong_branch_mass > 0.5)),
    }


def branch_kernel_tree(parent_of_leaf: np.ndarray, tau: float) -> np.ndarray:
    """K[p,c] = exp(-d_tree(parent_p, leaf_c)/tau), with d=1 own branch, 3 foreign."""
    p_count = int(parent_of_leaf.max()) + 1
    dist = np.full((p_count, len(parent_of_leaf)), 3.0, dtype=np.float64)
    for c, p in enumerate(parent_of_leaf):
        dist[p, c] = 1.0
    return np.exp(-dist / float(tau))


def branch_kernel_ultrametric(parent_of_leaf: np.ndarray, rho: float, tau: float) -> np.ndarray:
    """Two-shell prefix ultrametric compatibility: d=rho for shared parent, d=1 otherwise."""
    p_count = int(parent_of_leaf.max()) + 1
    dist = np.ones((p_count, len(parent_of_leaf)), dtype=np.float64)
    for c, p in enumerate(parent_of_leaf):
        dist[p, c] = float(rho)
    return np.exp(-dist / float(tau))


def route_vanilla(semantic_probs: np.ndarray) -> np.ndarray:
    return semantic_probs.copy()


def route_hard_map(semantic_probs: np.ndarray, q: np.ndarray, parent_of_leaf: np.ndarray,
                   active: Optional[np.ndarray] = None) -> np.ndarray:
    out = semantic_probs.copy()
    if active is None:
        active = np.ones(len(q), dtype=bool)
    map_parent = q.argmax(axis=1)
    for i in np.where(active)[0]:
        mask = parent_of_leaf == map_parent[i]
        row = out[i] * mask
        s = row.sum()
        out[i] = row / s if s > EPS else semantic_probs[i]
    return out


def route_soft_kernel(semantic_probs: np.ndarray, q: np.ndarray, kernel: np.ndarray,
                      active: Optional[np.ndarray] = None) -> np.ndarray:
    if active is None:
        active = np.ones(len(q), dtype=bool)
    compat = np.clip(q @ kernel, EPS, None)
    out = semantic_probs * compat
    denom = out.sum(axis=1, keepdims=True)
    out = out / np.clip(denom, EPS, None)
    out[~active] = semantic_probs[~active]
    return out


def corrupt_q(q: np.ndarray, y_parent: np.ndarray, epsilon: float, seed: int) -> np.ndarray:
    """Controlled adversarial-but-random branch corruption toward one wrong gold branch.

    r is a one-hot distribution on a randomly selected non-gold parent. Starting from the
    natural predicted q (rather than exact one-hot provenance) makes MAP sensitive to finite
    corruption while retaining a causal, controlled intervention.
    """
    if epsilon <= 0:
        return q.copy()
    rng = np.random.default_rng(seed)
    p_count = q.shape[1]
    wrong = np.empty(len(q), dtype=int)
    for i, gold in enumerate(y_parent.astype(int)):
        choices = np.delete(np.arange(p_count), gold)
        wrong[i] = rng.choice(choices)
    r = one_hot(wrong, p_count)
    out = (1.0 - float(epsilon)) * q + float(epsilon) * r
    return out / out.sum(axis=1, keepdims=True)


def exact_sign_flip_p(diff: Sequence[float]) -> float:
    d = np.asarray(diff, dtype=np.float64)
    n = len(d)
    obs = abs(d.mean())
    if n > 20:
        raise ValueError("Exact sign-flip is intentionally limited to n<=20.")
    ge = 0
    total = 2 ** n
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        val = abs(np.mean(d * np.asarray(signs)))
        ge += val >= obs - 1e-15
    return ge / total


def paired_stats(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have same shape")
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    crit = float(student_t.ppf(1 - alpha / 2, df=n - 1)) if n > 1 else float("nan")
    dz = mean / sd if sd > 0 else (math.copysign(math.inf, mean) if mean else 0.0)
    return {
        "delta": mean,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "cohen_dz": float(dz),
        "p_sign_flip": float(exact_sign_flip_p(d)) if n <= 20 else float("nan"),
    }


def audit_raw_wos_labels(y_flat: np.ndarray, y_parent: np.ndarray, y_l2: np.ndarray,
                         expected_n: int = 46985, expected_leaf: int = 134,
                         expected_parent: int = 7):
    """Validate the released WOS-46985 label arrays without inventing a hierarchy.

    WOS-46985 publishes three numeric targets.  ``Y`` is the 134-class flat
    target, ``YL1`` is the 7-class parent/domain target, and ``YL2`` is a
    parent-local level-2 code.  The public release is not perfectly relationally
    consistent: some flat ``Y`` ids occur with more than one ``YL1`` value and
    the observed ``(YL1,YL2)`` pairs need not total 134.  Consequently, ingestion
    must preserve ``Y`` as the leaf target and treat the other arrays as noisy
    hierarchy metadata rather than merging flat classes.

    Returns zero-based contiguous Y and YL1 arrays plus diagnostics.  No
    train/validation/test information is used here.
    """
    y_flat = np.asarray(y_flat, dtype=int)
    y_parent = np.asarray(y_parent, dtype=int)
    y_l2 = np.asarray(y_l2, dtype=int)
    if not (len(y_flat) == len(y_parent) == len(y_l2) == expected_n):
        raise ValueError(
            f"Expected {expected_n} documents, got Y={len(y_flat)}, "
            f"YL1={len(y_parent)}, YL2={len(y_l2)}"
        )

    leaves = np.unique(y_flat)
    parents = np.unique(y_parent)
    if len(leaves) != expected_leaf:
        raise ValueError(f"Expected {expected_leaf} flat Y classes, got {len(leaves)}: {leaves}")
    if len(parents) != expected_parent:
        raise ValueError(f"Expected {expected_parent} YL1 classes, got {len(parents)}: {parents}")
    if not np.array_equal(leaves, np.arange(leaves.min(), leaves.min() + expected_leaf)):
        raise ValueError("Flat Y ids are not contiguous; inspect WOS labels before proceeding.")
    if not np.array_equal(parents, np.arange(parents.min(), parents.min() + expected_parent)):
        raise ValueError("YL1 ids are not contiguous; inspect WOS labels before proceeding.")

    y_leaf0 = y_flat - leaves.min()
    y_parent0 = y_parent - parents.min()
    paths = np.c_[y_parent0, y_l2]
    unique_paths = np.unique(paths, axis=0)

    # Whole-release diagnostics only; these are not used to define the taxonomy.
    flat_to_parent_counts = {}
    ambiguous_flat = 0
    for c in range(expected_leaf):
        vals, counts = np.unique(y_parent0[y_leaf0 == c], return_counts=True)
        flat_to_parent_counts[str(c)] = {str(int(v)): int(n) for v, n in zip(vals, counts)}
        ambiguous_flat += int(len(vals) > 1)

    path_to_flat_ambiguous = 0
    for pair in unique_paths:
        mask = (paths[:, 0] == pair[0]) & (paths[:, 1] == pair[1])
        path_to_flat_ambiguous += int(len(np.unique(y_leaf0[mask])) > 1)

    diagnostics = {
        "n_documents": int(expected_n),
        "n_flat_Y_classes": int(len(leaves)),
        "n_YL1_parents": int(len(parents)),
        "n_unique_YL1_YL2_pairs": int(len(unique_paths)),
        "n_flat_Y_ids_with_multiple_observed_YL1": int(ambiguous_flat),
        "n_YL1_YL2_pairs_with_multiple_flat_Y": int(path_to_flat_ambiguous),
        "flat_Y_to_observed_YL1_counts": flat_to_parent_counts,
        "leaf_definition": "released flat Y (134 classes)",
        "YL2_role": "diagnostic parent-local code; not a global leaf id",
    }
    return y_leaf0, y_parent0, diagnostics


def derive_train_taxonomy(y_leaf: np.ndarray, y_parent_raw: np.ndarray, train_idx: np.ndarray,
                          expected_leaf: int = 134, expected_parent: int = 7):
    """Recover a deterministic 134->7 taxonomy from training rows only.

    For each released flat leaf ``Y=c``, select the modal observed ``YL1`` among
    training rows with that leaf.  This repairs sparse relational inconsistencies
    in the public release while preventing validation/test labels from determining
    the taxonomy.  The returned canonical parent label for every document is then
    ``parent_of_leaf[Y]``; raw YL1 is retained only for diagnostics.
    """
    y_leaf = np.asarray(y_leaf, dtype=int)
    y_parent_raw = np.asarray(y_parent_raw, dtype=int)
    train_idx = np.asarray(train_idx, dtype=int)
    parent_of_leaf = np.full(expected_leaf, -1, dtype=int)
    per_leaf = {}

    for c in range(expected_leaf):
        rows = train_idx[y_leaf[train_idx] == c]
        if len(rows) == 0:
            raise ValueError(f"Leaf {c} has no training rows; cannot infer taxonomy without leakage.")
        vals, counts = np.unique(y_parent_raw[rows], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        vals, counts = vals[order], counts[order]
        if len(counts) > 1 and counts[0] == counts[1]:
            raise ValueError(
                f"Leaf {c} has a tied modal parent in training data: "
                f"{dict(zip(vals.tolist(), counts.tolist()))}."
            )
        parent_of_leaf[c] = int(vals[0])
        per_leaf[str(c)] = {
            "parent": int(vals[0]),
            "train_count": int(counts.sum()),
            "train_parent_purity": float(counts[0] / counts.sum()),
            "observed_train_parent_counts": {str(int(v)): int(n) for v, n in zip(vals, counts)},
        }

    if np.any((parent_of_leaf < 0) | (parent_of_leaf >= expected_parent)):
        raise ValueError("Derived taxonomy contains invalid parent ids.")
    represented = np.unique(parent_of_leaf)
    if len(represented) != expected_parent:
        raise ValueError(
            f"Derived taxonomy represents {len(represented)} parents rather than {expected_parent}: {represented}"
        )

    y_parent_canonical = parent_of_leaf[y_leaf]
    train_purities = np.array([per_leaf[str(c)]["train_parent_purity"] for c in range(expected_leaf)])
    diagnostics = {
        "taxonomy_rule": "modal YL1 per flat Y estimated on training split only",
        "taxonomy_uses_validation_or_test_labels": False,
        "n_leaves": int(expected_leaf),
        "n_parents": int(expected_parent),
        "min_train_parent_purity": float(train_purities.min()),
        "mean_train_parent_purity": float(train_purities.mean()),
        "n_leaves_train_purity_lt_0_95": int(np.sum(train_purities < 0.95)),
        "n_leaves_train_purity_lt_0_80": int(np.sum(train_purities < 0.80)),
        "per_leaf": per_leaf,
    }
    return parent_of_leaf, y_parent_canonical, diagnostics


def split_parent_disagreement(y_parent_raw: np.ndarray, y_parent_canonical: np.ndarray,
                              train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    """Report raw-YL1 disagreement with the train-derived deterministic taxonomy."""
    out = {}
    for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        idx = np.asarray(idx, dtype=int)
        disagree = y_parent_raw[idx] != y_parent_canonical[idx]
        out[name] = {
            "rows": int(len(idx)),
            "disagreement_rows": int(disagree.sum()),
            "disagreement_rate": float(disagree.mean()),
        }
    return out


def audit_labels(y_leaf: np.ndarray, y_parent: np.ndarray, y_l2: Optional[np.ndarray] = None,
                 expected_n: int = 46985, expected_leaf: int = 134, expected_parent: int = 7) -> np.ndarray:
    """Backward-compatible strict audit for already canonical hierarchical leaf ids.

    New WOS ingestion should use :func:`audit_raw_wos_labels` followed by :func:`derive_train_taxonomy`.
    """
    if len(y_leaf) != expected_n or len(y_parent) != expected_n:
        raise ValueError(f"Expected {expected_n} documents, got leaf={len(y_leaf)}, parent={len(y_parent)}")
    leaves = np.unique(y_leaf)
    parents = np.unique(y_parent)
    if len(leaves) != expected_leaf or len(parents) != expected_parent:
        raise ValueError(
            f"Expected {expected_leaf} leaves/{expected_parent} parents, "
            f"got {len(leaves)}/{len(parents)}"
        )
    parent_of_leaf = np.full(expected_leaf, -1, dtype=int)
    for c in range(expected_leaf):
        ps = np.unique(y_parent[y_leaf == c])
        if len(ps) != 1:
            raise ValueError(f"Leaf {c} maps to {len(ps)} parents: {ps}")
        parent_of_leaf[c] = int(ps[0])
    return parent_of_leaf


@dataclass(frozen=True)
class RoutingParams:
    tree_tau: float = 1.0
    ultra_rho: float = 0.5
    ultra_tau: float = 1.0
    shell_a: float = 0.33
    shell_tau: float = 1.0
    poincare_tau: float = 1.0
