#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedShuffleSplit

from experiment_core import (
    RoutingParams, apply_temperature, audit_raw_wos_labels, branch_kernel_tree,
    branch_kernel_ultrametric, corrupt_q, derive_train_taxonomy, ece, leaf_metrics,
    nll, one_hot, paired_stats, route_hard_map, route_soft_kernel, route_vanilla,
    split_parent_disagreement,
)

MENDELEY_CANONICAL_URL = "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded"
GITHUB_WOS_REPO = "wuchengyuan88/document-classification"
GITHUB_WOS_COMMIT = "d482ffcc0a56e6e668929b1c75b7c6cf8f19e93d"
GITHUB_WOS_BASE = f"https://raw.githubusercontent.com/{GITHUB_WOS_REPO}/{GITHUB_WOS_COMMIT}/WOS46985"
HF_WOS_BASE = "https://huggingface.co/datasets/FlavioSoriano/wos-46985-rag-fuse/resolve/main/raw/WOS46985"
WOS_FILES = ("X.txt", "Y.txt", "YL1.txt", "YL2.txt")
MODEL_ID = "answerdotai/ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 109]
EPS_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download_file(url: str, target: Path, *, timeout: int = 180) -> None:
    import requests

    headers = {
        "User-Agent": "wos-routing-repro/1.4 (+research reproducibility)",
        "Accept": "*/*",
    }
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True, headers=headers) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty file from {url}")
    tmp.replace(target)


def _download_raw_set(dest: Path, base_url: str, source_name: str) -> Path:
    """Download the four WOS46985 raw files from a mirror.

    Files are not trusted solely because the HTTP request succeeds. The main
    pipeline subsequently enforces the 46,985 / 134 / 7 canonical audit and
    unique leaf-to-parent mapping before any model computation begins.
    """
    wos = dest / "extracted" / "WOS46985"
    wos.mkdir(parents=True, exist_ok=True)
    for name in WOS_FILES:
        target = wos / name
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{base_url}/{name}"
        if "huggingface.co" in base_url:
            url += "?download=true"
        print(f"Downloading {name} from {source_name} ...", flush=True)
        _download_file(url, target)
    return dest / "extracted"


def _download_github_raw(dest: Path) -> Path:
    # Public mirror pinned to the commit that introduced the WOS datasets.
    return _download_raw_set(dest, GITHUB_WOS_BASE, "pinned GitHub mirror")


def _download_hf_raw(dest: Path) -> Path:
    # Secondary mirror only: some environments receive HTTP 401 from this repo.
    return _download_raw_set(dest, HF_WOS_BASE, "Hugging Face mirror")


def _download_mendeley_archive(dest: Path) -> Path:
    """Legacy fallback for the canonical Mendeley archive."""
    import requests

    archive = dest / "wos_mendeley_archive.bin"
    if not archive.exists():
        headers = {
            "User-Agent": "Mozilla/5.0 wos-routing-repro/1.2",
            "Accept": "application/octet-stream,*/*",
            "Referer": "https://data.mendeley.com/datasets/9rw3vkcfy4/2",
        }
        with requests.get(MENDELEY_CANONICAL_URL, stream=True, timeout=180,
                          allow_redirects=True, headers=headers) as r:
            r.raise_for_status()
            with archive.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    extract = dest / "extracted"
    if not extract.exists():
        extract.mkdir(parents=True)
    if not any(extract.iterdir()):
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                z.extractall(extract)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                t.extractall(extract)
        else:
            raise RuntimeError("Canonical archive format not recognized. Keep the file and inspect its magic bytes.")
    return extract


def download_canonical(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    errors = []

    # 1) Public GitHub mirror, pinned to an immutable commit.
    try:
        return _download_github_raw(dest)
    except Exception as exc:
        errors.append(("GitHub", exc))
        print(f"Pinned GitHub raw-file download failed: {exc}", flush=True)

    # 2) Hugging Face mirror. Kept as a fallback because it can require auth
    #    depending on repository state / regional edge configuration.
    try:
        return _download_hf_raw(dest)
    except Exception as exc:
        errors.append(("Hugging Face", exc))
        print(f"Hugging Face raw-file download failed: {exc}", flush=True)

    # 3) Historical Mendeley direct archive endpoint. It is canonical but may
    #    reject scripted clients with HTTP 403.
    print("Trying legacy Mendeley archive fallback ...", flush=True)
    try:
        return _download_mendeley_archive(dest)
    except Exception as exc:
        errors.append(("Mendeley", exc))

    detail = "; ".join(f"{name}: {err}" for name, err in errors)
    raise RuntimeError(
        "Could not download WOS-46985 automatically from any configured source. "
        "You can manually place X.txt, Y.txt, YL1.txt and YL2.txt under a "
        "WOS46985 folder and rerun with --data-root /path/to/folder-containing-WOS46985. "
        f"Errors: {detail}"
    ) from errors[-1][1]


def find_wos_dir(root: Path) -> Path:
    candidates = [p for p in root.rglob("WOS46985") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find WOS46985 below {root}")
    for c in candidates:
        if all((c / name).exists() for name in ("X.txt", "Y.txt", "YL1.txt", "YL2.txt")):
            return c
    raise FileNotFoundError("Found WOS46985 directory but required X/Y/YL1/YL2 files are missing")


def load_canonical(wos_dir: Path):
    def lines(name):
        return (wos_dir / name).read_text(encoding="utf-8", errors="replace").splitlines()
    texts = np.asarray(lines("X.txt"), dtype=object)
    y_raw = np.asarray([int(x.strip()) for x in lines("Y.txt")], dtype=int)
    y1_raw = np.asarray([int(x.strip()) for x in lines("YL1.txt")], dtype=int)
    y2 = np.asarray([int(x.strip()) for x in lines("YL2.txt")], dtype=int)
    y, y1_raw, label_audit = audit_raw_wos_labels(y_raw, y1_raw, y2)
    if len(texts) != len(y):
        raise ValueError(f"Expected X and labels to align, got X={len(texts)} and Y={len(y)}")
    return texts, y, y1_raw, y2, label_audit


def make_split(y: np.ndarray, split_seed: int = 1729):
    s1 = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=split_seed)
    train_idx, rest_idx = next(s1.split(np.zeros(len(y)), y))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=split_seed + 1)
    val_rel, test_rel = next(s2.split(np.zeros(len(rest_idx)), y[rest_idx]))
    return np.sort(train_idx), np.sort(rest_idx[val_rel]), np.sort(rest_idx[test_rel])


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


def extract_embeddings(texts, out_path: Path, batch_size: int, max_length: int, device: str,
                       model_source: str = MODEL_ID, model_revision: str | None = MODEL_REVISION):
    if out_path.exists():
        arr = np.load(out_path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[0] == len(texts) and arr.shape[1] > 0:
            return arr
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers>=4.48 is required for ModernBERT embedding extraction") from e
    source_path = Path(model_source).expanduser()
    local_model = source_path.exists()
    common = {"local_files_only": True} if local_model else {"revision": model_revision}
    tokenizer = AutoTokenizer.from_pretrained(model_source, **common)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModel.from_pretrained(model_source, torch_dtype=dtype, **common).to(device).eval()
    hidden_size = int(model.config.hidden_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype="float16", shape=(len(texts), hidden_size))
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [str(x) for x in texts[start:start + batch_size]]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items() if k != "token_type_ids"}
            out = model(**tok)
            h = masked_mean(out.last_hidden_state, tok["attention_mask"]).float().cpu().numpy()
            mm[start:start + len(h)] = h.astype(np.float16)
            if (start // batch_size) % 50 == 0:
                print(f"embedding {start + len(h)}/{len(texts)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.load(out_path, mmap_mode="r")


class DualHead(torch.nn.Module):
    def __init__(self, dim: int, n_parent: int, n_leaf: int):
        super().__init__()
        self.parent = torch.nn.Linear(dim, n_parent)
        self.leaf = torch.nn.Linear(dim, n_leaf)

    def forward(self, x):
        return self.parent(x), self.leaf(x)


def logits_all(model, X: np.ndarray, device: str, batch_size: int = 4096):
    ps, ls = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X[start:start+batch_size], dtype=np.float32)).to(device)
            p, l = model(xb)
            ps.append(p.cpu().numpy()); ls.append(l.cpu().numpy())
    return np.concatenate(ps), np.concatenate(ls)


def train_heads(X, y_parent, y_leaf, train_idx, val_idx, seed: int, device: str,
                epochs: int = 80, batch_size: int = 1024, lr: float = 2e-3, weight_decay: float = 1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    # Normalize with train-only moments; this is saved implicitly by returning standardized arrays later.
    mu = np.asarray(X[train_idx], dtype=np.float32).mean(axis=0)
    sd = np.asarray(X[train_idx], dtype=np.float32).std(axis=0) + 1e-5
    Xn = (np.asarray(X, dtype=np.float32) - mu) / sd
    model = DualHead(Xn.shape[1], 7, 134).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    best = None; best_val = float("inf"); stale = 0
    for epoch in range(epochs):
        model.train()
        order = train_idx.copy(); rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            xb = torch.from_numpy(Xn[ids]).to(device)
            yp = torch.from_numpy(y_parent[ids]).long().to(device)
            yl = torch.from_numpy(y_leaf[ids]).long().to(device)
            p, l = model(xb)
            loss = ce(p, yp) + ce(l, yl)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        vp, vl = logits_all(model, Xn[val_idx], device)
        leaf_val_nll = float(-np.mean(vl[np.arange(len(val_idx)), y_leaf[val_idx]] - np.logaddexp.reduce(vl, axis=1)))
        parent_val_nll = float(-np.mean(vp[np.arange(len(val_idx)), y_parent[val_idx]] - np.logaddexp.reduce(vp, axis=1)))
        val_loss = leaf_val_nll + parent_val_nll
        if val_loss < best_val - 1e-5:
            best_val = val_loss; stale = 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best)
    return model, Xn, mu, sd


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def objective(log_t):
        t = math.exp(float(log_t))
        p = apply_temperature(logits, t)
        return nll(p, y)
    res = minimize_scalar(objective, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(res.x))


def tree_shortest_distance(parent_of_leaf: np.ndarray) -> np.ndarray:
    n_p, n_l = 7, 134
    n = 1 + n_p + n_l
    adj = np.full((n, n), np.inf); np.fill_diagonal(adj, 0.0)
    for p in range(n_p):
        adj[0, 1+p] = adj[1+p, 0] = 1.0
    for c, p in enumerate(parent_of_leaf):
        a, b = 1+p, 1+n_p+c
        adj[a,b] = adj[b,a] = 1.0
    # Floyd-Warshall on 142 nodes is negligible.
    d = adj.copy()
    for k in range(n):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    return d


def poincare_distance(x: torch.Tensor) -> torch.Tensor:
    # pairwise Poincare-ball distance, curvature -1
    x2 = (x*x).sum(-1)
    diff2 = ((x[:,None,:] - x[None,:,:])**2).sum(-1)
    denom = (1-x2)[:,None] * (1-x2)[None,:]
    arg = 1 + 2*diff2 / denom.clamp_min(1e-6)
    return torch.acosh(arg.clamp_min(1+1e-6))


def fit_poincare(parent_of_leaf: np.ndarray, cache: Path, dim: int = 10, steps: int = 2500, seed: int = 271828):
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (142, dim): return arr
    torch.manual_seed(seed)
    target = torch.tensor(tree_shortest_distance(parent_of_leaf), dtype=torch.float32)
    raw = torch.nn.Parameter(torch.randn(142, dim) * 0.01)
    log_scale = torch.nn.Parameter(torch.tensor(-0.2))
    opt = torch.optim.Adam([raw, log_scale], lr=0.03)
    tri = torch.triu(torch.ones(142,142,dtype=torch.bool), diagonal=1)
    for step in range(steps):
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x = 0.95 * torch.tanh(norm) * raw / norm
        dh = poincare_distance(x)
        scale = torch.exp(log_scale)
        loss = torch.mean((dh[tri] - scale*target[tri])**2) + 1e-4*(x*x).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x = (0.95 * torch.tanh(norm) * raw / norm).cpu().numpy()
    cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, x)
    return x


def poincare_parent_leaf_kernel(emb: np.ndarray, tau: float) -> np.ndarray:
    p = emb[1:8]
    l = emb[8:]
    p2 = (p*p).sum(1)[:,None]; l2=(l*l).sum(1)[None,:]
    diff2=((p[:,None,:]-l[None,:,:])**2).sum(-1)
    arg=1+2*diff2/np.clip((1-p2)*(1-l2),1e-8,None)
    d=np.arccosh(np.clip(arg,1+1e-8,None))
    return np.exp(-d/float(tau))


def tune_method(method, semantic_val, q_val, y_val, parent_val, parent_of_leaf, poincare_emb):
    if method in ("vanilla", "hard_map"):
        return {}
    best = (float("inf"), None)
    if method == "tree":
        grid = [{"tau":x} for x in (0.15,0.25,0.4,0.67,1.0,1.5,2.5,4.0)]
    elif method == "ultrametric":
        grid = [{"rho":0.5,"tau":x} for x in (0.08,0.12,0.2,0.33,0.5,0.8,1.2,2.0)]
    elif method == "geometric_shell":
        grid = [{"a":a,"tau":1.0} for a in (0.15,0.22,0.33,0.45,0.6,0.75,0.9)]
    elif method == "poincare":
        grid = [{"tau":x} for x in (0.1,0.15,0.25,0.4,0.67,1.0,1.5,2.5,4.0)]
    else: raise KeyError(method)
    for prm in grid:
        if method == "tree": k=branch_kernel_tree(parent_of_leaf, prm["tau"])
        elif method == "ultrametric": k=branch_kernel_ultrametric(parent_of_leaf, prm["rho"], prm["tau"])
        elif method == "geometric_shell": k=branch_kernel_ultrametric(parent_of_leaf, prm["a"], prm["tau"])
        else: k=poincare_parent_leaf_kernel(poincare_emb, prm["tau"])
        p=route_soft_kernel(semantic_val,q_val,k)
        score=nll(p,y_val)
        if score<best[0]: best=(score,prm)
    return best[1]


def route(method, semantic, q, parent_of_leaf, prm, poincare_emb, active=None):
    if method=="vanilla": return route_vanilla(semantic)
    if method=="hard_map": return route_hard_map(semantic,q,parent_of_leaf,active)
    if method=="tree": k=branch_kernel_tree(parent_of_leaf,prm["tau"])
    elif method=="ultrametric": k=branch_kernel_ultrametric(parent_of_leaf,prm["rho"],prm["tau"])
    elif method=="geometric_shell": k=branch_kernel_ultrametric(parent_of_leaf,prm["a"],prm["tau"])
    elif method=="poincare": k=poincare_parent_leaf_kernel(poincare_emb,prm["tau"])
    else: raise KeyError(method)
    return route_soft_kernel(semantic,q,k,active)


def latex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def write_outputs(rows, outdir: Path):
    df=pd.DataFrame(rows)
    df.to_csv(outdir/"results_per_seed.csv",index=False)
    keys=["method","regime","epsilon"]
    metrics=["leaf_accuracy","macro_f1","path_accuracy","wrong_parent_rate","nll","ece","wrong_branch_mass_mean","catastrophic_wrong_branch"]
    agg=df.groupby(keys,dropna=False)[metrics].agg(["mean","std"]).reset_index()
    agg.to_csv(outdir/"results_aggregate.csv",index=False)
    corruption = agg[agg["regime"] == "corrupted"].copy()
    corruption.to_csv(outdir/"corruption_results.csv", index=False)
    comparisons={}
    for regime,eps in [("exact",0.0),("predicted",0.0),("corrupted",0.3)]:
        sub=df[(df.regime==regime)&(df.epsilon==eps)]
        for metric in ("leaf_accuracy","catastrophic_wrong_branch","wrong_branch_mass_mean"):
            pivot=sub.pivot(index="seed",columns="method",values=metric)
            if {"ultrametric","hard_map"}.issubset(pivot.columns):
                comparisons[f"{regime}_eps{eps}_{metric}_ultra_minus_hard"]=paired_stats(pivot["ultrametric"],pivot["hard_map"])
    (outdir/"pairwise_comparisons.json").write_text(json.dumps(comparisons,indent=2),encoding="utf-8")
    # Compact manuscript table for the natural-predicted provenance condition.
    nat=df[(df.regime=="predicted")&(df.epsilon==0.0)]
    tab=nat.groupby("method")[metrics].agg(["mean","std"])
    order=["vanilla","hard_map","tree","poincare","ultrametric","geometric_shell"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{External WOS-46985 transfer with frozen ModernBERT-base and natural predicted provenance. Mean $\pm$ SD over 10 paired head seeds.}",
        r"\label{tab:wos_external}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Leaf Acc. & Macro-F1 & Wrong-parent & Catastrophic wrong-branch \\",
        r"\midrule",
    ]
    for m in order:
        if m not in tab.index:
            continue
        def fmt(k):
            return f"{tab.loc[m,(k,'mean')]:.4f} $\\pm$ {tab.loc[m,(k,'std')]:.4f}"
        lines.append(
            f"{latex_escape(m)} & {fmt('leaf_accuracy')} & {fmt('macro_f1')} & "
            f"{fmt('wrong_parent_rate')} & {fmt('catastrophic_wrong_branch')} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir/"results_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return df,agg,comparisons


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("wos_run"))
    ap.add_argument("--data-root",type=Path,default=None,help="Existing extracted Mendeley root; otherwise download canonical archive")
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--max-length",type=int,default=512)
    ap.add_argument("--model-source", default=MODEL_ID, help="HF model id or a local ModernBERT snapshot directory")
    ap.add_argument("--model-revision", default=MODEL_REVISION, help="Pinned HF revision; ignored for a local model directory")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--head-epochs",type=int,default=80)
    ap.add_argument("--head-batch",type=int,default=1024)
    args=ap.parse_args()
    out=args.workdir; out.mkdir(parents=True,exist_ok=True)
    root=args.data_root if args.data_root else download_canonical(out/"data")
    wos=find_wos_dir(root)
    texts,y,yp_raw,y2,label_audit=load_canonical(wos)
    train_idx,val_idx,test_idx=make_split(y)
    parent_of_leaf,yp,taxonomy_audit=derive_train_taxonomy(y,yp_raw,train_idx)
    parent_disagreement=split_parent_disagreement(yp_raw,yp,train_idx,val_idx,test_idx)
    taxonomy_hash=hashlib.sha256(parent_of_leaf.astype(np.int16).tobytes()).hexdigest()[:16]
    audit={
        "n_documents":len(y),
        "n_leaf":int(len(np.unique(y))),
        "n_parent":int(len(np.unique(yp))),
        "train":len(train_idx),"val":len(val_idx),"test":len(test_idx),
        "model":args.model_source,"model_revision":args.model_revision,"max_length":args.max_length,
        "label_audit":label_audit,
        "taxonomy_audit":taxonomy_audit,
        "raw_YL1_vs_canonical_parent":parent_disagreement,
        "taxonomy_hash":taxonomy_hash,
        "evaluation_parent_definition":"train-derived parent_of_leaf[Y] (raw YL1 retained for audit only)",
    }
    (out/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(
        "WOS label audit: "
        f"Y={audit['n_leaf']} flat leaves; raw (YL1,YL2) pairs={label_audit['n_unique_YL1_YL2_pairs']}; "
        f"ambiguous Y->raw YL1 ids={label_audit['n_flat_Y_ids_with_multiple_observed_YL1']}; "
        f"test raw-YL1 disagreement={parent_disagreement['test']['disagreement_rate']:.4%}; "
        f"min train taxonomy purity={taxonomy_audit['min_train_parent_purity']:.4f}",
        flush=True,
    )
    X=extract_embeddings(texts,out/"modernbert_base_meanpool_fp16.npy",args.batch_size,args.max_length,args.device,args.model_source,args.model_revision)
    pembed=fit_poincare(parent_of_leaf,out/f"poincare10_taxonomy_{taxonomy_hash}.npy")
    methods=["vanilla","hard_map","tree","poincare","ultrametric","geometric_shell"]
    rows=[]; tuning={}; diagnostics=[]
    for seed in SEEDS:
        print(f"seed {seed}",flush=True)
        model,Xn,mu,sd=train_heads(X,yp,y,train_idx,val_idx,seed,args.device,epochs=args.head_epochs,batch_size=args.head_batch)
        p_val,l_val=logits_all(model,Xn[val_idx],args.device)
        p_test,l_test=logits_all(model,Xn[test_idx],args.device)
        tp=fit_temperature(p_val,yp[val_idx]); tl=fit_temperature(l_val,y[val_idx])
        q_val=apply_temperature(p_val,tp); semantic_val=apply_temperature(l_val,tl)
        q_test=apply_temperature(p_test,tp); semantic_test=apply_temperature(l_test,tl)
        diagnostics.append({
            "seed": seed,
            "parent_head_accuracy": float(np.mean(q_test.argmax(axis=1) == yp[test_idx])),
            "parent_head_nll": nll(q_test, yp[test_idx]),
            "parent_head_ece": ece(q_test, yp[test_idx]),
            "semantic_leaf_accuracy": float(np.mean(semantic_test.argmax(axis=1) == y[test_idx])),
            "semantic_leaf_nll": nll(semantic_test, y[test_idx]),
            "semantic_leaf_ece": ece(semantic_test, y[test_idx]),
        })
        seed_tune={}
        for m in methods:
            seed_tune[m]=tune_method(m,semantic_val,q_val,y[val_idx],yp[val_idx],parent_of_leaf,pembed)
        tuning[str(seed)]={"parent_temperature":tp,"leaf_temperature":tl,"routing":seed_tune}
        regimes=[("exact",0.0,one_hot(yp[test_idx],7),np.ones(len(test_idx),bool)),
                 ("predicted",0.0,q_test,np.ones(len(test_idx),bool)),
                 ("missing",0.0,np.full_like(q_test,1/7),np.zeros(len(test_idx),bool))]
        # Use one fixed wrong-parent assignment per seed/sample across all eps values.
        corruption_seed = seed * 1000 + 17
        for eps in EPS_GRID:
            regimes.append(("corrupted",eps,corrupt_q(q_test,yp[test_idx],eps,corruption_seed),np.ones(len(test_idx),bool)))
        for regime,eps,q_eval,active in regimes:
            for m in methods:
                probs=route(m,semantic_test,q_eval,parent_of_leaf,seed_tune[m],pembed,active)
                met=leaf_metrics(probs,y[test_idx],yp[test_idx],parent_of_leaf)
                rows.append({"seed":seed,"method":m,"regime":regime,"epsilon":eps,**met})
    (out/"tuning.json").write_text(json.dumps(tuning,indent=2),encoding="utf-8")
    pd.DataFrame(diagnostics).to_csv(out/"head_diagnostics.csv", index=False)
    _,_,comparisons=write_outputs(rows,out)
    (out/"STATUS.txt").write_text("PASS\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","audit":audit,"comparisons":comparisons},indent=2))

if __name__=="__main__": main()
