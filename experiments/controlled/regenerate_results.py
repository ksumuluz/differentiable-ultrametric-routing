#!/usr/bin/env python3
"""Regenerate reviewer-facing tables/figures from committed controlled results."""
from pathlib import Path
import argparse, csv, json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "controlled"
DEFAULT_THEORY = ROOT / "results" / "theory" / "reported_theorem_verification.json"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--theory-result", type=Path, default=DEFAULT_THEORY)
    ap.add_argument("--outdir", type=Path, default=Path("controlled_regenerated"))
    args=ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

    can=load(args.results_dir/"exact_provenance_results.json")
    unc=load(args.results_dir/"uncertain_provenance_results.json")
    dif=load(args.results_dir/"difficulty_continuum_results.json")
    th=load(args.theory_result)
    worst=th[0]["worst"] if isinstance(th,list) else th["worst"]
    assert worst["mono_tel"]==0 and worst["mono_exp"]==0 and worst["bound_viol"]==0

    with open(args.outdir/"exact_provenance_table.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["method","mean","sd","ci_low","ci_high"])
        for r in can["summary"]: w.writerow([r["method"],r["mean"],r["sd"],*r["ci95"]])

    with open(args.outdir/"uncertain_provenance_table.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["mode","method","path_mean","path_sd","cat_mean","cat_sd"])
        for r in unc["summary"]:
            if r["method"] in ("hard_map","soft_exp","soft_tel"):
                w.writerow([r["mode"],r["method"],r["path_acc_mean"],r["path_acc_sd"],
                            r["catastrophic_contamination_rate_mean"],r["catastrophic_contamination_rate_sd"]])

    labels=[r["method"] for r in can["summary"]]; means=[r["mean"] for r in can["summary"]]; errs=[r["sd"] for r in can["summary"]]
    plt.figure(figsize=(8.5,4.4)); plt.bar(range(len(means)),means,yerr=errs)
    plt.xticks(range(len(labels)),labels,rotation=35,ha="right"); plt.ylabel("Full-path accuracy")
    plt.tight_layout(); plt.savefig(args.outdir/"exact_provenance_regenerated.pdf"); plt.close()

    rows=dif["rows"] if isinstance(dif,dict) else dif
    plt.figure(figsize=(6.4,4.2))
    for m in ["vanilla","tree","ultra","hard_norm_boundary"]:
        rr=sorted([x for x in rows if x["method"]==m], key=lambda x:x.get("pressure",x.get("lambda",0)))
        xs=[x.get("pressure",x.get("lambda")) for x in rr]; ys=[x.get("mean",x.get("path_acc_mean")) for x in rr]
        if rr: plt.plot(xs,ys,marker="o",label=m)
    plt.xlabel("Structural-conflict pressure"); plt.ylabel("Full-path accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(args.outdir/"difficulty_continuum_regenerated.pdf"); plt.close()

    report={"theorem_result_check":True,"exact_provenance_rows":len(can["summary"]),
            "uncertain_provenance_rows":len(unc["summary"]),"status":"PASS"}
    (args.outdir/"audit_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
