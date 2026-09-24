# Reproducibility materials

This repository accompanies **“Differentiable Ultrametric Routing for Hierarchical Evidence under Uncertain Provenance.”** The layout follows the scientific structure of the manuscript rather than internal development stages.

Public repository: https://github.com/ksumuluz/differentiable-ultrametric-routing

Repository snapshot synchronized with the submission manuscripts on **24 September 2026**.

## Repository map

- `paper/` — current submission manuscript PDFs, synchronized editable LaTeX sources for the main manuscript and supplement, bibliography, figure PDFs, and searchable text exports.
- `experiments/controlled/` — regeneration utility for the controlled-experiment tables and figures from the committed machine-readable results.
- `experiments/wos/` — WOS-46985 external experiment, matched finite-bias ablation, and bootstrap recomputation.
- `theory/` — executable depth-sensitivity theorem verification.
- `results/controlled/` — final machine-readable controlled-experiment results.
- `results/wos/` — WOS seed-level, matched-ablation, corruption, and document-bootstrap results.
- `results/theory/` — reported theorem-verification summary plus an independent executable grid check.

The controlled experiments use synthetic hierarchical data by design to isolate provenance, evidence salience, and branch membership; the external assay uses real WOS-46985 scientific abstracts with frozen ModernBERT representations.

## Quick checks

Regenerate reviewer-facing controlled tables/figures from the committed final JSON results:

```bash
python experiments/controlled/regenerate_results.py --outdir controlled_regenerated
```

Run the independent theorem check:

```bash
python theory/verify_depth_sensitivity_theorem.py --outdir theory_check
```

Recompute the WOS document bootstrap without model retraining:

```bash
python experiments/wos/recompute_document_bootstrap.py \
  --input results/wos/per_document_key_methods.csv.gz \
  --outdir bootstrap_check
```

Run the WOS external pipeline (raw WOS data may be local or acquired by the script):

```bash
python experiments/wos/run_experiment.py \
  --data-root /path/to/root-containing-WOS46985 \
  --workdir wos_run
```

## Provenance notes

The final controlled machine-readable results are preserved directly. The exact orchestration code that produced those JSON files is not included; the repository provides the results and a separate table/figure regeneration utility.

`paper/main_manuscript.tex` and `paper/supplementary_material.tex` are synchronized editable submission sources for the current 23-page main manuscript and 5-page supplement.

See `REPRODUCIBILITY_COVERAGE.md`, `VALIDATION.md`, and `KNOWN_LIMITATIONS.md` for exact coverage.
