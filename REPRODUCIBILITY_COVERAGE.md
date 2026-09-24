# Reproducibility coverage

| Manuscript component | Repository evidence | Status |
|---|---|---|
| Main manuscript | `paper/main_manuscript.pdf` + `paper/main_manuscript.tex` | Current submission PDF + editable synchronized source |
| Supplement | `paper/supplementary_material.pdf` + `paper/supplementary_material.tex` | Current submission PDF + editable synchronized source |
| Bibliography | `paper/references.bib` | Present |
| Figure source PDFs | `paper/figures/` | Seven files present |
| Controlled exact-provenance results | `results/controlled/exact_provenance_results.json` | Final machine-readable results present |
| Controlled uncertain-provenance results | `results/controlled/uncertain_provenance_results.json` | Final machine-readable results present |
| Difficulty continuum | `results/controlled/difficulty_continuum_results.json` | Final machine-readable results present |
| Controlled orchestration code used to generate final JSON | none | Not included / not claimed |
| Controlled table/figure regeneration | `experiments/controlled/regenerate_results.py` | Recomputable from committed final JSON |
| Reported 3,888-point theorem verification | `results/theory/reported_theorem_verification.json` | Reported summary present |
| Independent theorem execution | `theory/verify_depth_sensitivity_theorem.py` | Independent executable check |
| Independent theorem grid output | `results/theory/independent_verification_grid.csv` | Present |
| Exact theorem point coordinates for the reported check | none | Not available |
| WOS dataset/split/taxonomy audit | `results/wos/dataset_audit.json` | Audit present |
| Pinned ModernBERT revision | audit + WOS code | Present |
| Embedding-extraction configuration | audit + WOS code | Present |
| Validation-selected WOS parameters | `results/wos/selected_hyperparameters.json` | Present for all ten seeds |
| All-method WOS seed results | `results/wos/results_per_seed.csv` | Present |
| Matched MAP finite-bias ablation | `results/wos/map_finite_bias_*` | Present |
| WOS document bootstrap | `results/wos/document_bootstrap.*` + per-document data | Present and recomputable |
| WOS corruption results | `results/wos/corruption_results.csv` | Present |
| Frozen ModernBERT embedding cache | regenerable | Cache not committed |
| Main-manuscript LaTeX source | `paper/main_manuscript.tex` | Present and compiles to 23 pages |
| Supplement LaTeX source | `paper/supplementary_material.tex` | Present and compiles to 5 pages |
| Wrong-vote supplemental raw/code | none | Not available |
| Secondary pretrained-synthetic control raw/code | none | Not available |
