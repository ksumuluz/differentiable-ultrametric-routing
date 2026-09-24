# Validation

The following checks were run while synchronizing the public reproducibility repository with the 24 September 2026 submission files.

- The current main manuscript and supplement PDFs are stored under stable repository filenames and match the synchronized LaTeX sources.
- `paper/main_manuscript.tex` compiles successfully with the included figure PDFs to a 23-page manuscript; extracted prose, equations, tables, numerical results, declarations, funding, and availability text were checked against the current submission PDF.
- `paper/supplementary_material.tex` compiles successfully with the included figure PDFs to a 5-page supplement; theorem caveats, paired-risk sign convention, numerical values, and reproducibility-package wording were synchronized with the current supplement.
- The committed controlled/theory result artifacts were loaded successfully.
- `exact_provenance_results.json` records 10 seeds and 300 traces per seed and contains the reported exact-provenance method summaries.
- `uncertain_provenance_results.json` records 10 seeds, 800 traces per seed, 30% noisy provenance, 35% missing provenance, and the operational salience sweep.
- `reported_theorem_verification.json` records 3,888 samples, zero telescoping/exponential monotonicity violations, zero bound violations, and the reported worst relative identity error.
- `difficulty_continuum_results.json` contains the five-seed pressure continuum used for the controlled difficulty figure.
- The controlled result-regeneration utility completes successfully against the committed JSON files.
- The independent theorem-verification script completes successfully with zero monotonicity and bound violations.
- The WOS all-method per-seed output contains the previously validated ten-seed results and the matched-ablation/document-bootstrap files reproduce the reported results; a redundant seed-summary file with a misleading bootstrap-oriented filename was removed.
- All Python files in `experiments/` and `theory/` pass syntax compilation.
- Repository filenames and directories use scientific labels (`controlled`, `wos`, `theory`) rather than internal development-stage labels.
- The manuscript and supplement contain the public GitHub URL and describe reproducibility artifacts consistently with the repository contents.
