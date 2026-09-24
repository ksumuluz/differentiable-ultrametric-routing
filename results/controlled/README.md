# Controlled-experiment results

- `exact_provenance_results.json` — ten held-out seeds, 300 traces per seed; includes vanilla, Euclidean, tree, fitted 10D Poincare, shuffled-taxonomy, fixed-ultrametric, tuned prefix-ultrametric, and normalized-hard results.
- `uncertain_provenance_results.json` — ten held-out seeds, 800 traces per seed for noisy/missing provenance plus the foreign-salience sweep.
- `difficulty_continuum_results.json` — five-seed structural-conflict-pressure continuum.

These are the machine-readable results used for the controlled results reported in the manuscript.

Reviewer-facing metadata note: a descriptive internal label in the source exact-provenance JSON was normalized to “controlled-experiment”; numerical fields are unchanged. The SHA-256 of the source artifact before this metadata-only normalization was `55b6d7fb22cf1e03687a991452466e555187e469d080a3e8f4007fe719a19df4`.
