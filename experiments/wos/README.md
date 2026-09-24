# WOS-46985 experiments

`run_experiment.py` implements the WOS-46985 external experiment described in the manuscript: public-data acquisition/audit, train-only taxonomy derivation, the fixed 70/15/15 split, pinned ModernBERT embedding extraction, ten paired head seeds, validation temperature calibration, routing evaluation, and corruption analyses.

`run_map_finite_bias_ablation.py` evaluates the matched MAP-plus-finite-bias control used to separate support hardness from posterior preservation.

`recompute_document_bootstrap.py` recomputes the document-level bootstrap from committed per-document outputs.

Example:

```bash
python experiments/wos/run_experiment.py --data-root /path/to/root-containing-WOS46985 --workdir wos_run
```
