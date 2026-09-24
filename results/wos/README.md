# WOS-46985 results

This directory combines the historical all-method seed-level outputs with the matched finite-bias ablation and document-bootstrap evidence used in the manuscript.

Core files:
- `dataset_audit.json` — dataset/split/taxonomy audit and pinned ModernBERT revision.
- `selected_hyperparameters.json` — validation-selected temperatures and routing parameters for all ten seeds.
- `results_per_seed.csv` — all-method seed-level metrics.
- `results_aggregate.csv`, `corruption_results.csv`, `pairwise_comparisons.json` — deterministic summaries.
- `map_finite_bias_per_seed.csv`, `map_finite_bias_aggregate.csv`, `map_finite_bias_comparisons.json` — hard MAP, matched MAP finite-bias, and posterior finite-bias comparison.
- `per_document_key_methods.csv.gz`, `document_bootstrap.csv/.json`, `per_document_differences.csv.gz` — document-level evidence and 20,000-resample bootstrap outputs.
