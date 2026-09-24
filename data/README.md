# Data

The raw WOS-46985 corpus is not redistributed in this repository. `experiments/wos/run_experiment.py` accepts a local `WOS46985/` directory and also contains public-source acquisition fallbacks.

The historical frozen ModernBERT embedding cache is not committed. It can be regenerated from the pinned model revision recorded in `results/wos/dataset_audit.json`. Exact bitwise equality of a regenerated cache can depend on software/hardware details, so the repository does not claim the regenerated cache is byte-identical to the historical cache.
