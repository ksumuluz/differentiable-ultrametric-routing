# Theory verification

`verify_depth_sensitivity_theorem.py` is an independent executable check based on the equations printed in the current manuscript and supplement. It checks monotonicity, the condition-number identity, the stated bounds, and consecutive-difference identities.

`results/theory/reported_theorem_verification.json` is the historical summary reporting 3,888 evaluated points, zero monotonicity/bound violations, and the recorded worst relative identity error. The exact historical point coordinates are unavailable.

The independent verifier uses an explicitly stated 3,888-point grid and writes its outputs to `results/theory/` by default.
