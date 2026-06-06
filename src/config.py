"""Central configuration: paths, seed, and model hyperparameters.

Pipeline-level knobs live here so every run is inspectable and reproducible.
Domain-structure constants (elasticity utilization edges, tariff caps, monitor
weights) stay with their own modules where they are semantically owned.
"""
from __future__ import annotations

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets OP_26 Analytics"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

# Demand model (gradient boosting) hyperparameters.
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                   subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, random_state=SEED)


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy RNGs for reproducible runs."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
