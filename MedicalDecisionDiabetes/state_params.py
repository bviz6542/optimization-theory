from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict

class State(Enum):
    M = "M"         # Metformin
    SENS = "Sens"   # Sensitizer
    SECR = "Secr"   # Secretagoge
    AGI = "AGI"     # Alpha - glucosidase inhibitor
    PA = "PA"       # Peptide analog

class Param(Enum):
    MU_0 = "mu_0"
    SIGMA_0 = "sigma_0"
    MU_TRUTH = "mu_truth"
    SIGMA_TRUTH = "sigma_truth"
    MU_FIXED = "mu_fixed"
    FIXED_UNIFORM_A = "fixed_uniform_a"
    FIXED_UNIFORM_B = "fixed_uniform_b"
    PRIOR_MULT_A = "prior_mult_a"
    PRIOR_MULT_B = "prior_mult_b"

@dataclass
class Params:
    mu_0: float
    sigma_0: float
    mu_truth: float
    sigma_truth: float
    mu_fixed: float
    fixed_uniform_a: float
    fixed_uniform_b: float
    prior_mult_a: float
    prior_mult_b: float

def _row(p: Params) -> Dict[Param, float]:
    d = asdict(p)
    return {Param(k): float(v) for k, v in d.items()}

PARAMS: Dict[State, Dict[Param, float]] = {
    State.M: _row(Params(0.32, 0.12, 0.25, 0.0, 0.3, -0.15, 0.15, -0.5, 0.5)),
    State.SENS: _row(Params(0.28, 0.19, 0.30, 0.0, 0.3, -0.15, 0.15, -0.5, 0.5)),
    State.SECR: _row(Params(0.30, 0.17, 0.28, 0.0, 0.3, -0.15, 0.15, -0.5, 0.5)),
    State.AGI: _row(Params(0.26, 0.15, 0.34, 0.0, 0.3, -0.15, 0.15, -0.5, 0.5)),
    State.PA: _row(Params(0.21, 0.21, 0.24, 0.0, 0.3, -0.15, 0.15, -0.5, 0.5)),
}
