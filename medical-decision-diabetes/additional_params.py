from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, List

class Config(Enum):
    SIGMA_W = "sigma_w"
    N = "N"
    L = "L"
    THETA_START = "theta_start"
    THETA_END = "theta_end"
    INCREMENT = "increment"
    TRUTH_TYPE = "truth_type"
    POLICY = "policy"

class TruthType(Enum):
    KNOWN = "known"
    FIXED_UNIFORM = "fixed_uniform"
    PRIOR_UNIFORM = "prior_uniform"
    NORMAL = "normal"

class Policy(Enum):
    IE = "IE"
    UCB = "UCB"
    PURE_EXPLOITATION = "PureExploitation"
    PURE_EXPLORATION = "PureExploration"

@dataclass
class ConfigParams:
    sigma_w: float
    N: int
    L: int
    theta_start: float
    theta_end: float
    increment: float
    truth_type: TruthType
    policy: List[Policy]

def config_from_dataclass(params: ConfigParams) -> Dict[Config, Any]:
    raw = asdict(params)
    return {
        Config(k): (
            [p if isinstance(p, Enum) else Policy(p) for p in v] if k == "policy" else v
        )
        for k, v in raw.items()
    }

CONFIG: Dict[Config, Any] = config_from_dataclass(ConfigParams(
    sigma_w=0.5,
    N=20,
    L=1000,
    theta_start=0.0,
    theta_end=2.1,
    increment=0.2,
    truth_type=TruthType.FIXED_UNIFORM,
    policy=[Policy.IE],
))
