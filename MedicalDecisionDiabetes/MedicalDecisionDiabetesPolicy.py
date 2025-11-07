import math
from typing import Dict

import numpy as np

from additional_params import Policy
from state_params import State


class MedicalDecisionDiabetesPolicy:
    def __init__(self, seed):
        self.prng = np.random.RandomState(seed)

    def upper_confidence_bound(self, model_curr, theta) -> State:
        """
        X_UCB = argmax_x ( mu_hat_x + theta * sqrt(log(t+1) / (N_x + 1)) )
        Can't implement this at time t=0 (from t=1 at least).
        Can't divide by zero, which means we need each drug to have been tested at least once.
        Note that state has a list of 3 entries, for each key(type of drug) in the dictionary
        {"drug" : [mu_empirical, beta, number of times drug given to patient]}
        """
        t = model_curr.t
        stats: Dict[State, float] = {}
        for s in State:
            mu_hat, _, n_x = model_curr.state[s]
            n_x_safe = 1 + n_x  # non zero denominator trick
            bonus = math.sqrt(math.log(t + 1) / n_x_safe)
            stats[s] = mu_hat + theta * bonus
        return max(stats, key=stats.get) # optimal decision

    def interval_estimation(self, model_curr, theta) -> State:
        """
        X_IE = argmax_x(mu_hat_x + theta * sigma_hat_x)
        sigma_hat_x = 1 / sqrt(beta_hat_x)
        """
        stats: Dict[State, float] = {}
        for s in State:
            mu_hat, beta, _n_x = model_curr.state[s]
            sigma_hat = 1.0 / math.sqrt(beta)
            stats[s] = mu_hat + theta * sigma_hat
        return max(stats, key=stats.get)

    def pure_exploitation(self, model_curr) -> State:
        """drug with the highest mu_hat every time (theta = 0)"""
        stats = {s: model_curr.state[s][0] for s in State}
        return max(stats, key=stats.get)

    def pure_exploration(self) -> State:
        """random drug every time"""
        return self.prng.choice(list(State))

    def decide_drug(self, policy: Policy, model_curr, theta) -> State:
        if policy == Policy.UCB:
            return self.upper_confidence_bound(model_curr, theta)
        elif policy == Policy.IE:
            return self.interval_estimation(model_curr, theta)
        elif policy == Policy.PURE_EXPLOITATION:
            return self.pure_exploitation(model_curr)
        elif policy == Policy.PURE_EXPLORATION:
            return self.pure_exploration()
        else:
            raise NotImplementedError
