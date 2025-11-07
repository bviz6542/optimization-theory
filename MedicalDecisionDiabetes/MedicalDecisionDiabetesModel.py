import numpy as np

from MedicalDecisionDiabetes.additional_params import CONFIG, Config, TruthType
from MedicalDecisionDiabetes.state_params import State, PARAMS, Param


class MedicalDecisionDiabetesModel:
    def __init__(self, seed):
        self.prng = np.random.RandomState(seed)

        self.state: dict[State, list[float]] = {
            s: [PARAMS[s][Param.MU_0], self.beta(PARAMS[s][Param.SIGMA_0]), 0]
            for s in State
        } # {"drug": [mu_empirical, beta, number of times drug given to patient]}
        self.obj = 0.0
        self.obj_sum = 0.0
        self.mu = {}  # updated using "exog_info_sample_mu" at the beginning of each sample path
        self.t = 0  # time counter (in months)

        self.truth_params_dict = {}
        if CONFIG[Config.TRUTH_TYPE] == TruthType.FIXED_UNIFORM:
            self.truth_params_dict = {
                s: [PARAMS[s][Param.MU_FIXED], PARAMS[s][Param.FIXED_UNIFORM_A], PARAMS[s][Param.FIXED_UNIFORM_B]]
                for s in list(State)
            }
        elif CONFIG[Config.TRUTH_TYPE] == TruthType.PRIOR_UNIFORM:
            self.truth_params_dict = {
                s: [PARAMS[s][Param.MU_0], PARAMS[s][Param.PRIOR_MULT_A], PARAMS[s][Param.PRIOR_MULT_B]]
                for s in list(State)
            }
        else:
            self.truth_params_dict = {
                s: [PARAMS[s][Param.MU_TRUTH], PARAMS[s][Param.SIGMA_TRUTH], 0]
                for s in list(State)
            }

    def exog_info_sample_mu(self):
        if CONFIG[Config.TRUTH_TYPE] == TruthType.KNOWN:
            self.mu = {
                x: self.truth_params_dict[x][0]
                for x in list(State)
            }
        elif CONFIG[Config.TRUTH_TYPE] == TruthType.FIXED_UNIFORM:
            self.mu = {
                x: self.truth_params_dict[x][0] + self.prng.uniform(self.truth_params_dict[x][1], self.truth_params_dict[x][2])
                for x in list(State)
            }
        elif CONFIG[Config.TRUTH_TYPE] == TruthType.PRIOR_UNIFORM:
            self.mu = {
                x: self.truth_params_dict[x][0] + self.prng.uniform(
                    self.truth_params_dict[x][1] * self.truth_params_dict[x][0],
                    self.truth_params_dict[x][2] * self.truth_params_dict[x][0]
                )
                for x in list(State)
            }
        else:
            self.mu = {
                x: self.prng.normal(self.truth_params_dict[x][0], self.truth_params_dict[x][1])
                for x in list(State)
            }

    def exog_info_fn(self, decision: State) -> tuple[float, float, float]:
        """
        Gives the exogenous information that is dependent on a random process
        W^(n+1) = mu_x + eps^(n+1)
        eps^(n+1) is normally distributed with mean 0 and known variance
        W^(n+1)_x : reduction in A1C level
        """
        w = self.prng.normal(self.mu[decision], CONFIG[Config.SIGMA_W])
        beta_w = self.beta(CONFIG[Config.SIGMA_W])
        mu = self.mu[decision]
        return w, beta_w, mu

    def transition_fn(self, decision: State, exog_info: tuple[float, float, float]) -> dict[State, list[float]]:
        """
        update only the state of chosen drug
        exog_info = (w, beta_w, mu)
        """
        w, beta_w, _ = exog_info

        mu_prev, beta_prev, n_prev = self.state[decision]

        beta_new = beta_prev + beta_w
        mu_new = (beta_prev * mu_prev + beta_w * w) / beta_new
        n_new = n_prev + 1

        return {decision: [mu_new, beta_new, n_new]}

    def objective_fn(self, exog_info: tuple[float, float, float]) -> float:
        _, _, mu = exog_info
        return mu

    def step(self, decision: State)-> tuple[float, float, float]:
        """Performs one simulation step."""
        exog_info = self.exog_info_fn(decision)
        updated = self.transition_fn(decision, exog_info)

        self.obj += self.objective_fn(exog_info)

        self.state[decision] = updated[decision]

        self.t += 1

        return exog_info

    @staticmethod
    def beta(sigma):
        """returns precision (beta), given the standard deviation (sigma)"""
        return 1 / sigma ** 2
