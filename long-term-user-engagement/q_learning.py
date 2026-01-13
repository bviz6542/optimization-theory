import numpy as np


class TabularQLearning:
    """Tabular Q-learning with discrete state encoder over continuous contexts."""

    def __init__(
        self,
        encoder,
        n_states,
        n_actions,
        gamma=0.95,
        alpha=0.1,
        epsilon=0.1,
        seed=42,
        name=None,
    ):
        self.encoder = encoder  # maps high-dim context -> discrete state id
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.rng = np.random.RandomState(seed)

        self.state_visits = np.zeros(self.n_states, dtype=int)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.name = name or f"Q(g={gamma},a={alpha},e={epsilon})"

    def select_action(self, context):
        s = self.encoder.encode(context)  # discretized latent state (state abstraction)
        self.state_visits[s] += 1

        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.n_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, context, action, reward, next_context):
        s = self.encoder.encode(context)
        s_next = self.encoder.encode(next_context)

        best_next = float(np.max(self.Q[s_next]))
        td_target = float(reward) + self.gamma * best_next  # Bellman target
        td_error = td_target - self.Q[s, int(action)]
        self.Q[s, int(action)] += self.alpha * td_error


class QuantileBinner:
    """Discretize a scalar score using empirical quantile thresholds."""

    def __init__(self, thresholds):
        self.thresholds = np.asarray(thresholds, dtype=float)

    def bin(self, x: float) -> int:
        return int(np.searchsorted(self.thresholds, x, side="right"))

    @property
    def n_bins(self) -> int:
        return int(len(self.thresholds) + 1)


class EngagementStateEncoder:
    """Encode context into engagement-level state via random projection + quantile binning."""

    def __init__(self, proj_vec, binner: QuantileBinner):
        self.v = np.asarray(proj_vec, dtype=float)
        self.binner = binner

    def encode(self, context):
        score = float(context @ self.v)  # proxy score for latent engagement
        return int(np.clip(self.binner.bin(score), 0, self.binner.n_bins - 1))

    @property
    def n_bins(self):
        return self.binner.n_bins


class EngagementSensitivityEncoder:
    """Joint state encoder capturing (engagement, sensitivity) as a 2D discrete grid."""

    def __init__(self, proj_e, proj_s, binner_e: QuantileBinner, binner_s: QuantileBinner):
        self.v_e = np.asarray(proj_e, dtype=float)
        self.v_s = np.asarray(proj_s, dtype=float)
        self.binner_e = binner_e
        self.binner_s = binner_s

    def encode(self, context):
        e_score = float(context @ self.v_e)
        s_score = float(context @ self.v_s)

        e_bin = int(np.clip(self.binner_e.bin(e_score), 0, self.binner_e.n_bins - 1))
        s_bin = int(np.clip(self.binner_s.bin(s_score), 0, self.binner_s.n_bins - 1))
        return int(e_bin * self.binner_s.n_bins + s_bin)  # flatten 2D -> 1D state id

    @property
    def n_e_bins(self):
        return self.binner_e.n_bins

    @property
    def n_s_bins(self):
        return self.binner_s.n_bins
