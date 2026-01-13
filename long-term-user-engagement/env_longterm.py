import numpy as np


class LongTermOBDEnv:
    """
    Long-term recommendation environment built on top of Open Bandit Dataset (OBD) contexts.

    목적
    ----
    - OBD의 (user/item 기반) context를 관측으로 사용하고,
    - 관측되지 않는 latent user engagement(숨은 상태)를 두어,
      "단기 클릭 vs 장기 관계(engagement)"의 trade-off를 가지는
      장기 추천(Long-term) 시뮬레이션 환경을 만든다.

    관측/상태/행동/보상
    ------------------
    - observation: context 벡터 x_t  (OBD에서 샘플링)
    - hidden state: user_engagement e_t in [0,1]
    - action: 추천 행동 a_t (0..n_actions-1)
      * 여기서는 단순화해서 action parity(짝/홀)에 따라 추천 스타일이 바뀐다.
    - reward: 클릭 r_t ∈ {0,1}, Bernoulli(p_click)

    Tabular Q-learning을 위해서는 state를 engagement bin으로 discretize 하여
    s_t = discretize(e_t) 를 제공한다.
    """

    def __init__(
        self,
        feedback,
        horizon=2000,
        seed=42,
        n_engagement_bins=5,
        p_min=0.01,
        p_max=0.80,
    ):
        """
        Parameters
        ----------
        feedback : dict
            OpenBanditDataset.obtain_batch_bandit_feedback() 결과.
            최소한 다음 키가 필요:
            - feedback["context"] : shape (n_rounds, context_dim)
            - feedback["n_actions"] : int
        horizon : int
            한 에피소드의 최대 길이(= step 호출 최대 횟수).
        seed : int
            랜덤 시드.
        n_engagement_bins : int
            tabular state id를 만들기 위한 engagement discretization bins.
        p_min, p_max : float
            클릭 확률 p_click을 안정적으로 유지하기 위한 clipping 범위.
        """
        # ---- logged bandit data (OBD) ----
        self.feedback = feedback
        self.contexts = feedback["context"]
        self.n_rounds, self.context_dim = self.contexts.shape
        self.n_actions = int(feedback["n_actions"])

        # ---- episode / state parameters ----
        self.horizon = int(horizon)
        self.n_engagement_bins = int(n_engagement_bins)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        # ---- RNG ----
        self.rng = np.random.RandomState(seed)

        # ---- user heterogeneity (latent parameters) ----
        # 초기 engagement 및 프로모션 민감도(sensitivity)를 context로부터 만들기 위한 weight
        self.w_init_engagement = self.rng.normal(0, 0.1, size=self.context_dim)
        self.w_engagement_sensitivity = self.rng.normal(0, 0.1, size=self.context_dim)

        # internal state (t, current_idx, user_engagement) 초기화
        self.reset()

    # =========================================================
    # Low-level helpers
    # =========================================================
    def _sample_context_index(self) -> int:
        """OBD 로그에서 다음 라운드 인덱스를 uniform하게 샘플링."""
        return int(self.rng.randint(self.n_rounds))

    def _observe_context(self) -> np.ndarray:
        """현재 인덱스에 해당하는 context 관측."""
        return self.contexts[self.current_idx]

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically simple sigmoid."""
        return float(1.0 / (1.0 + np.exp(-x)))

    # =========================================================
    # Reset
    # =========================================================
    def reset(self):
        """
        새로운 에피소드를 시작한다.

        Returns
        -------
        obs : np.ndarray
            초기 관측 context 벡터 x_0
        state_id : int
            tabular 알고리즘을 위한 상태 id (discretized engagement)
        """
        self.t = 0
        self.current_idx = self._sample_context_index()

        context = self._observe_context()

        # 초기 engagement는 context의 선형 결합을 sigmoid로 squash하여 만든다.
        raw = float(context @ self.w_init_engagement)
        self.user_engagement = self._sigmoid(raw)

        return context

    # =========================================================
    # Action semantics (recommendation style)
    # =========================================================
    @staticmethod
    def _is_promotion_focused(action: int) -> bool:
        """
        action을 두 가지 추천 스타일 중 하나로 맵핑한다.

        - promotion_focused (짝수 action): 단기 클릭을 얻기 쉬우나 장기 engagement에 악영향(피로)
        - engagement_focused (홀수 action): 단기 클릭은 약하지만 장기 engagement를 키울 수 있음
        """
        return (int(action) % 2) == 0

    # =========================================================
    # Reward model
    # =========================================================
    @staticmethod
    def _base_click_probability(engagement: float, promotion_focused: bool) -> float:
        """
        engagement와 추천 스타일로부터 (stationary) 클릭 확률을 만든다.

        design intuition (제출용 설명)
        -----------------------------
        - promotion_focused:
            engagement가 낮아도 어느 정도 클릭을 얻지만, engagement가 높아질수록 효율이 감소하도록 설계
        - engagement_focused:
            engagement가 높을수록 클릭이 점점 잘 나오도록 설계 (장기 관계형 추천)
        """
        if promotion_focused:
            base_p = 0.10 + 0.05 * (1.0 - engagement)
        else:
            base_p = 0.02 + 0.30 * engagement

        return float(base_p)

    def _sample_reward(self, p_click: float) -> int:
        """Bernoulli(p_click)로 클릭(0/1) 샘플링."""
        return int(self.rng.binomial(1, p_click))

    # =========================================================
    # Latent engagement dynamics (state transition)
    # =========================================================
    @staticmethod
    def _update_user_engagement(
        engagement: float,
        promotion_focused: bool,
        reward: int,
        sensitivity: float,
    ) -> float:
        """
        hidden state e_t = user_engagement 의 전이 규칙.

        sensitivity는 "프로모션 과다 노출에 대한 사용자 민감도"로 해석:
        - sensitivity가 클수록 promotion_focused의 피로(engagement 감소)가 더 커진다.
        """
        if promotion_focused:
            # 프로모션 추천: 클릭이 나와도 engagement가 감소(피로 누적)하도록 설계
            if reward:
                engagement -= 0.20 + 0.40 * sensitivity
            else:
                engagement -= 0.30 + 0.50 * sensitivity
        else:
            # 관계형 추천: 클릭이 나오면 engagement 상승, 클릭이 없으면 소폭 감소
            if reward:
                engagement += 0.25 + 0.35 * (1.0 - sensitivity)
            else:
                engagement -= 0.02 + 0.05 * sensitivity

        return float(np.clip(engagement, 0.0, 1.0))

    def _compute_sensitivity(self, context: np.ndarray) -> float:
        """
        context-dependent sensitivity 추정:
        - context의 선형 결합을 sigmoid로 squash하여 [0,1] 범위로 만든다.
        """
        raw = float(context @ self.w_engagement_sensitivity)
        return self._sigmoid(raw)

    # =========================================================
    # Step
    # =========================================================
    def step(self, action: int):
        """
        환경을 한 스텝 진행한다.

        진행 흐름 (제출용, 읽기 쉽게)
        -----------------------------
        1) 현재 관측 context x_t 와 hidden engagement e_t 읽기
        2) action -> 추천 스타일(promotion vs engagement) 결정
        3) (e_t, style)로 클릭 확률 p_click 계산 후 clipping
        4) Bernoulli(p_click)로 reward(클릭) 샘플링
        5) context로 사용자 sensitivity 추정
        6) (style, reward, sensitivity)에 따라 engagement 업데이트 (state transition)
        7) 다음 context를 OBD에서 샘플링하고 time 증가
        8) 종료 조건(done) 계산 및 (obs, state_id, reward, done) 반환

        Returns
        -------
        next_obs : np.ndarray
            다음 관측 context x_{t+1}
        next_state_id : int
            discretized engagement state id (tabular methods)
        reward : int
            클릭 (0 or 1)
        done : bool
            horizon 도달 여부
        """
        # (1) observe
        context = self._observe_context()
        engagement = float(self.user_engagement)

        # (2) action semantics
        promotion_focused = self._is_promotion_focused(action)

        # (3) click probability
        p_click = self._base_click_probability(engagement, promotion_focused)
        p_click = float(np.clip(p_click, self.p_min, self.p_max))

        # (4) reward sampling
        reward = self._sample_reward(p_click)

        # (5) user sensitivity (context-dependent)
        sensitivity = self._compute_sensitivity(context)

        # (6) hidden engagement transition
        self.user_engagement = self._update_user_engagement(
            engagement=engagement,
            promotion_focused=promotion_focused,
            reward=reward,
            sensitivity=sensitivity,
        )

        # (7) advance time + next context
        self.current_idx = self._sample_context_index()
        self.t += 1

        # (8) termination + return
        done = bool(self.t >= self.horizon)
        next_context = self._observe_context()

        return next_context, reward, done

    # =========================================================
    # Properties (for agents)
    # =========================================================
    @property
    def obs_dim(self) -> int:
        """dimension of observation vector (context)."""
        return int(self.context_dim)

    @property
    def num_states(self) -> int:
        """number of discrete states for tabular Q-learning."""
        return int(self.n_engagement_bins)
