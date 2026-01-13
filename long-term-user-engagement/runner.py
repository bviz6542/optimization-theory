import os
import numpy as np
import matplotlib.pyplot as plt
import inspect

from obp.policy import (
    Random,
    EpsilonGreedy,
    BernoulliTS,
    LinUCB,
    LinTS,
)

from env_longterm import LongTermOBDEnv
from q_learning import TabularQLearning, EngagementStateEncoder, EngagementSensitivityEncoder, QuantileBinner


# ----------------------------
# Plot style + IO
# ----------------------------
def set_plot_style():
    plt.style.use("ggplot")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("=", "")
         .replace(",", "")
         .replace("(", "")
         .replace(")", "")
         .replace("×", "x")
         .replace(":", "")
    )


# ----------------------------
# Rollouts
# ----------------------------
def run_episode_bandit(env, policy):
    context = env.reset()
    rewards = []

    sel_sig = inspect.signature(policy.select_action)
    takes_context = (len(sel_sig.parameters) == 1)

    upd_sig = inspect.signature(policy.update_params)
    accepts_context = (len(upd_sig.parameters) == 3)

    for _ in range(env.horizon):
        if takes_context:
            a = int(policy.select_action(context.reshape(1, -1))[0])
        else:
            a = int(policy.select_action()[0])

        next_context, r, done = env.step(a)
        rewards.append(r)

        if accepts_context:
            policy.update_params(action=int(a), reward=float(r), context=context.reshape(1, -1))
        else:
            policy.update_params(action=int(a), reward=float(r))

        context = next_context
        if done:
            break

    return np.asarray(rewards, dtype=float)


def run_episode_qlearning(env, q_policy):
    context = env.reset()
    rewards = []

    for _ in range(env.horizon):
        a = q_policy.select_action(context)
        next_context, r, done = env.step(a)
        rewards.append(r)

        q_policy.update(context, a, r, next_context)
        context = next_context

        if done:
            break

    return np.asarray(rewards, dtype=float)


# ----------------------------
# Q policies
# ----------------------------
def build_q_policies(context_dim, calib_contexts, seed):
    rng = np.random.RandomState(seed)

    # random projections (no env leakage)
    v_e = rng.normal(0, 1.0, size=context_dim)
    v_s = rng.normal(0, 1.0, size=context_dim)

    # scores for calibration
    e_scores = calib_contexts @ v_e
    s_scores = calib_contexts @ v_s

    # 5 bins -> 4 thresholds (quantiles)
    e_thr = np.quantile(e_scores, [0.2, 0.4, 0.6, 0.8])
    s_thr = np.quantile(s_scores, [0.2, 0.4, 0.6, 0.8])

    enc_e = EngagementStateEncoder(proj_vec=v_e, binner=QuantileBinner(e_thr))
    enc_es = EngagementSensitivityEncoder(
        proj_e=v_e, proj_s=v_s,
        binner_e=QuantileBinner(e_thr),
        binner_s=QuantileBinner(s_thr),
    )

    q_list = []

    # E5 grid
    e5_grid = [
        (0.9,   0.10, 0.10),
        (0.995, 0.10, 0.10),
        (1.0,   0.10, 0.03),
        (0.995, 0.10, 0.03),
        (0.995, 0.10, 0.00),
    ]
    for g, a, eps in e5_grid:
        name = f"Q[E5] g={g} a={a} eps={eps}"
        q_list.append(TabularQLearning(
            encoder=enc_e,
            n_states=enc_e.n_bins,
            n_actions=2,
            gamma=g, alpha=a, epsilon=eps,
            seed=seed,
            name=name
        ))

    # ES25 grid
    es25_grid = [
        (0.90,  0.10, 0.10),
        (0.995, 0.10, 0.10),
        (1.0,   0.10, 0.03),
        (0.995, 0.10, 0.03),
        (0.995, 0.10, 0.00),
    ]
    for g, a, eps in es25_grid:
        name = f"Q[ES25] g={g} a={a} eps={eps}"
        q_list.append(TabularQLearning(
            encoder=enc_es,
            n_states=enc_es.n_e_bins * enc_es.n_s_bins,
            n_actions=2,
            gamma=g, alpha=a, epsilon=eps,
            seed=seed,
            name=name
        ))

    return q_list


# ----------------------------
# Plot helpers
# ----------------------------
def pad_and_mean(curves):
    max_T = max(len(c) for c in curves)
    padded = np.zeros((len(curves), max_T), dtype=float)
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c
        if len(c) < max_T:
            padded[i, len(c):] = c[-1]
    return padded.mean(axis=0)


def plot_curves(curves_dict, title, filename):
    plt.figure(figsize=(8, 5))
    for name, curve in curves_dict.items():
        x = np.arange(len(curve)) + 1
        plt.plot(x, curve, label=name)
    plt.xlabel("t")
    plt.ylabel("Average reward per step")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_state_value(q_policy, filename):
    Q = q_policy.Q
    V = Q.max(axis=1)
    enc = q_policy.encoder

    if isinstance(enc, EngagementStateEncoder):
        n_bins = enc.n_bins
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(n_bins), V)
        plt.xlabel("Engagement bin")
        plt.ylabel("V(s) = max_a Q(s,a)")
        plt.title(q_policy.name)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return

    if isinstance(enc, EngagementSensitivityEncoder):
        n_e = enc.n_e_bins
        n_s = enc.n_s_bins
        value_map = np.zeros((n_e, n_s), dtype=float)
        for state_id, v in enumerate(V):
            e_bin = state_id // n_s
            s_bin = state_id % n_s
            value_map[e_bin, s_bin] = v

        plt.figure(figsize=(6, 5))
        plt.imshow(value_map, origin="lower", aspect="auto", cmap="viridis")
        plt.colorbar(label="V(s) = max_a Q(s,a)")
        plt.xlabel("Sensitivity bin")
        plt.ylabel("Engagement bin")
        plt.title(q_policy.name)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return

    raise ValueError("Unknown encoder type")


def make_quantile_thresholds(contexts, n_bins):
    # n_bins=5 -> thresholds length 4
    qs = [i / n_bins for i in range(1, n_bins)]
    return np.quantile(contexts, qs)



# ----------------------------
# Main evaluation
# ----------------------------
def evaluate_policies(
    feedback,
    horizon=3000,
    n_episodes=20,
    seed=20251212,
    out_dir="out",
):
    set_plot_style()
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "state_value"))

    def make_env(run_seed):
        return LongTermOBDEnv(feedback=feedback, horizon=horizon, seed=run_seed)

    n_actions = int(feedback["n_actions"])
    context_dim = int(feedback["context"].shape[1])

    # ----------------------------
    # Bandit builders (epsilon sweeps)
    # ----------------------------
    eps_list = [0.10, 0.30, 1.00]
    bandit_builders = {
        "Random": lambda: Random(n_actions=n_actions, len_list=1, random_state=seed),
        "BernoulliTS": lambda: BernoulliTS(n_actions=n_actions, len_list=1, random_state=seed),
        "LinTS": lambda: LinTS(dim=context_dim, n_actions=n_actions, len_list=1, random_state=seed),
    }
    for eps in eps_list:
        bandit_builders[f"EpsGreedy eps={eps}"] = (lambda eps=eps: EpsilonGreedy(
            n_actions=n_actions, len_list=1, epsilon=eps, random_state=seed
        ))
        bandit_builders[f"LinUCB eps={eps}"] = (lambda eps=eps: LinUCB(
            dim=context_dim, n_actions=n_actions, len_list=1, epsilon=eps, random_state=seed
        ))

    results = {}

    # ----------------------------
    # Bandit evaluation
    # ----------------------------
    for name, builder in bandit_builders.items():
        all_cum = []
        for ep in range(n_episodes):
            env = make_env(seed + ep)
            policy = builder()
            rewards = run_episode_bandit(env, policy)
            all_cum.append(np.cumsum(rewards) / (np.arange(len(rewards)) + 1))
        results[name] = pad_and_mean(all_cum)
        print(f"[Bandit] {name}: final={results[name][-1]:.4f}")

    plot_curves(
        {k: v for k, v in results.items() if k.startswith("EpsGreedy")},
        title="Bandit: EpsilonGreedy epsilon sweep",
        filename=os.path.join(out_dir, "bandit_epsgreedy_sweep.png"),
    )
    plot_curves(
        {k: v for k, v in results.items() if k.startswith("LinUCB")},
        title="Bandit: LinUCB epsilon sweep",
        filename=os.path.join(out_dir, "bandit_linucb_sweep.png"),
    )

    # ----------------------------
    # Q-learning evaluation (NO env leakage, NO per-episode re-init)
    # ----------------------------
    q_results = {}  # name -> list[curve]

    # calibration contexts
    rng = np.random.RandomState(seed)
    ctx_all = feedback["context"]
    calib_n = min(200000, ctx_all.shape[0])
    calib_idx = rng.choice(ctx_all.shape[0], size=calib_n, replace=False)
    calib_contexts = ctx_all[calib_idx]

    # build Q policies once
    q_policies = build_q_policies(
        context_dim=context_dim,
        calib_contexts=calib_contexts,
        seed=seed,
    )

    for q_policy in q_policies:
        all_cum = []
        for ep in range(n_episodes):
            env = make_env(seed + ep)
            rewards = run_episode_qlearning(env, q_policy)
            all_cum.append(np.cumsum(rewards) / (np.arange(len(rewards)) + 1))

        results[q_policy.name] = pad_and_mean(all_cum)
        q_results[q_policy.name] = all_cum
        print(f"[Q] {q_policy.name}: final={results[q_policy.name][-1]:.4f}")

    # Q compare plot (ALL Q policies)
    q_curves = {name: results[name] for name in q_results.keys()}
    plot_curves(
        q_curves,
        title="Q-learning: hyperparam/state comparison (mean over episodes)",
        filename=os.path.join(out_dir, "q_compare_all.png"),
    )

    # ----------------------------
    # State value plots (one per Q policy)
    # ----------------------------
    for q_policy in q_policies:
        fn = os.path.join(out_dir, "state_value", f"{safe_filename(q_policy.name)}.png")
        plot_state_value(q_policy, fn)
        print(f"[PLOT] {fn}")

    # ----------------------------
    # Final performance plot (Bandits + Best Q only)
    # ----------------------------
    best_q_name = max(q_results.keys(), key=lambda n: results[n][-1])
    best_q_curve = results[best_q_name]
    print(f"[BEST Q] {best_q_name}: final={best_q_curve[-1]:.4f}")

    results_best = {k: v for k, v in results.items() if k not in q_results}
    results_best[f"BestQ: {best_q_name}"] = best_q_curve

    plot_curves(
        results_best,
        title="Online performance (Bandit baselines + Best Q-learning)",
        filename=os.path.join(out_dir, "performance_all.png"),
    )
