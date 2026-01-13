import numpy as np
from obp.dataset import OpenBanditDataset


def load_obd():
    dataset = OpenBanditDataset(
        behavior_policy="bts",
        campaign="all",
        data_path="./open_bandit_dataset",
    )
    feedback = dataset.obtain_batch_bandit_feedback()
    print(f"[INFO] rounds={feedback['n_rounds']}  actions={feedback['n_actions']}")
    feedback["position"] = np.zeros_like(feedback["action"], dtype=int)
    return feedback
