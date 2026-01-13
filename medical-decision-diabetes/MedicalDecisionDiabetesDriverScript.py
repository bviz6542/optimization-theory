from collections import Counter
from copy import copy
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from additional_params import CONFIG, Config, Policy
from state_params import State
from MedicalDecisionDiabetesModel import MedicalDecisionDiabetesModel
from MedicalDecisionDiabetesPolicy import MedicalDecisionDiabetesPolicy


def normalize_counter(counter):
    total = sum(counter.values(), 0.0)
    for key in counter:
        counter[key] /= total
    return counter

def main():
    seed = 19783167

    # each time step is 1 month.
    t_stop = int(CONFIG[Config.N])  # number of times we test the drugs
    L = int(CONFIG[Config.L])  # number of samples
    theta_range = np.arange(CONFIG[Config.THETA_START], CONFIG[Config.THETA_END], CONFIG[Config.INCREMENT])

    # dictionaries to store the stats for different values of theta
    theta_obj = {p: [] for p in list(Policy)}
    theta_obj_std = {p: [] for p in list(Policy)}

    output_path = []

    # data structures to accumulate best treatment count
    best_treat = {(p, theta): [] for p in list(Policy) for theta in theta_range}
    best_treat_count_list = {(p, theta): [] for p in list(Policy) for theta in theta_range}
    best_treat_counter_hist = {(p, theta): [] for p in list(Policy) for theta in theta_range}
    best_treat_chosen_hist = {(p, theta): [] for p in list(Policy) for theta in theta_range}

    # data structures to accumulate the decisions
    decision_given_best_treat_list = {(p, theta, s): [] for p in list(Policy) for theta in theta_range for s in list(State)}

    decision_all_list = {(p, theta): [] for p in list(Policy) for theta in theta_range}
    decision_all_counter = {(p, theta): [] for p in list(Policy) for theta in theta_range}

    model = MedicalDecisionDiabetesModel(seed)
    policy = MedicalDecisionDiabetesPolicy(seed)

    for chosen_policy in CONFIG[Config.POLICY]:
        policy_start = time.time()

        for theta in theta_range:
            f_hat = []
            last_state_dict = {x: [0., 0., 0] for x in list(State)}

            for l in range(1, L + 1):
                model_copy = copy(model)

                # sample the truth - the truth is going to be the same for the N experiments in the budget
                model_copy.exog_info_sample_mu()

                # determine the best treatment for the sampled truth
                best_treatment = max(model_copy.mu, key=model_copy.mu.get)
                best_treat[(chosen_policy, theta)].append(best_treatment)
                best_treat_count = 0
                decision_list = []

                # prepare record for output
                mu_output = [model_copy.mu[x] for x in State]
                record_sample_l = [chosen_policy, CONFIG[Config.TRUTH_TYPE], theta, l] + mu_output + [best_treatment]

                # loop over time (N, in notes)
                for n in range(t_stop):
                    # formatting pre-decision state for output
                    state_mu = [model_copy.state[s][0] for s in State]
                    state_sigma = [1.0 / math.sqrt(model_copy.state[s][1]) for s in State]
                    state_n = [model_copy.state[s][2] for s in State]

                    # make decision based on chosen policy
                    decision = policy.decide_drug(chosen_policy, model_copy, theta)
                    decision_list.append(decision)

                    # step forward in time: exog_info is a tuple (w, beta_w, mu)
                    w, beta_w, mu = model_copy.step(decision)
                    best_treat_count += 1 if (decision == best_treatment) else 0

                    # adding record for output
                    record_sample_t = [n] + state_mu + state_sigma + state_n + [
                        decision,            # chosen action
                        w,                   # reduction (W)
                        model_copy.obj,      # accumulated objective
                        1 if (decision == best_treatment) else 0
                    ]
                    output_path.append(record_sample_l + record_sample_t)

                # updating end of experiments stats
                f_hat.append(model_copy.obj)
                for s in State:
                    last_state_dict[s][0] += model_copy.state[s][0]
                    last_state_dict[s][1] += model_copy.state[s][1]
                    last_state_dict[s][2] += model_copy.state[s][2]

                best_treat_count_list[(chosen_policy, theta)].append(best_treat_count)
                decision_given_best_treat_list[(chosen_policy, theta, best_treatment)] += decision_list
                decision_all_list[(chosen_policy, theta)] += decision_list

            # updating end of theta stats
            f_hat_mean = np.mean(np.array(f_hat))
            f_hat_var = np.sum((np.array(f_hat) - f_hat_mean) ** 2) / (L - 1) if L > 1 else 0.0
            theta_obj[chosen_policy].append(f_hat_mean)
            theta_obj_std[chosen_policy].append(np.sqrt(f_hat_var / L) if L > 0 else 0.0)

            print(
                "Finishing policy = {}, Truth_type {} and theta = {:.3f}. F_bar_mean = {:.3f} and F_bar_std = {:.3f}".format(
                    chosen_policy, CONFIG[Config.TRUTH_TYPE], float(theta), f_hat_mean, np.sqrt(f_hat_var / L) if L > 0 else 0.0
                )
            )

            states_avg = {s: [last_state_dict[s][0] / L, last_state_dict[s][1] / L, last_state_dict[s][2] / L] for s in State}
            print("Averages along {} iterations and {} budget trial:".format(L, t_stop))
            for s in State:
                print(
                    "Treatment {}: m_bar {:.2f}, beta_bar {:.2f} and N {}".format(
                        s, states_avg[s][0], states_avg[s][1], int(states_avg[s][2])
                    )
                )

            best_treat_counter = Counter(best_treat[(chosen_policy, theta)])
            best_treat_counter_hist.update({(chosen_policy, theta): best_treat_counter})

            hist, bin_edges = np.histogram(np.array(best_treat_count_list[(chosen_policy, theta)]), t_stop)
            best_treat_chosen_hist.update({(chosen_policy, theta): hist})

            print("Histogram best_treatment")
            print(normalize_counter(best_treat_counter))

            print("Histogram decisions")
            decision_all_counter[(chosen_policy, theta)] = normalize_counter(
                Counter(decision_all_list[(chosen_policy, theta)])
            )
            print(decision_all_counter[(chosen_policy, theta)])

            decision_given_best_treat_dict = {
                s: dict(normalize_counter(Counter(decision_given_best_treat_list[(chosen_policy, theta, s)])))
                for s in State
            }
            decision_df = pd.DataFrame(decision_given_best_treat_dict)
            print(decision_df.head())
            print("\n\n")

        # updating end of policy stats
        policy_end = time.time()
        print("Ending policy {}. Elapsed time {:.3f} secs\n\n\n".format(chosen_policy, policy_end - policy_start))

    # =============================================================================
    #     Generating Plots
    # =============================================================================
    fig1, axsubs = plt.subplots(1, 2)
    fig1.suptitle(
        'Comparison of policies for the Medical Decisions Diabetes Model:\n(N = {}, L = {}, Truth_type = {})'
        .format(t_stop, L, CONFIG[Config.TRUTH_TYPE])
    )

    color_list = ['b', 'g', 'r', 'm']
    for idx, chosen_policy in enumerate(CONFIG[Config.POLICY]):
        axsubs[0].plot(theta_range, theta_obj[chosen_policy], f"{color_list[idx]}o-", label=f"{chosen_policy}")
        axsubs[0].set_title('Mean')
        axsubs[0].legend()
        axsubs[0].set_xlabel('theta')
        axsubs[0].set_ylabel('estimated value for (F_bar)')

        axsubs[1].plot(theta_range, theta_obj_std[chosen_policy], f"{color_list[idx]}+:", label=f"{chosen_policy}")
        axsubs[1].set_title('Std')
        axsubs[1].legend()
        axsubs[1].set_xlabel('theta')
        # axsubs[1].set_ylabel('estimated value for (F_bar)')

    plt.show()
    fig1.savefig('Policy_Comparison_{}.jpg'.format(CONFIG[Config.TRUTH_TYPE]))

    # fig = plt.figure()
    # plt.title('Comparison of policies for the Medical Decisions Diabetes Model: \n (N = {}, L = {}, Truth_type = {} )'.format(t_stop, L, Model.truth_type))
    # color_list = ['b','g','r','m']
    # nPolicies = list(range(len(policy_list)))
    # for policy_chosen,p in zip(policy_list,nPolicies):
    #    plt.plot(theta_range_1, theta_obj[policy_chosen], "{}o-".format(color_list[p]),label = "mean for {}".format(policy_chosen))
    #    if plot_std:
    #        plt.plot(theta_range_1, theta_obj_std[policy_chosen], "{}+:".format(color_list[p]),label = "std for {}".format(policy_chosen))
    # plt.legend()
    # plt.xlabel('theta')
    # plt.ylabel('estimated value (F_bar)')
    # plt.show()
    # fig.savefig('Policy_Comparison_{}.jpg'.format(Model.truth_type))

if __name__ == "__main__":
    main()
