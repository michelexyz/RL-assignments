#!/usr/bin/env python3
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def first_problem(params):
    MAX_ITEMS = params["MAX_ITEMS"]
    MAX_T = params["MAX_T"]
    actions = params["actions"]
    p = params["p"]

    p_matrix = np.asarray([p, 1 - p])
    v_matrix = np.zeros(shape=(MAX_ITEMS + 1, MAX_T + 1))
    alpha_matrix = np.ones(shape=v_matrix.shape) * 99

    for t in range(1, MAX_T + 1):
        a = (p * actions) + np.asarray(
            [v_matrix[:-1, t - 1], v_matrix[1:, t - 1]]
        ).T.dot(p_matrix)
        v_matrix[1:, t] = np.max(a, axis=1)
        alpha_matrix[1:, t] = np.argmax(a, axis=1)

    return v_matrix, alpha_matrix


def second_problem(params):
    MAX_ITEMS = params["MAX_ITEMS"]
    MAX_T = params["MAX_T"]
    actions = params["actions"]
    p = params["p"]

    p_matrix = np.asarray([p, 1 - p])
    num_of_actions = len(actions)

    v_matrix = np.zeros(shape=(MAX_ITEMS + 1, num_of_actions, MAX_T + 1))
    alpha_matrix = np.empty(shape=v_matrix.shape)
    alpha_matrix[:] = np.nan

    for t in range(1, MAX_T + 1):
        last_50 = (p[2] * actions[2]) + np.asarray(
            [v_matrix[:-1, 2, t - 1], v_matrix[1:, 2, t - 1]]
        ).T.dot(
            p_matrix[:, 2]
        )  # if the last action is 50 only consider the 50 states in the next step
        v_matrix[1:, 2, t] = last_50
        last_100 = np.column_stack(
            [
                (p[1] * actions[1])
                + np.asarray([v_matrix[:-1, 1, t - 1], v_matrix[1:, 1, t - 1]]).T.dot(
                    p_matrix[:, 1]
                ),
                (p[2] * actions[2])
                + np.asarray([v_matrix[:-1, 2, t - 1], v_matrix[1:, 2, t - 1]]).T.dot(
                    p_matrix[:, 2]
                ),
            ]
        )
        v_matrix[1:, 1, t] = np.max(
            last_100, axis=1
        )  # if the last action is 100 consider both 50 and 100 states in the next step
        last_200 = np.column_stack(
            [
                (p[0] * actions[0])
                + np.asarray([v_matrix[:-1, 0, t - 1], v_matrix[1:, 0, t - 1]]).T.dot(
                    p_matrix[:, 0]
                ),
                (p[1] * actions[1])
                + np.asarray([v_matrix[:-1, 1, t - 1], v_matrix[1:, 1, t - 1]]).T.dot(
                    p_matrix[:, 1]
                ),
                (p[2] * actions[2])
                + np.asarray([v_matrix[:-1, 2, t - 1], v_matrix[1:, 2, t - 1]]).T.dot(
                    p_matrix[:, 2]
                ),
            ]
        )
        v_matrix[1:, 0, t] = np.max(
            last_200, axis=1
        )  # if the last action is 200 consider all states in the next step
        alpha_matrix[1:, 2, t] = 2
        alpha_matrix[1:, 1, t] = np.argmax(last_100, axis=1) + 1
        alpha_matrix[1:, 0, t] = np.argmax(last_200, axis=1)

    return v_matrix, alpha_matrix


def simulate_process(alpha_matrix, params):
    ran_vals = np.random.rand(1000, 500)
    actions = params["actions"]
    p = params["p"]

    def _simulate_process(ran_vals):
        items = 100
        t = 500
        reward = 0
        while items > 0 and t > 0:
            if ran_vals[t - 1] < p[int(alpha_matrix[items, t])]:
                reward += actions[int(alpha_matrix[items, t])]
                items -= 1
                t -= 1
            else:
                t -= 1
        return reward

    return np.asarray([_simulate_process(ran_val) for ran_val in ran_vals])


def plot_optimal_policy_A(alpha_matrix, fname="imgs/policyA.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(alpha_matrix[1:, 1:], cmap=["#F3E457", "#9FD772"], ax=ax, cbar=False)
    ax.set_title(r"Optimal policy $\alpha^*$")
    ax.set_yticks(np.arange(0, 100, 10) + 0.5)  # ticks position
    ax.set_yticklabels(np.concatenate([np.asarray([1]), np.arange(10, 100, 10)]))
    ax.set_xticks(np.arange(0, 500, 50) + 0.5)
    ax.set_xticklabels(np.concat([np.asarray([1]), np.arange(50, 500, 50)]))
    # set the x and y labels
    ax.set_xlabel("Time left ")
    ax.set_ylabel("Items left")
    # create a legend on the right side of the plot
    legend_elements = [
        mpatches.Patch(facecolor="#F3E457", edgecolor="black", label="200"),
        mpatches.Patch(facecolor="#9FD772", edgecolor="black", label="100"),
    ]
    ax.legend(
        handles=legend_elements,
        title="Action",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    dirname = os.path.dirname(os.path.abspath(fname))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig("imgs/policyA.png")


def plot_rewards_hist(v_matrix, rewards, fname="imgs/rewards.png"):
    avg_reward = np.mean(rewards)
    print(f"Simulations average reward: {avg_reward}")
    fig, ax = plt.subplots()
    ax.hist(rewards, bins=20, color="skyblue", edgecolor="black")
    ax.set_title(r"Total rewards under optimal policy $\alpha^*$")
    ax.axvline(avg_reward, color="red", linestyle="dashed", linewidth=1)
    ax.axvline(v_matrix[100, 500], color="green", linestyle="dashed", linewidth=1)
    ax.set_xlabel("Total reward")
    ax.set_ylabel("Frequency")
    ax.legend(["Average Reward", "Expected Reward"])

    dirname = os.path.dirname(os.path.abspath(fname))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(fname)


def plot_optimal_policy_B(alpha_matrix, actions, fname="imgs/policyB.png"):
    legend_elements = [
        mpatches.Patch(facecolor="#F3E457", edgecolor="black", label="200"),
        mpatches.Patch(facecolor="#9FD772", edgecolor="black", label="100"),
        mpatches.Patch(facecolor="#FFA500", edgecolor="black", label="50"),
    ]
    fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for i in range(3):
        if i == 2:
            sns.heatmap(alpha_matrix[1:, i, 1:], cmap=["#FFA500"], ax=ax[i], cbar=False)
        if i == 1:
            sns.heatmap(
                alpha_matrix[1:, i, 1:],
                cmap=["#9FD772", "#FFA500"],
                ax=ax[i],
                cbar=False,
            )
        if i == 0:
            sns.heatmap(
                alpha_matrix[1:, i, 1:],
                cmap=["#F3E457", "#9FD772"],
                ax=ax[i],
                cbar=False,
            )
        ax[i].set_yticks(np.arange(0, 100, 10) + 0.5)
        ax[i].set_yticklabels(np.concatenate([np.asarray([1]), np.arange(10, 100, 10)]))
        ax[i].set_xticks(np.arange(0, 500, 50) + 0.5)
        ax[i].set_xticklabels(np.concatenate([np.asarray([1]), np.arange(50, 500, 50)]))
        ax[i].set_xlabel("Time left")
        ax[i].set_ylabel("Items left")
        ax[i].legend(
            handles=legend_elements,
            title="Action",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        ax[i].set_title(f"Optimal policy when the last price is {actions[i]}")

    dirname = os.path.dirname(os.path.abspath(fname))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(fname)


if __name__ == "__main__":
    np.random.seed(0)

    params = {
        "MAX_ITEMS": 100,
        "MAX_T": 500,
        "actions": np.asarray([200.0, 100.0, 50.0]),
        "p": np.asarray([0.1, 0.5, 0.8]),
    }

    v_matrix, alpha_matrix = first_problem(params)
    print(f"First problem total expected reward: {np.round(v_matrix[100, 500], 3)}")
    plot_optimal_policy_A(alpha_matrix)
    rewards = simulate_process(alpha_matrix, params)
    plot_rewards_hist(v_matrix, rewards)
    v_matrix, alpha_matrix = second_problem(params)
    print(f"Second problem total expected reward: {np.round(v_matrix[100,:, 500], 3)}")
    plot_optimal_policy_B(alpha_matrix, params["actions"], fname="imgs/policyB.png")
