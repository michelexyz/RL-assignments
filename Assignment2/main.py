import time

import numpy as np

MAX_ITEMS = 20
MAX_ORDER = 5  # Policy fixed to replenish up to 5
PI_0 = np.array((20, 20))  # Initial state (irrelevant)
H = np.array([1, 2])  # Holding costs for each item type
K = 5  # Order costs


def expected_costs():
    costs = np.zeros((MAX_ITEMS, MAX_ITEMS))
    for i in range(0, MAX_ITEMS):
        for j in range(0, MAX_ITEMS):
            state = np.array([i, j]) + 1
            costs[i, j] = H @ state + 5 * (1 if np.any(state == 1) else 0)
    return costs.ravel()


def compute_prob_fixed(x1, x2, y1, y2):
    cond1 = x1 - 1 + (4 if x1 == 1 else 0) <= y1 <= x1 + (4 if x1 == 1 else 0)
    cond2 = x2 - 1 + (4 if x2 == 1 else 0) <= y2 <= x2 + (4 if x2 == 1 else 0)
    return 1 / 4 if cond1 and cond2 else 0


def prob_matrix():
    p = np.zeros((MAX_ITEMS**2, MAX_ITEMS**2))
    for i in range(MAX_ITEMS**2):
        for j in range(MAX_ITEMS**2):
            x1, x2 = divmod(i, MAX_ITEMS)
            y1, y2 = divmod(j, MAX_ITEMS)
            x1, x2, y1, y2 = (v + 1 for v in (x1, x2, y1, y2))  # Indices start at 0
            p[i, j] = compute_prob_fixed(x1, x2, y1, y2)
    assert (np.sum(p, axis=1) == 1).all()
    return p


def question_c(n=10_000):
    """Simulation"""
    # The initial state doesn't matter
    pi = np.asarray(PI_0)
    # Probs of selling
    probs = np.random.randint(2, size=(2, n))

    # Initialize total cost and visits
    cost = pi @ H
    visits = np.zeros((MAX_ITEMS, MAX_ITEMS))
    visits[*(pi - 1)] += 1

    # Loop over simulations
    for i in range(n):
        # FIRST order (before any potential sales can take place)
        order = MAX_ORDER - pi if np.any(pi == 1) else 0
        # THEN demand
        pi -= probs[:, i]
        # Order arrives at the end of timestep
        pi += order
        # Update cost
        cost += K * np.any(order) + pi @ H
        # Update visits
        visits[*(pi - 1)] += 1

    return cost / n, visits.ravel() / n


def question_d(n=10_000):
    pi = np.eye(MAX_ITEMS**2)[-1]
    p = prob_matrix()
    for _ in range(n):
        pi = pi @ p
    return pi @ expected_costs(), pi


def question_e(n=10_000, eps=10e-5):
    v = np.zeros((MAX_ITEMS**2))
    r = expected_costs()
    p = prob_matrix()
    for _ in range(n):
        v = r + p @ v
    phi_star = r + p @ v - v
    assert np.ptp(phi_star) < eps
    return (v - v.min(), phi_star[0])


def main():
    # Question c
    c, _ = question_c()
    print(f"Question c) long-run average costs from simulation: {c}")
    # Question d
    c_lim, _ = question_d()
    print(f"Question d) long-run average costs for lim distr.: {c_lim}")
    v, phi_star = question_e()
    print(f"Question e) phi star: {phi_star}")


if __name__ == "__main__":
    s = time.time()
    main()
    e = time.time()
    print(f"Execution time: {e - s} sec.")
