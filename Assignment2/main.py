import time

import numpy as np

from tqdm import tqdm

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

def compute_prob(x1, x2, y1, y2, t_y1, t_y2):
    """
    Compute the probability of transitioning from state (x1, x2) to state (y1, y2) given the sales topping policy t_y1 and t_y2
    """
    cond1 = x1 - 1 + (t_y1 if x1 == 1 else 0) <= y1 <= x1 + (t_y2 if x1 == 1 else 0)
    cond2 = x2 - 1 + (t_y1 if x2 == 1 else 0) <= y2 <= x2 + (t_y2 if x2 == 1 else 0)
    



def prob_matrix_fixed():
    p = np.zeros((MAX_ITEMS**2, MAX_ITEMS**2))
    for i in range(MAX_ITEMS**2):
        for j in range(MAX_ITEMS**2):
            x1, x2 = divmod(i, MAX_ITEMS)
            y1, y2 = divmod(j, MAX_ITEMS)
            x1, x2, y1, y2 = (v + 1 for v in (x1, x2, y1, y2))  # Indices start at 0
            p[i, j] = compute_prob_fixed(x1, x2, y1, y2)
    assert (np.sum(p, axis=1) == 1).all()
    return p

# the same as above but with with the funcion to compute the probability for the single element as an argument
def prob_matrix( prob_func, *args):
    p = np.zeros((MAX_ITEMS**2, MAX_ITEMS**2))
    for i in range(MAX_ITEMS**2):
        for j in range(MAX_ITEMS**2):
            x1, x2 = divmod(i, MAX_ITEMS)
            y1, y2 = divmod(j, MAX_ITEMS)
            x1, x2, y1, y2 = (v + 1 for v in (x1, x2, y1, y2))  # Indices start at 0
            p[i, j] = prob_func(x1, x2, y1, y2, *args)
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
    p = prob_matrix_fixed()
    for _ in range(n):
        pi = pi @ p
    return pi @ expected_costs(), pi


def question_e(n=10_000, eps=10e-5):
    v = np.zeros((MAX_ITEMS**2))
    r = expected_costs()
    p = prob_matrix_fixed()
    for _ in range(n):
        v = r + p @ v
    phi_star = r + p @ v - v
    assert np.ptp(phi_star) < eps
    return (v - v.min(), phi_star[0])

# Question f: Define the Bellman Equation and solve it using value iteration (minimizing the long-run average costs). Find the optimal policy and report the corresponding long-run average costs.

def compute_p_vector(y1, y2): #compute the probability vector for the single output state y1, y2 after the policy is applied

    p_vector = np.zeros(MAX_ITEMS**2)
    

    p_vector[y1 * MAX_ITEMS + y2] = 1/4
    p_vector[y1 * MAX_ITEMS + y2 - 1] = 1/4
    p_vector[(y1 * MAX_ITEMS-1) + y2 + 1] = 1/4
    p_vector[(y1 * MAX_ITEMS-1) + y2-1] = 1/4

    return p_vector

def compute_best_action(x1, x2, v): #compute the best action for the single state x1, x2 given the value function v

    p_vector = np.zeros(MAX_ITEMS**2)

    best_t_y1 = None
    best_t_y2 = None

    best_value = np.inf

    if x1 > 0 and x2 > 0: # we order only when we are out of stock. # THIS CAN BE REMOVED IF WE WANT CONSIDER THE GENERAL CASE

        p_vector = compute_p_vector(x1, x2)

        expected_value = p_vector @ v

        return None, None, expected_value


    else:

        for t_y1 in range(0,20): # we consider also not ordering anything
            for t_y2 in range(0,20):


                y1, y2 = x1 + t_y1, x2 + t_y2

                if y1 > 19 or y2 > 19: # we can't have more than 20 items

                    continue
                if y1 == 0 or y2 == 0: # we don't consider the case where we sell everything
                        
                        continue
                
                p_vector = compute_p_vector(t_y1, t_y2)

                expected_value = p_vector @ v

                # assert that the value is a scalar or a 1x1 matrix

                assert expected_value.shape == (1,1) or expected_value.shape == (), f"Expected value has shape {expected_value.shape}"

                if expected_value < best_value:
                    best_value = expected_value
                    best_t_y1 = t_y1
                    best_t_y2 = t_y2

    
    return best_t_y1, best_t_y2, best_value

def question_f(n=10_000, eps=10e-5):

    v = np.zeros((MAX_ITEMS**2))
    r = expected_costs()
    #p = prob_matrix_fixed()

    new_v = np.zeros((MAX_ITEMS**2))

    policy = np.zeros((MAX_ITEMS**2, 2))



    for _ in tqdm(range(n), desc="Value iteration"):
        for i in range(MAX_ITEMS**2):

            v = new_v

            

            x1, x2 = divmod(i, MAX_ITEMS)
            t_y1, t_y2, value = compute_best_action(x1, x2, v)

            policy[i] = [t_y1, t_y2]

            new_v[i] = value + r[i]

    return new_v- new_v.min(), new_v-v, policy
 




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
