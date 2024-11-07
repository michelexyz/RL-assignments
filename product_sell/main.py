#!/usr/bin/env python3

import numpy as np


def main():
    MAX_ITEMS = 100
    MAX_T = 10

    actions = np.asarray([200.0, 100.0, 50.0])
    p = np.asarray([0.1, 0.5, 0.8])
    p_matrix = np.asarray([p, 1 - p])
    v_matrix = np.zeros(shape=(MAX_ITEMS + 1, MAX_T + 1))

    for t in range(1, MAX_T + 1):
        v_matrix[1:, t] = np.max(
            (p * actions)
            + np.asarray([v_matrix[:-1, t - 1], v_matrix[1:, t - 1]]).T.dot(p_matrix),
            axis=1,
        )

    return v_matrix


if __name__ == "__main__":
    v = main()
    print(v)
