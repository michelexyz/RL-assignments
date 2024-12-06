from enum import IntEnum
from typing import Optional, Set

import numpy as np


class PlayerType(IntEnum):
    US = 1
    OPPONENT = 2


def available_actions(grid) -> Set[int]:
    return set(np.where(grid == 0)[1])


def check_winner(grid: np.ndarray) -> Optional[PlayerType]:

    for player in [PlayerType.US, PlayerType.OPPONENT]:
        # Horizontal check
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1] - 3):
                if np.all(grid[row, col + i] == player for i in range(4)):
                    return player

        # Vertical check
        for col in range(grid.shape[1]):
            for row in range(grid.shape[0] - 3):
                if np.all(grid[row + i, col] == player for i in range(4)):
                    return player

        # Diagonal down (\) check
        for row in range(grid.shape[0] - 3):
            for col in range(grid.shape[1] - 3):
                if np.all(grid[row + i, col + i] == player for i in range(4)):
                    return player

        # Diagonal up (/) check
        for row in range(3, grid.shape[0]):
            for col in range(grid.shape[1] - 3):
                if np.all(grid[row - i, col + i] == player for i in range(4)):
                    return player
    return
