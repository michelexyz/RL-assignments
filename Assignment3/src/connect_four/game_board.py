import random
from enum import IntEnum
from typing import Optional, Set, Tuple

import numpy as np


class PlayerType(IntEnum):
    US = 1
    OPPONENT = 2


# Import the convolution function
from scipy.signal import convolve2d

class GameBoard:
    grid: np.ndarray

    @property
    def is_finished(self) -> bool: 
        
        return 0 not in self.grid  # Check if the matrix `self.grid` is full or not

    @property
    def available_actions(self) -> Set[int]:

        free_columns = set(np.where(self.grid == 0)[1])
        return {int(i) for i in free_columns} # not needed if np.int64 type is okay
  
    def __init__(self, nrows: int, ncols: int, inital_state = True) -> None:

        if inital_state:
            # Init with the grid shown on the assignment pdf
            assert nrows == 6 and ncols == 7
            self.grid = np.array(
                [
                    [2, 2, 2, 1, 0, 1, 0],
                    [2, 1, 1, 1, 0, 2, 0],
                    [1, 2, 2, 2, 0, 1, 0],
                    [2, 1, 1, 1, 0, 2, 0],
                    [1, 1, 1, 2, 0, 2, 0],
                    [2, 2, 1, 2, 0, 1, 0],
                ]
            )
        else:
            self.grid = np.zeros((nrows, ncols))
    def snapshot(self) -> np.ndarray:
        return self.grid.copy()

    # TODO test this method more
    def game_result(self) -> Optional[int]: 
        if not self.is_finished:
            return None  # No matter if game is not done yet
        
        # check for horizontal win
        # Define the kernel for convolution-like checks
        kernels = {
            "horizontal": np.array([[1, 1, 1, 1]]),  # Horizontal pattern
            "vertical": np.array([[1], [1], [1], [1]]),  # Vertical pattern
            "diagonal_down": np.eye(4, dtype=int),  # Diagonal down (\)
            "diagonal_up": np.fliplr(np.eye(4, dtype=int)),  # Diagonal up (/)
        }
        winner = None
        for player in [1, 2]:
            # Create a binary grid where the player's pieces are 1 and others are 0
            player_grid = (self.grid == player).astype(int)

            for kernel in kernels.values():
                # Perform convolution between the player's grid and the kernel
                conv_result = convolve2d(player_grid, kernel, mode='valid')

                # If any position in conv_result equals 4, the player has won
                if np.any(conv_result == 4):
                    if winner == 1:
                        raise ValueError("Both players can't win at the same time")
                    winner = player
                    

        # If no winner and the board is full, it's a draw
        return winner

    # TODO
    def step(self, action: int, player: int) -> bool:
        action = self.validate_action(action)
        if action is None:
            return self.is_finished  # If action is non valid, just leave it like it is
        
        row = np.max(np.where(self.grid[:, action] == 0))

        self.grid[row, action] = player
        
        return self.is_finished

    def validate_action(self, action: int) -> Optional[int]:
        if 0 in self.grid[:, action]:
            return action
        else:
            return None

    def play(self, action: int, state: Optional[np.ndarray] = None) -> Tuple[int, bool]:
        # Choose to either start from state or just keep going with current stored state
        # This is good for rollout, where we keep going from where we left
        self.grid = state if state else self.grid

        self.step(action=action, player=PlayerType.US)
        # Opponent turn is part of the transition, so always play his turn
        # if possible (if game is not finished)
        if not self.is_finished:
            self.step(
                action=random.choice(list(self.available_actions)),
                player=PlayerType.OPPONENT,
            )
        return self.game_result(), self.is_finished

    def rollout(self) -> int:
        while not self.is_finished:
            # We left the game in a certain state when calling `play`, start
            # rollllling out from there
            self.play(action=random.choice(list(self.available_actions)))
        return self.game_result()
