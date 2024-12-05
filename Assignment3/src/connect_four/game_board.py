import random
from enum import IntEnum
from typing import Optional, Set, Tuple

import numpy as np
from connect_four.utils import available_actions


class PlayerType(IntEnum):
    US = 1
    OPPONENT = 2


class GameBoard:
    grid: np.ndarray
    nrows: int
    ncols: int

    @property
    def available_actions(self) -> Set[int]:
        return available_actions(self._grid)

    @property
    def is_finished(self) -> bool:
        return 0 not in self._grid
        # return False if self.available_actions else True

    def __init__(self, nrows: int = 6, ncols: int = 7) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self._grid = np.zeros((nrows, ncols))

    def set_state(self, state: np.ndarray) -> None:
        assert (self.nrows, self.ncols) == state.shape
        self._grid = state.copy()

    def snapshot(self) -> np.ndarray:
        return self._grid.copy()

    def game_result(self) -> int:
        if not self.is_finished:
            return 0  # A cero is fine, we are not backpropagating anyways.
        match self.check_winner():
            case PlayerType.US:
                return 1
            case PlayerType.OPPONENT:
                return -1
            case None:
                return 0

    def check_winner(self) -> Optional[int]:

        for player in [1, 2]:
            # Horizontal check
            for row in range(self._grid.shape[0]):
                for col in range(self._grid.shape[1] - 3):
                    if all(self._grid[row, col + i] == player for i in range(4)):
                        return player

            # Vertical check
            for col in range(self._grid.shape[1]):
                for row in range(self._grid.shape[0] - 3):
                    if all(self._grid[row + i, col] == player for i in range(4)):
                        return player

            # Diagonal down (\) check
            for row in range(self._grid.shape[0] - 3):
                for col in range(self._grid.shape[1] - 3):
                    if all(self._grid[row + i, col + i] == player for i in range(4)):
                        return player

            # Diagonal up (/) check
            for row in range(3, self._grid.shape[0]):
                for col in range(self._grid.shape[1] - 3):
                    if all(self._grid[row - i, col + i] == player for i in range(4)):
                        return player
        return None

    #     def check_winner(self):
    # # Check horizontal, vertical and diagonal lines for a win
    # for row in range(self.nrows):
    # for col in range(self.ncols):
    # if self._grid[row][col] != 0:
    # if (
    # self.check_line(self._grid, row, col, 1, 0)
    # or self.check_line(row, col, 0, 1)
    # or self.check_line(row, col, 1, 1)
    # or self.check_line(row, col, 1, -1)
    # ):
    # return self._grid[row][col]
    # return None

    # def check_line(self, start_row, start_col, d_row, d_col):
    # # Check a line of 4 pieces starting from (start_row, start_col) in direction (d_row, d_col)
    # for i in range(1, 4):
    # r = start_row + d_row * i
    # c = start_col + d_col * i
    # if (
    # not (0 <= r < 6 and 0 <= c < 7)
    # or self._grid[r][c] != self._grid[start_row][start_col]
    # ):
    # return False
    # return True

    # # TODO test this method more
    #  def check_winner(self) -> Optional[PlayerType]:
    # # check for horizontal win
    # # Define the kernel for convolution-like checks
    # kernels = {
    # "horizontal": np.array([[1, 1, 1, 1]]),  # Horizontal pattern
    # "vertical": np.array([[1], [1], [1], [1]]),  # Vertical pattern
    # "diagonal_down": np.eye(4, dtype=int),  # Diagonal down (\)
    # "diagonal_up": np.fliplr(np.eye(4, dtype=int)),  # Diagonal up (/)
    # }
    # winner = None
    # for player in [1, 2]:
    # # Create a binary grid where the player's pieces are 1 and others are 0
    # player_grid = (self._grid == player).astype(int)

    # for kernel in kernels.values():
    # # Perform convolution between the player's grid and the kernel
    # conv_result = convolve2d(player_grid, kernel, mode="valid")

    # # If any position in conv_result equals 4, the player has won
    # if np.any(conv_result == 4):
    # if winner == 1:
    # print(self._grid)
    # raise ValueError("Both players can't win at the same time")
    # winner = player

    # # If no winner and the board is full, it's a draw
    # return winner

    def step(self, action: int, player: int) -> bool:
        action = self.validate_action(action)
        if action is None:
            return self.is_finished  # If action is non valid, just leave it like it is

        row = np.max(np.where(self._grid[:, action] == 0))

        self._grid[row, action] = player

        return self.is_finished

    def validate_action(self, action: int) -> Optional[int]:
        if 0 in self._grid[:, action]:
            return action
        return None

    def play(self, action: int) -> Tuple[int, bool]:
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
