from __future__ import annotations

import random
from enum import IntEnum
from typing import Optional, Set

import numpy as np
from connect_four.utils import available_actions


class InvalidActionError(Exception):
    pass


class GameNotFinishedError(Exception):
    pass


class PlayerType(IntEnum):
    US = 1
    OPPONENT = 2


class GameResult(IntEnum):
    WIN = 1
    DRAW = 0
    LOSE = -1


class GameBoard:
    nrows: int
    ncols: int
    _grid: np.ndarray

    @property
    def available_actions(self) -> Set[int]:
        return available_actions(self._grid)

    @property
    def is_finished(self) -> bool:
        return 0 not in self._grid

    def __init__(self, nrows: int = 6, ncols: int = 7) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self._grid = np.zeros((nrows, ncols))

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> GameBoard:
        nrows, ncols = grid.shape
        gb = cls(nrows, ncols)
        gb._grid = grid
        return gb

    def set_state(self, state: np.ndarray) -> None:
        assert (self.nrows, self.ncols) == state.shape
        self._grid = state.copy()  # This is indeed needed
        return

    def snapshot(self) -> np.ndarray:
        return self._grid.copy()  # This too

    def game_result(self) -> GameResult:
        if not self.is_finished:
            # This should not happen if we are doing things correctly, so raise
            raise GameNotFinishedError

        match self.check_winner():
            case PlayerType.US:
                return GameResult.WIN
            case PlayerType.OPPONENT:
                return GameResult.LOSE
            case None:
                return GameResult.DRAW

    def check_winner(self) -> Optional[PlayerType]:

        for player in [PlayerType.US, PlayerType.OPPONENT]:
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
        return

    def step(self, action: int, player: int) -> None:
        action = self.validate_action(action)
        row = np.max(np.where(self._grid[:, action] == 0))
        self._grid[row, action] = player

    def validate_action(self, action: int) -> int:
        if 0 in self._grid[:, action]:
            return action
        # This should not happen if we are doing things correctly, so raise
        raise InvalidActionError

    def play(self, action: int) -> None:
        self.step(action=action, player=PlayerType.US)
        # Opponent turn is part of the transition, so always play his turn
        # if possible (if game is not finished)
        if not self.is_finished:
            self.step(
                action=random.choice(list(self.available_actions)),
                player=PlayerType.OPPONENT,
            )
        return

    def rollout(self) -> GameResult:
        while not self.is_finished:
            # We left the game in a certain state when calling `play`, start
            # rollllling out from there
            self.play(action=random.choice(list(self.available_actions)))
        return self.game_result()
