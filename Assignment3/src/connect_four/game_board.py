from __future__ import annotations

import random
from enum import IntEnum
from typing import Optional, Set

import numpy as np
from connect_four.utils import (
    PlayerType,
    available_actions,
    check_winner,
    is_game_finished,
)


class InvalidActionError(Exception):
    pass


class GameNotFinishedError(Exception):
    pass


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
        return is_game_finished(self._grid)

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
        self._grid = state.copy()
        return

    def snapshot(self) -> np.ndarray:
        return self._grid.copy()

    def game_result(self) -> GameResult:
        if not self.is_finished:
            raise GameNotFinishedError

        match self.check_winner():
            case PlayerType.US:
                return GameResult.WIN
            case PlayerType.OPPONENT:
                return GameResult.LOSE
            case None:
                return GameResult.DRAW

    def check_winner(self) -> Optional[PlayerType]:
        return check_winner(grid=self._grid)

    def step(self, action: int, player: PlayerType) -> None:
        action = self.validate_action(action)
        row = np.max(np.where(self._grid[:, action] == 0))
        self._grid[row, action] = player

    def validate_action(self, action: int) -> int:
        if 0 in self._grid[:, action]:
            return action
        raise InvalidActionError

    def play(self, first_action: int, second_action: Optional[int] = None) -> None:
        """First move must always be explicit. Second move can be `None`,
        and in this case the second player will play randomly
        """
        # First play
        self.step(action=first_action, player=PlayerType.US)

        # Second play
        second_action = (
            second_action
            if second_action is not None
            else random.choice(list(self.available_actions))
        )
        if not self.is_finished:
            self.step(action=second_action, player=PlayerType.OPPONENT)
        return

    def rollout(self) -> GameResult:
        while not self.is_finished:
            # We left the game in a certain state when calling `play`, start
            # rollllling out from there
            self.play(first_action=random.choice(list(self.available_actions)))
        return self.game_result()
