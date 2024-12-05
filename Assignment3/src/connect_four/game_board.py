import random
from enum import IntEnum
from typing import Optional, Set, Tuple

import numpy as np


class PlayerType(IntEnum):
    US = 1
    OPPONENT = 2


class GameBoard:
    grid: np.ndarray

    # TODO
    @property
    def is_finished(self) -> bool: ...  # Check if the matrix `self.grid` is full or not

    # TODO
    @property
    def available_actions(self) -> Set[int]:
        return ...  # Just a set with available actions, like {5, 7} or {5,}

    # TODO
    def __init__(self, nrows: int, ncols: int) -> None:
        self.grid = ...  # Init with the grid shown on the assignment pdf

    def snapshot(self) -> np.ndarray:
        return self.grid.copy()

    # TODO
    def game_result(self) -> int:
        if not self.is_finished:
            return 0  # No matter if game is not done yet
        return ...  # [1, 0, -1] depending on result

    # TODO
    def step(self, action: int, player: int) -> bool:
        action = self.validate_action(action)
        if action is None:
            return self.is_finished  # If action is non valid, just leave it like it is
        ...  # update the `self.grid` according to `action` and `player` type
        return self.is_finished

    # TODO
    def validate_action(self, action: int) -> Optional[int]:
        if ...:  # Check whether `action` column of our `self.grid` has spots left
            return None  # Don't panic if is invalid, let `step` handle a `None`
        return action

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
