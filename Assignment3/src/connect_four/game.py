import sys
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np
from connect_four.game_board import GameBoard, InvalidActionError
from connect_four.mcts import MCTS
from connect_four.utils import display_plt, display_circles


class GameType(Enum):
    HUMAN_VS_RANDOM = auto()
    MCTS_VS_RANDOM = auto()
    HUMAN_VS_MCTS = auto()


class Game:

    def __init__(
        self,
        game_state: Optional[np.ndarray] = None,
        game_type: GameType = GameType.MCTS_VS_RANDOM,
        mcts_maxiter: int = 1048,
    ):
        self.mcts = MCTS(maxiter=mcts_maxiter)
        self.game_board = (
            GameBoard() if game_state is None else GameBoard.from_grid(game_state)
        )
        self.game_type = game_type

    def parse_input(self, s: str) -> int:
        if s in (":q", "quit", "quit()", "exit", "exit()"):
            sys.exit("Quitting game upon user request ...")
        return int(s)

    def input_action(self) -> int:
        av = self.game_board.available_actions
        while True:
            try:
                action = self.parse_input(
                    input(f"Choose an action from: {[str(a) for a in av]}")
                )
                if action in av:
                    return action
                raise InvalidActionError(f"Invalid action {action}")
            except Exception as ex:
                print(ex)

    def get_qvalues(self) -> Dict[int, str]:
        qvalues = {i: "X" for i in range(self.game_board.ncols)}
        match self.game_type:
            case GameType.HUMAN_VS_MCTS:
                qvalues.update({i: "?" for i in self.game_board.available_actions})
                return qvalues
            case GameType.MCTS_VS_RANDOM:
                qvalues.update(
                    {a: str(round(v, 2)) for a, v in self.mcts.qvalues.items()}
                )
                return qvalues

    def select_best_action(self) -> int:
        # Here is where the tree search is actually done
        # Return the best action (greedy) that can be taken from root
        print("Running tree search to choose action ...")
        qvalues = self.mcts.run(game_state=self.game_board.snapshot())
        print(f"Qvalues: {qvalues}")
        print(f"Choosing {int(max(qvalues, key=qvalues.get))}")
        return int(max(qvalues, key=qvalues.get))

    def first_move(self) -> int:
        """Returns action to be taken depending on game type"""
        match self.game_type:
            case GameType.HUMAN_VS_RANDOM | GameType.HUMAN_VS_MCTS:
                return self.input_action()
            case GameType.MCTS_VS_RANDOM:
                return self.select_best_action()

    def second_move(self) -> Optional[int]:
        """Returns action to be taken for second player, `None` if
        second player is Random"""
        match self.game_type:
            case GameType.HUMAN_VS_MCTS:
                return self.select_best_action()
            case GameType.MCTS_VS_RANDOM | GameType.HUMAN_VS_RANDOM:
                return None  # Return `None` so that we can play randomly

    def play(self, show: bool = True, display_fun=display_circles) -> None:

        while not self.game_board.is_finished:
            self.game_board.play(
                first_action=self.first_move(), second_action=self.second_move()
            )
            if show:
                display_fun(grid=self.game_board.snapshot(), qvalues=self.get_qvalues())

        winner = self.game_board.check_winner()
        print(f"The winner of the game is: {winner if winner else 'DRAW'}")
