import sys
from enum import Enum, auto
from typing import Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from connect_four.game_board import GameBoard, InvalidActionError
from connect_four.mcts import MCTS


def display(grid: np.ndarray, qvalues: Dict[int, str]) -> None:
    _, ax = plt.subplots(figsize=[7, 6])
    cmap = mcolors.ListedColormap(["white", "green", "blue"])
    norm = mcolors.BoundaryNorm([-1, 0.5, 1.5, 2.5], cmap.N)
    ax.matshow(grid, cmap=cmap, norm=norm)

    for x in range(8):
        ax.plot([x - 0.5, x - 0.5], [-0.5, 5.5], "k")
    for y in range(7):
        ax.plot([-0.5, 6.5], [y - 0.5, y - 0.5], "k")

    for a, value in qvalues.items():
        ax.text(
            a,
            -1,
            str(value),
            ha="center",
            va="center",
            fontsize=20,
            color="black",
        )

    ax.set_axis_off()
    plt.show()
    plt.close()


class GameType(Enum):
    HUMAN_VS_RANDOM = auto()
    MCTS_VS_RANDOM = auto()
    HUMAN_VS_MCTS = auto()


class Game:

    def __init__(
        self,
        game_state: Optional[np.ndarray] = None,
        game_type: GameType = GameType.MCTS_VS_RANDOM,
    ):
        self.mcts = MCTS()
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
        print("Running tree search to choose action ...")
        qvalues = self.mcts.run(game_state=self.game_board.snapshot(), maxiter=1048)
        print(f"Qvalues: {qvalues}")
        print(f"Choosing {int(max(qvalues, key=qvalues.get))}")
        return int(max(qvalues, key=qvalues.get))  # this is greedy already

    def first_move(self) -> int:
        match self.game_type:
            case GameType.HUMAN_VS_RANDOM | GameType.HUMAN_VS_MCTS:
                return self.input_action()
            case GameType.MCTS_VS_RANDOM:
                return self.select_best_action()

    def second_move(self) -> Optional[int]:
        match self.game_type:
            case GameType.HUMAN_VS_MCTS:
                return self.select_best_action()
            case GameType.MCTS_VS_RANDOM | GameType.HUMAN_VS_RANDOM:
                return None

    def play(self, show: bool = True) -> None:

        while not self.game_board.is_finished:
            self.game_board.play(
                first_action=self.first_move(), second_action=self.second_move()
            )
            if show:
                display(grid=self.game_board.snapshot(), qvalues=self.get_qvalues())

        winner = self.game_board.check_winner()
        print(f"The winner of the game is: {winner if winner else 'DRAW'}")


# def game_play(
# game_type: GameType = GameType.HUMAN_VS_RANDOM, empty_start: bool = False
# ):

# display_actions = [
# {"index": 0, "val": "X"},
# {"index": 1, "val": "X"},
# {"index": 2, "val": "X"},
# {"index": 3, "val": "X"},
# {"index": 4, "val": "X"},
# {"index": 5, "val": "X"},
# {"index": 6, "val": "X"},
# ]

# if empty_start:

# initial_game = GameBoard()
# mcts = MCTS(game_state=initial_game.snapshot())

# else:
# mcts = MCTS()

# initial_game = mcts.game

# game = GameBoard.from_grid(initial_game.snapshot())

# done = False
# while not done:
# print("Thinking...")
# mcts.run(maxiter=1048)

# best_action, best_mean, all_actions = mcts.best_action_value()

# for from_action, mean, n_visits in all_actions:
# display_actions[from_action]["val"] = round(mean, 2)

# display(game.snapshot(), display_actions)
# if game_type == GameType.HUMAN_VS_RANDOM:

# try:
# act = int(
# input(
# f"Choose an available column: game.available_actions: {game.available_actions}"
# )
# )

# if act not in game.available_actions:
# raise ValueError
# except ValueError:
# print(f"{act} is Invalid input. Please enter a number between 0 and 6.")
# else:  # mcts as green player
# act = best_action

# if game_type in [
# GameType.MCTS_VS_RANDOM,
# GameType.HUMAN_VS_RANDOM,
# ]:  # if we are playing against random agent we step for both players
# game.play(act)

# elif game_type == GameType.MCTS_VS_HUMAN:

# game.step(act, PlayerType.US)

# for action in range(len(display_actions)):
# if action not in game.available_actions:

# display_actions[action]["val"] = "X"
# else:
# display_actions[action]["val"] = "?"

# display(game.snapshot(), display_actions)
# if not game.is_finished:

# try:

# opponent_act = int(
# input(
# f"Choose an available column: game.available_actions: {game.available_actions}"
# )
# )

# if opponent_act not in game.available_actions:
# print(
# f"{opponent_act} is Invalid input. Please enter a number between 0 and 6."
# )

# raise ValueError

# except ValueError:
# print("Invalid input. Please enter a number between 0 and 6.")
# continue

# game.step(opponent_act, PlayerType.OPPONENT)

# if game.is_finished:  # if finished show results

# reward = game.game_result()

# match reward:
# case 1:
# print("Green wins")
# case 0:
# print("Blue wins")
# case -1:
# print("Draw")

# for action in range(len(display_actions)):

# display_actions[action]["val"] = "X"

# display(game.snapshot(), display_actions)

# done = True
# else:
# mcts = MCTS(game_state=game.snapshot())


# def display(board, values):
# fig, ax = plt.subplots(figsize=[7, 6])
# cmap = mcolors.ListedColormap(["white", "green", "blue"])
# norm = mcolors.BoundaryNorm([-1, 0.5, 1.5, 2.5], cmap.N)
# ax.matshow(board, cmap=cmap, norm=norm)

# for x in range(8):
# ax.plot([x - 0.5, x - 0.5], [-0.5, 5.5], "k")
# for y in range(7):
# ax.plot([-0.5, 6.5], [y - 0.5, y - 0.5], "k")

# for v in values:
# ax.text(
# v["index"],
# -1,
# str(v["val"]),
# ha="center",
# va="center",
# fontsize=20,
# color="black",
# )

# ax.set_axis_off()
# plt.show()
# plt.close()
