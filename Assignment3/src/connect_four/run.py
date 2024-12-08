import numpy as np
from connect_four.game import Game
from connect_four.utils import display_ascii

game_init = np.array(
    [
        [2, 2, 2, 1, 0, 1, 0],
        [2, 1, 1, 1, 0, 2, 0],
        [1, 2, 2, 2, 0, 1, 0],
        [2, 1, 1, 1, 0, 2, 0],
        [1, 1, 1, 2, 0, 2, 0],
        [2, 2, 1, 2, 0, 1, 0],
    ]
)


def run() -> None:
    game = Game(game_state=game_init, mcts_maxiter=1000)
    game.play(display_fun=display_ascii)
